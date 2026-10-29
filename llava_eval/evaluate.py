"""
Unified evaluation entry.

Two independent paths coexist:

* baseline path (no ``--method``): the original free-generation evaluation for
  v0-v3, cotm_v3, 1shot, transformers models, etc.
* active decision methods (``--method``): only ``default_v2`` and
  ``real_crop_v2``.
"""

import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLaVA-NeXT")
)

import argparse
import json

import jsonlines
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

from evaluation_config import *
from simple_model_worker import SimpleModelWrapper, TransformersModelWrapper
from methods import (
    DEFAULT_V2_CFG,
    REAL_CROP_V2_CFG,
    MethodConfig,
    run_default_v2,
    run_real_crop_v2,
)


def _generic_v0_prompt(subset):
    name = str(subset or "object").replace("_", " ")
    return (
        f"Are there any defects on the {name} in this image?\n"
        "A. Yes\n"
        "B. No\n"
        "Answer with the option's letter from the given choices directly."
    )


def _resolve_prompt(prompt_src, subset, fallback_generic=False):
    if isinstance(prompt_src, dict):
        stem = str(subset or "").strip().lower()
        for key, value in prompt_src.items():
            if str(key).strip().lower() == stem:
                return value
        for key in ("default", "object"):
            if key in prompt_src:
                return prompt_src[key]
    elif isinstance(prompt_src, str) and prompt_src:
        return prompt_src
    if fallback_generic:
        return _generic_v0_prompt(subset)
    raise ValueError(f"No prompt for subset={subset!r}")


def make_convs(convs, img_num=0):
    data_convs = []
    roles = ["human", "gpt"]
    for i, c in enumerate(convs):
        if i == 0:
            if "<image>" not in c:
                c = "".join(["<image>\n"] * img_num) + c
        data_convs.append({"from": roles[i % 2], "value": c})
    return data_convs


def make_train_data_json(data_id, img_ps, convs, img_p=None, masks=None, bboxes=None, **kwargs):
    data = {}
    data["id"] = data_id
    if isinstance(img_ps, str):
        img_ps = [img_ps]
    if len(img_ps) > 1:
        data["src"] = img_ps
    data["image"] = img_ps[0] if img_p is None else img_p
    data["image_num"] = len(img_ps)
    data["conversations"] = make_convs(convs, len(img_ps))
    if masks is not None:
        if isinstance(masks, list) and len(masks) == 1:
            masks = masks[0]
        data["mask"] = masks
    if bboxes is not None:
        data["bbox"] = bboxes
    data.update(kwargs)
    return data


def data_read(data_path):
    data_dir = os.path.dirname(data_path)
    dataset_name = os.path.basename(data_dir)
    all_datas = []
    if data_path.endswith(".jsonl"):
        with jsonlines.open(data_path) as f:
            jsonl_datas = list(f)
        for i in jsonl_datas:
            question_id = i["question_id"]
            img_p = os.path.join("imgs", i["image"])
            text = i["text"]
            ref = i.get("reference", None)
            if "sub_dataset" in i:
                subset = i["sub_dataset"]
            else:
                default_subset = i["origin_path"].split("/")[0]
                if default_subset == dataset_name:
                    default_subset = i["origin_path"].split("/")[1]
                subset = i.get("subset", default_subset)
            gt = i.get("gt", None)
            origin_path = i["origin_path"]
            mask_p = i.get("mask", None)
            if mask_p is not None:
                mask_p = os.path.join(mask_p)
            bbox = i.get("bbox", None)
            musc_scores = i.get("musc_scores", None)
            if ref is not None:
                ref = os.path.join("mvtec", ref)
                img_ps = [img_p, ref]
            else:
                img_ps = [img_p]
            all_datas.append(
                make_train_data_json(
                    question_id,
                    img_ps,
                    [text, ""],
                    img_ps[0],
                    mask_p,
                    bbox,
                    subset=subset,
                    gt=gt,
                    origin_path=origin_path,
                    musc_scores=musc_scores,
                )
            )
    else:
        with open(data_path) as f:
            all_datas = json.load(f)
        for i in all_datas:
            if "subset" not in i:
                i["subset"] = i["image"].split("/")[1]
            if "gt" not in i:
                i["gt"] = 0 if "/good/" in i["image"] else 1
            if "origin_path" not in i:
                i["origin_path"] = i["image"]
            if "bbox" not in i:
                i["bbox"] = None
            if "mask" not in i:
                i["mask"] = None
    return all_datas


def data_write(ans, out_path, format="json", mode="w"):
    ans_dir = os.path.dirname(out_path)
    if ans_dir:
        os.makedirs(ans_dir, exist_ok=True)
    if format == "jsonl":
        out_path = out_path.replace(".json", ".jsonl")
        with jsonlines.open(out_path, mode) as f:
            f.write(ans)
    else:
        out_path = out_path.replace(".jsonl", ".json")
        with open(out_path, mode) as f:
            json.dump(ans, f, indent=4)


def read_mask(mask_p):
    if mask_p.endswith((".png", ".jpg", ".PNG", ".JPG")):
        return Image.open(mask_p).convert("L")
    if mask_p.endswith((".npz", ".npy")):
        mask = np.load(mask_p)
        if isinstance(mask, dict):
            mask = mask["anomaly_map"]
        return mask
    raise ValueError(f"Unsupported mask file: {mask_p}")


def _output_path(data_dir, path):
    return path if os.path.isabs(path) else os.path.join(data_dir, path)


def _filter_datas(all_datas, subsets, start, limit):
    eval_datas = []
    for d in all_datas[start:]:
        if subsets is not None and d["subset"] not in subsets:
            continue
        eval_datas.append(d)
    if limit is not None:
        eval_datas = eval_datas[:limit]
    return eval_datas


def _load_worker(args):
    if args.model_type == "triad":
        return SimpleModelWrapper(
            args.model_path,
            args.model_base,
            args.model_name,
            "cuda",
            args.use_flash_attn,
            image_aspect_ratio=args.overwrite_image_aspect_ratio,
        )
    if args.model_type == "transformers":
        return TransformersModelWrapper(args.model_path, device="cuda")
    if args.model_type == "sglang":
        return None
    raise ValueError(f"unknown model_type: {args.model_type}")


def _load_ecpr(args):
    if not getattr(args, "ecpr_sidecar", None):
        return None, None
    import importlib.util

    slots = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "LLaVA-NeXT",
        "llava",
        "train",
        "ecpr_slots.py",
    )
    spec = importlib.util.spec_from_file_location("ecpr_slots", slots)
    ecpr = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(ecpr)
    index = ecpr.load_sidecar_index(args.ecpr_sidecar)
    print(f"ECPR sidecar loaded: {args.ecpr_sidecar} ({len(index)} keys)")
    return index, ecpr


def run_baseline(args, data_dir, question_path, answer_path):
    """Original free-generation path. No requery / no decision methods."""
    if not args.text_src:
        raise ValueError("--text_src is required for the baseline path.")

    ds_name = args.ds
    all_datas = data_read(question_path)
    subsets = None if args.subset == "all" else args.subset.split(",")

    qs_tmp = None
    prompt_version = None
    if args.text_src.startswith("config:"):
        prompt_version = args.text_src.replace("config:", "", 1)
        assert ds_name in SUPPORTED_QSET, f"{args.ds} not in eval config"
        assert prompt_version in SUPPORTED_QSET[ds_name], (
            f"{args.ds} has no prompt version {prompt_version} in eval config"
        )
        qs_tmp = SUPPORTED_QSET[ds_name][prompt_version]
        print(f"using prompt {prompt_version} for {ds_name} in eval config")
    elif args.text_src.startswith("input:"):
        qs_tmp = args.text_src.replace("input:", "", 1)
        print(f"using input prompt {qs_tmp} as question")
    elif args.text_src.startswith("dataset"):
        print("using prompt from eval data")
    else:
        raise ValueError(f"unknown text_src: {args.text_src}")

    worker = _load_worker(args)
    ecpr_index, _ecpr = _load_ecpr(args)

    def _eval_one(d):
        subset = d["subset"]
        if isinstance(qs_tmp, dict):
            if subset not in qs_tmp:
                if "default" in qs_tmp:
                    qs = qs_tmp["default"]
                elif prompt_version == "v0":
                    print(f"WARNING: subset {subset} not in v0 prompts; using generic v0")
                    qs = _generic_v0_prompt(subset)
                else:
                    raise ValueError(
                        f"{subset} in {ds_name} version {prompt_version} not in eval config"
                    )
            else:
                qs = qs_tmp[subset]
        else:
            qs = qs_tmp

        if "src" in d:
            img_p = d["src"]
        elif "image" in d and d["image"] is not None:
            img_p = d["image"]
            if isinstance(img_p, str):
                img_p = [img_p]
        else:
            img_p = []

        imgs = [Image.open(os.path.join(data_dir, i)).convert("RGB") for i in img_p]
        img_num = len(imgs) if args.image_num is None else args.image_num
        if img_num < len(imgs):
            imgs = imgs[:img_num]
        elif img_num > len(imgs):
            img_e = Image.new("RGB", imgs[0].size, (0, 0, 0))
            imgs = imgs + [img_e] * (img_num - len(imgs))

        img_q = imgs[0]
        img_r = imgs[1:]
        if args.ref_mode == "all_empty":
            img_r = [Image.new("RGB", i.size, (0, 0, 0)) for i in img_r]
        elif args.ref_mode == "copy_query":
            img_r = [img_q] * (img_num - 1)
        elif args.ref_mode != "default":
            raise ValueError(f"unknown ref_mode {args.ref_mode}")
        imgs = [img_q] + img_r

        if "mask" in d and d["mask"] is not None:
            mask_p = d["mask"]
            if isinstance(mask_p, str):
                mask_p = [mask_p]
            masks = [read_mask(os.path.join(data_dir, i)) for i in mask_p]
        else:
            masks = None
        mask_o = masks if masks is not None else []
        mask_numr = len(imgs) - len(mask_o)
        if mask_numr > 0:
            mask_o = mask_o + [None] * mask_numr
        elif mask_numr < 0:
            mask_o = mask_o[:mask_numr]

        if args.mask_mode == "all_none":
            masks = None
        elif args.mask_mode == "all_empty":
            masks = []
            for img in imgs:
                img_np = np.array(img)
                masks.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
        elif args.mask_mode == "empty_when_missing":
            masks = []
            for idx, img in enumerate(imgs):
                if mask_o[idx] is None:
                    img_np = np.array(img)
                    masks.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
                else:
                    masks.append(mask_o[idx])
        elif args.mask_mode == "default":
            masks = mask_o
        else:
            raise ValueError(f"unknown mask_mode {args.mask_mode}")

        if not masks:
            masks = [None] * len(imgs)
        mask_q = masks[0]
        mask_r = masks[1:]
        mask_rnum = len(mask_r)
        if args.ref_mask_mode == "none_for_ref":
            mask_r = [None] * mask_rnum
        elif args.ref_mask_mode == "empty_for_ref":
            mask_r = []
            for img in imgs[1:]:
                img_np = np.array(img)
                mask_r.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
        elif args.ref_mask_mode == "copy_query_mask":
            mask_r = [mask_q] * mask_rnum
        elif args.ref_mask_mode == "default":
            mask_r = []
            for idx, img in enumerate(imgs[1:], start=1):
                mask_r.append(mask_q if mask_o[idx] is None else mask_o[idx])
        elif args.ref_mask_mode != "follow_mask_mode":
            raise ValueError(f"unknown ref_mask_mode {args.ref_mask_mode}")
        masks = [mask_q] + mask_r
        if masks is not None and all(m is None for m in masks):
            masks = None

        boxes_list = d["bbox"] if "bbox" in d and d["bbox"] is not None else None

        if ecpr_index is not None:
            img_key = None
            if "origin_path" in d and d["origin_path"]:
                img_key = d["origin_path"]
            elif "image" in d and d["image"]:
                img_key = d["image"] if isinstance(d["image"], str) else d["image"][0]
            elif "src" in d and d["src"]:
                img_key = d["src"] if isinstance(d["src"], str) else d["src"][0]
            if img_key:
                ecpr_boxes = _ecpr.apply_sidecar_boxes(img_key, ecpr_index)
                if ecpr_boxes is not None:
                    boxes_list = [ecpr_boxes]
                    masks = [None] * len(imgs) if masks is not None else None

        sample_id = d.get("id") or d.get("origin_path") or d.get("image")
        if isinstance(sample_id, (list, tuple)):
            sample_id = sample_id[0] if sample_id else None
        inputs = {
            "text": qs,
            "temperature": args.temperature,
            "top_p": 0.7,
            "max_new_tokens": args.max_new_tokens,
            "use_pad_when_db": args.use_pad_when_db,
            "images": imgs,
            "masks": masks,
            "boxes_list": boxes_list,
            "subset": subset,
            "sample_id": sample_id,
        }

        assert worker is not None
        ans = deepcopy(d)
        if qs is None:
            qs = d["conversations"]
        if isinstance(qs, list):
            if args.qa_mode == "single_qa":
                convs = worker.multi_qa(inputs, qs, forget_state=True)
            elif args.qa_mode == "default":
                convs = worker.multi_qa(inputs, qs)
            else:
                raise ValueError(f"unknown qa_mode {args.qa_mode}")
            ans["conversations"] = convs
        elif isinstance(qs, str):
            _, out = worker.single_qa(inputs)
            ans["conversations"] = make_convs([qs, out])
        else:
            raise ValueError(f"unknown qs type {type(qs)}")
        return ans

    eval_datas = _filter_datas(all_datas, subsets, args.start, args.limit)
    all_ans = []
    for d in tqdm(eval_datas):
        ans = _eval_one(d)
        data_write(ans, answer_path, "jsonl", "a")
        all_ans.append(ans)

    data_write(all_ans, answer_path, "json", "w")
    jsonl_tmp = answer_path.replace(".json", ".jsonl")
    if os.path.isfile(jsonl_tmp):
        os.remove(jsonl_tmp)
    return all_ans


def _prepare_sample(d, data_dir):
    subset = d["subset"]
    img_p = d.get("src") or d.get("image")
    if isinstance(img_p, str):
        img_p = [img_p]
    if not img_p:
        raise ValueError(f"No image for sample {d.get('id')}")
    imgs = [Image.open(os.path.join(data_dir, img_p[0])).convert("RGB")]

    mask_p = d.get("mask")
    if mask_p:
        if isinstance(mask_p, list):
            mask_p = mask_p[0]
        masks = [read_mask(os.path.join(data_dir, mask_p))]
    else:
        masks = None

    return {
        "images": imgs,
        "masks": masks,
        "boxes_list": d.get("bbox"),
        "subset": subset,
        "sample_id": d.get("id") or d.get("origin_path") or img_p[0],
    }


def run_active_method(args, data_dir, question_path, answer_path):
    if args.ds != "mvtec":
        raise ValueError("--method currently supports only --ds mvtec.")
    if args.model_type != "triad":
        raise ValueError("--method currently supports only --model-type triad.")
    if args.image_num is not None and int(args.image_num) != 1:
        raise ValueError("--method requires --image_num 1.")

    all_datas = data_read(question_path)
    subsets = None if args.subset == "all" else args.subset.split(",")
    eval_datas = _filter_datas(all_datas, subsets, args.start, args.limit)
    worker = _load_worker(args)

    cfg = DEFAULT_V2_CFG if args.method == "default_v2" else REAL_CROP_V2_CFG
    cfg.band_lo = args.band_lo
    cfg.band_hi = args.band_hi
    cfg.tau_ev = args.tau_ev
    cfg.delta = args.delta
    cfg.max_subq = args.max_subq
    if args.method == "real_crop_v2":
        cfg.cotm_max_new_tokens = args.cotm_max_new_tokens
        cfg.crop_layers = args.crop_layers
        cfg.crop_keep_frac = args.crop_keep_frac
        cfg.crop_alpha = args.crop_alpha
        cfg.crop_beta = args.crop_beta
        cfg.crop_tau_t = args.crop_tau_t
        cfg.crop_tau_i = args.crop_tau_i
        cfg.crop_view = args.crop_view
        cfg.crop_thr_q = args.crop_thr_q
        cfg.crop_expand = args.crop_expand
        cfg.crop_side_lo = args.crop_side_lo
        cfg.crop_side_hi = args.crop_side_hi
        cfg.crop_conc_floor = args.crop_conc_floor
        cfg.crop_box_expand = args.crop_box_expand
        cfg.crop_box_min_side = args.crop_box_min_side
        cfg.crop_overlay_dir = args.crop_overlay_dir
        cfg.crop_dir = args.crop_dir

    all_ans = []
    for d in tqdm(eval_datas, desc=args.method):
        sample = _prepare_sample(d, data_dir)
        subset = sample["subset"]
        sample["v0_prompt"] = _resolve_prompt(
            SUPPORTED_QSET["mvtec"]["v0"], subset, fallback_generic=True
        )

        if args.method == "default_v2":
            decision = run_default_v2(worker, sample, cfg)
        else:
            sample["cotm_prompt"] = _resolve_prompt(
                SUPPORTED_QSET["mvtec"]["cotm_v3"], subset, fallback_generic=False
            )
            decision = run_real_crop_v2(worker, sample, cfg)

        ans = deepcopy(d)
        ans["conversations"] = make_convs([sample["v0_prompt"], decision.letter])
        ans["method"] = args.method
        ans["pred_letter"] = decision.letter
        ans["ab_decision"] = decision.to_dict()
        data_write(ans, answer_path, "jsonl", "a")
        all_ans.append(ans)

    data_write(all_ans, answer_path, "json", "w")
    jsonl_tmp = answer_path.replace(".json", ".jsonl")
    if os.path.isfile(jsonl_tmp):
        os.remove(jsonl_tmp)
    return all_ans


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-type", type=str, default="triad", choices=["triad", "transformers", "sglang"])
    parser.add_argument("--use-flash-attn", action="store_true")

    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./evaluate/")
    parser.add_argument("--text_src", type=str, default=None)
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--from", type=str, default="question.jsonl", dest="question_file")
    parser.add_argument("--to", type=str, default="answer.json", dest="answer_file")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--mask_mode", type=str, default="default", choices=["all_none", "all_empty", "empty_when_missing", "default"])
    parser.add_argument("--ref_mode", type=str, default="default", choices=["all_empty", "copy_query", "default"])
    parser.add_argument("--ref_mask_mode", type=str, default="default", choices=["none_for_ref", "empty_for_ref", "copy_query_mask", "default", "follow_mask_mode"])
    parser.add_argument("--qa_mode", type=str, default="default", choices=["single_qa", "default"])
    parser.add_argument("--image_num", type=int, default=None)
    parser.add_argument("--overwrite_image_aspect_ratio", type=str, default=None)
    parser.add_argument("--use_pad_when_db", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ecpr_sidecar", type=str, default=None)

    parser.add_argument("--method", type=str, default=None, choices=["default_v2", "real_crop_v2"])
    parser.add_argument("--band_lo", type=float, default=0.4)
    parser.add_argument("--band_hi", type=float, default=0.7)
    parser.add_argument("--tau_ev", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--max_subq", type=int, default=3)
    parser.add_argument("--cotm_max_new_tokens", type=int, default=512)

    parser.add_argument("--crop_layers", type=str, default="midband")
    parser.add_argument("--crop_keep_frac", type=float, default=0.5)
    parser.add_argument("--crop_alpha", type=float, default=0.6)
    parser.add_argument("--crop_beta", type=float, default=0.4)
    parser.add_argument("--crop_tau_t", type=float, default=0.0)
    parser.add_argument("--crop_tau_i", type=float, default=0.0)
    parser.add_argument("--crop_view", type=str, default="anyres", choices=["anyres", "base"])
    parser.add_argument("--crop_thr_q", type=float, default=0.90)
    parser.add_argument("--crop_expand", type=float, default=1.3)
    parser.add_argument("--crop_side_lo", type=float, default=0.06)
    parser.add_argument("--crop_side_hi", type=float, default=0.45)
    parser.add_argument("--crop_conc_floor", type=float, default=0.30)
    parser.add_argument("--crop_box_expand", type=str, default="none", choices=["none", "egroi"])
    parser.add_argument("--crop_box_min_side", type=int, default=384)
    parser.add_argument("--crop_overlay_dir", type=str, default=None)
    parser.add_argument("--crop_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds_name = args.ds
    data_dir = args.data_path
    if data_dir.endswith("/") and not data_dir.endswith(ds_name + "/"):
        data_dir = os.path.join(data_dir, ds_name)
    question_path = os.path.join(data_dir, args.question_file)
    answer_path = _output_path(data_dir, args.answer_file)

    print(
        f"eval ds={ds_name} data={question_path} out={answer_path} "
        f"method={args.method or 'baseline'} temp={args.temperature}"
    )

    if args.method:
        run_active_method(args, data_dir, question_path, answer_path)
    else:
        run_baseline(args, data_dir, question_path, answer_path)


if __name__ == "__main__":
    main()
