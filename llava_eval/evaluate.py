"""
unified evaluate scripts for triad and transformers models on IAD benchmarks
"""
import sys
sys.path.append("./LLaVA-NeXT/")

import json
import jsonlines
import time
import math
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse

from evaluation_config import *

from PIL import Image
from copy import deepcopy
from simple_model_worker import SimpleModelWrapper,TransformersModelWrapper

def make_convs(convs,img_num=0):
    '''
    make_convs -> dict convert a list of convs to dict to meet limit of llava train data
    eg:
    [ q1, a1, q2, a2, ...] -> [{"from": "human", "value": q1}, {"from": "gpt", "value": a1}, ...]
    '''
    data_convs=[]
    roles=["human","gpt"]
    for i,c in enumerate(convs):
        if i==0:
            if "<image>" not in c:
                c="".join(["<image>\n"]*img_num)+c
        data_convs.append({"from":roles[i%2],"value":c})
    return data_convs

def make_train_data_json(data_id,img_ps,convs,img_p=None,masks=None,bboxes=None,**kwargs):
    '''
    make_train_data_json -> dict produce dict in format of llava train data
    data_id -> str : corresponding "id" in llava train data
    img_ps -> list[str] : list of image paths will be store in "src", note if only 1 image, "image" will replace "src"
    convs -> list[str] : list of conversations in format of [ q1, a1, q2, a2, ...], will be auto converted to llava train data format
    img_p -> str : image path that saved in "image", will use "src"[0] if set to None
    masks -> list[str] : mask paths that saved in "masks"
    bboxes -> list[list[list[int]]] : bboxes saved in "bboxes" for every image, every bbox, every axis
    **kwargs -> Any : any other kwargs will auto convert to key-value in data dict
    '''
    data={}
    data["id"]=data_id
    if type(img_ps) is str:
        img_ps=[img_ps]
    if len(img_ps)>1:
        data["src"]=img_ps
    if img_p is None:
        data["image"]=img_ps[0]
    else:
        data["image"]=img_p
    data["image_num"]=img_num=len(img_ps)
    data["conversations"]=make_convs(convs,img_num)
    if masks is not None:
        if type(masks) is list and len(masks)==1:
            masks=masks[0]
        data["mask"]=masks
    if bboxes is not None:
        data["bbox"]=bboxes
    data.update(kwargs)
    return data

def data_read(data_path):
    '''
    data_read -> list[dict] used to read data from question.jsonl or other json file storing eval data annotations, 
    current support llava train data format and question.jsonl format, after read data should be in llava train data format
    data_path -> str : the full path of the eval data file you want to read
    '''
    data_dir=os.path.dirname(data_path)
    dataset_name=os.path.basename(data_dir)
    all_datas=[]
    if data_path.endswith(".jsonl"): 
        with jsonlines.open(data_path) as f:
            jsonl_datas=list(f)
        for i in jsonl_datas:
            question_id=i["question_id"]
            img_p=os.path.join("imgs",i["image"])
            text=i["text"]
            ref=i.get("reference",None)
            if "sub_dataset" in i:
                subset=i["sub_dataset"] # in old formation it is sub_dataset
            else: # try to guess a subset
                default_subset=i["origin_path"].split("/")[0]
                if default_subset==dataset_name:
                    default_subset=i["origin_path"].split("/")[1]
                subset=i.get("subset", default_subset)
            gt=i.get("gt",None)
            origin_path=i["origin_path"]
            mask_p=i.get("mask",None)
            if mask_p is not None:
                mask_p=os.path.join(mask_p)
            bbox=i.get("bbox",None)
            musc_scores=i.get("musc_scores",None)
            if ref is not None:
                ref=os.path.join("mvtec",ref)
                img_ps=[img_p,ref]
            else:
                img_ps=[img_p]
            all_datas.append(make_train_data_json(
                question_id,img_ps,[text,""],img_ps[0],mask_p,bbox,
                subset=subset,gt=gt,origin_path=origin_path,
                musc_scores=musc_scores,
            ))
    else: # llava train data formation, still need "subset" "gt" "origin_path" just for unified mid format
        with open(data_path) as f:
            all_datas=json.load(f)
        for i in all_datas:
            if "subset" not in i:
                i["subset"]=i["image"].split("/")[1]
            if "gt" not in i:
                i["gt"]=0 if "/good/" in i["image"] else 1
            if "origin_path" not in i:
                i["origin_path"]=i["image"]
            if "bbox" not in i:
                i["bbox"]=None
            if "mask" not in i:
                i["mask"]=None
    return all_datas
    
def data_write(ans,out_path,format="json",mode="w"):
    '''
    data_write -> None write answers to json or jsonl file for mid storage, generally jsonl for mid temp storage and json for final result.
    ans -> list[dict] : all answers you want to save
    out_path -> str : full path of answer file path
    format -> ["json","jsonl"] : the out file format
    mode -> ["a","w"] : write mode of file
    '''
    ans_dir=os.path.dirname(out_path)
    os.makedirs(ans_dir,exist_ok=True)
    if format=="jsonl":
        out_path=out_path.replace(".json",".jsonl")
        with jsonlines.open(out_path,mode) as f:
            f.write(ans)
    else:
        out_path=out_path.replace(".jsonl",".json")
        with open(out_path,mode) as f:
            json.dump(ans,f,indent=4)
            
def read_mask(mask_p):
    '''
    read_mask -> np.array : read mask from mask_p support image and numpy mats
    '''
    if mask_p.endswith((".png",".jpg",".PNG",".JPG")):
        mask=Image.open(mask_p).convert('L')
    elif mask_p.endswith((".npz",".npy")):
        mask=np.load(mask_p)
        if isinstance(mask,dict):
            mask=mask["anomaly_map"]
    return mask
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--model-type", type=str, default='triad', choices=['triad', 'transformers', 'sglang'])
    parser.add_argument("--use-flash-attn", action="store_true")
    # dataset config
    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./evaluate/", help="DO NOT add / behind data_path if you want directly the dataset dir")
    parser.add_argument("--text_src", type=str, required=True)# choices=['dataset', 'config:version', 'input:prompt']
    parser.add_argument("--subset", type=str, default="all") # name of subsets in eval data, separate with ',' if multiple subsets
    parser.add_argument("--from", type=str, default='question.jsonl', dest='question_file')
    parser.add_argument("--to", type=str, default='answer.json', dest='answer_file')
    parser.add_argument("--start", type=int, default=0) # start from mid, for resuming eval
    # evaluation config
    parser.add_argument("--mask_mode", type=str, default='default', choices=['all_none', 'all_empty', 'empty_when_missing', 'default']) # default: read from eval data, none when missing
    parser.add_argument("--ref_mode",  type=str, default='default', choices=['all_empty', 'copy_query', 'default']) # default: read from eval data
    parser.add_argument("--ref_mask_mode",  type=str, default='default', choices=['none_for_ref', 'empty_for_ref', 'copy_query_mask', 'default', 'follow_mask_mode']) # default: use mask from eval data, if none copy query mask 
    parser.add_argument("--qa_mode",  type=str, default='default', choices=['single_qa', 'default']) # default: multi_qa according to eval data
    parser.add_argument("--image_num", type=int, default=None) # manaually set image num, will auto pad <image> and empty picture
    parser.add_argument("--overwrite_image_aspect_ratio", type=str, default=None)
    parser.add_argument("--use_pad_when_db", action='store_true') # force image_aspect_ratio use pad when input images > 1
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. <=0.001 disables sampling (greedy). Default 0 for reproducible musc_v0.",
    )
    parser.add_argument(
        "--ab_tau",
        type=float,
        default=None,
        help=(
            "Scheme A (v0 only). If set (>=0): first-token P(A)>=tau => A else B. "
            "Do not combine with --scheme_requery. Forbidden for cotm_*."
        ),
    )
    parser.add_argument(
        "--scheme_requery",
        type=str,
        default=None,
        choices=["hyp", "hyp_v2", "selfq", "attncrop"],
        help=(
            "Hypothesis requery (v0 only). NEVER musc_scores. "
            "hyp: fixed-template H_A/H_B, hard +1 aggregate. "
            "hyp_v2: fixed-template, soft votes + pair antisymmetry + p_f mix. "
            "selfq: model WRITES its own yes/no check (T2), answered by forced two-choice "
            "readout, entropy-gated + Bayesian logit fusion (no tau_ev/band/delta). "
            "attncrop (default_v4): attention-guided suspect box -> crop a high-fidelity ROI "
            "from the ORIGINAL image via the EG-RoI view layout, then the default_v2 checklist "
            "is re-asked on that crop. Attention only proposes the box; VLM A/B still decides. "
            "Selection: --requery_gate band (default) or quantile (Defv3). "
            "Fallback to llm_letter. Mutually exclusive with --ab_tau."
        ),
    )
    parser.add_argument("--band_lo", type=float, default=0.4, help="Band gate: uncertain-band low (inclusive).")
    parser.add_argument("--band_hi", type=float, default=0.7, help="Band gate: uncertain-band high (inclusive).")
    parser.add_argument(
        "--requery_gate",
        type=str,
        default="band",
        choices=["band", "quantile"],
        help=(
            "Hard-example selection. band: p_A in [band_lo, band_hi]. "
            "quantile: two-pass; requery top requery_top_frac by uncertainty u=1-|2p_A-1| "
            "(no absolute p_A thresholds). Default band."
        ),
    )
    parser.add_argument(
        "--requery_top_frac",
        type=float,
        default=0.1,
        help="Quantile gate: fraction of run to requery (most uncertain). Default 0.1.",
    )
    parser.add_argument(
        "--requery_tau_ev",
        type=float,
        default=0.7,
        help="Keep sub-answer as evidence only if max(P(A),P(B)) >= tau_ev. Default 0.7.",
    )
    parser.add_argument(
        "--requery_delta",
        type=float,
        default=None,
        help=(
            "Decision margin threshold. hyp: |S_A-S_B| >= delta (default 1.0). "
            "hyp_v2: soft |score| >= delta (default 0.5); else fall back to llm_letter."
        ),
    )
    parser.add_argument(
        "--requery_max_subq",
        type=int,
        default=2,
        help="Max sub-questions per hypothesis side (H_A / H_B). Default 2.",
    )
    parser.add_argument(
        "--requery_checklist",
        type=str,
        default="per_class",
        choices=["per_class", "default", "default_v3"],
        help=(
            "Hyp checklist: per_class | default (surface/stain/geometry) | "
            "default_v3 (contrastive continuity checks, Defv3 / no defect taxonomy words)."
        ),
    )
    parser.add_argument(
        "--requery_v2_conflict",
        type=float,
        default=0.3,
        help="hyp_v2: pair weight when H_A/H_B same Yes/No (conflict). Default 0.3.",
    )
    parser.add_argument(
        "--requery_v2_agree",
        type=float,
        default=1.2,
        help="hyp_v2: pair weight when H_A/H_B opposite Yes/No (agree). Default 1.2.",
    )
    parser.add_argument(
        "--requery_v2_margin_ref",
        type=float,
        default=1.0,
        help="hyp_v2: beta = min(1, margin/margin_ref) for mixing with p_f. Default 1.0.",
    )
    parser.add_argument(
        "--requery_v2_conflict_ratio",
        type=float,
        default=0.35,
        help="hyp_v2: if margin/mass < ratio -> fallback (noisy conflict). Default 0.35.",
    )
    parser.add_argument(
        "--requery_v2_temp",
        type=float,
        default=1.0,
        help="hyp_v2: tanh(score/temp) temperature. Default 1.0.",
    )
    parser.add_argument(
        "--requery_max_turns",
        type=int,
        default=2,
        help="selfq: max self-generated questions per selected sample. Default 2.",
    )
    parser.add_argument(
        "--requery_selfq_clip",
        type=float,
        default=6.0,
        help="selfq: saturation cap on each evidence logit (numeric guard, not a decision "
        "constant; ~p=0.9975). Default 6.0.",
    )
    parser.add_argument(
        "--requery_selfq_gen_max_new_tokens",
        type=int,
        default=64,
        help="selfq: max new tokens when the model writes its own question. Default 64.",
    )
    parser.add_argument(
        "--requery_attncrop_layers",
        type=str,
        default="midband",
        help="attncrop: attention layer spec. midband(default)|mid|last|last4|comma list.",
    )
    parser.add_argument(
        "--requery_attncrop_thr_q",
        type=float,
        default=0.90,
        help="attncrop: relative quantile threshold on the normalized anyres grid. Default 0.90.",
    )
    parser.add_argument(
        "--requery_attncrop_expand",
        type=float,
        default=1.3,
        help="attncrop: square box expansion factor around the attention blob. Default 1.3.",
    )
    parser.add_argument(
        "--requery_attncrop_side_lo",
        type=float,
        default=0.06,
        help="attncrop: min box side as a fraction of the short image side. Default 0.06.",
    )
    parser.add_argument(
        "--requery_attncrop_side_hi",
        type=float,
        default=0.45,
        help="attncrop: max box side as a fraction of the short image side. Default 0.45.",
    )
    parser.add_argument(
        "--requery_attncrop_conc_floor",
        type=float,
        default=0.30,
        help="attncrop: min top-5%% attention mass to accept a box (else diffuse -> no crop). "
        "Default 0.30.",
    )
    parser.add_argument(
        "--requery_attncrop_overlay_dir",
        type=str,
        default=None,
        help="attncrop: if set, write attention-overlay + crop-box PNGs (per sample id) here.",
    )
    parser.add_argument(
        "--ecpr_sidecar",
        type=str,
        default=None,
        help="Optional ECPR sidecar jsonl; when set, replace query boxes for juice samples.",
    )
    args = parser.parse_args()
    
    
    
    # read data
    ds_name=args.ds
    data_dir=args.data_path
    text_src=args.text_src
    start=args.start
    if data_dir.endswith("/") and not data_dir.endswith(ds_name+"/"): # guess it is on ./evaluate/
        data_dir=os.path.join(data_dir,ds_name)
    question_path=os.path.join(data_dir,args.question_file)
    answer_path=os.path.join(data_dir,args.answer_file)
    
    all_datas=data_read(question_path)
    
    qs_tmp=None
    prompt_version = None
    if text_src.startswith("config:"):
        v=text_src.replace("config:","",1)
        prompt_version = v
        assert ds_name in SUPPORTED_QSET, f"{args.ds} not in eval config"
        assert v in SUPPORTED_QSET[ds_name], f"{args.ds} has no prompt version {v} in eval config"
        qs_tmp=SUPPORTED_QSET[ds_name][v]
        print(f"using prompt {v} for {ds_name} in eval config")
    elif text_src.startswith("input:"):
        qs_tmp=text_src.replace("input:","",1)
        print(f"using input prompt {qs_tmp} as question")
    elif text_src.startswith("dataset"):
        print("using prompt from eval data")
    else:
        raise ValueError(f"unkown text_src: {text_src}")

    # ab_tau / scheme_requery: opt-in, v0 only, mutually exclusive.
    # No musc_scores hard gate (scheme_b removed).
    ab_tau = args.ab_tau
    if ab_tau is not None and float(ab_tau) < 0:
        ab_tau = None
    requery = None
    if args.scheme_requery is not None and ab_tau is not None:
        raise ValueError("Pick at most one of --scheme_requery, --ab_tau.")
    if args.scheme_requery is not None:
        if prompt_version != "v0":
            raise ValueError(
                f"--scheme_requery requires config:v0 (got text_src={text_src!r})."
            )
        mode = str(args.scheme_requery)
        if args.requery_delta is not None:
            delta = float(args.requery_delta)
        else:
            delta = 0.5 if mode == "hyp_v2" else 1.0
        gate = str(args.requery_gate).lower()
        requery = {
            "mode": mode,
            "lo": float(args.band_lo),
            "hi": float(args.band_hi),
            "tau_ev": float(args.requery_tau_ev),
            "delta": delta,
            "max_subq": int(args.requery_max_subq),
            "checklist": str(args.requery_checklist),
            "gate": gate,
            "top_frac": float(args.requery_top_frac),
            "max_turns": int(args.requery_max_turns),
            "selfq_clip": float(args.requery_selfq_clip),
            "selfq_gen_max_new_tokens": int(args.requery_selfq_gen_max_new_tokens),
        }
        if mode == "hyp_v2":
            requery.update({
                "conflict_lambda": float(args.requery_v2_conflict),
                "agree_lambda": float(args.requery_v2_agree),
                "margin_ref": float(args.requery_v2_margin_ref),
                "conflict_ratio": float(args.requery_v2_conflict_ratio),
                "soft_temp": float(args.requery_v2_temp),
            })
        if mode == "attncrop":
            # default_v4: default checklist + all 3 geometry pairs unless overridden.
            if str(args.requery_checklist) == "per_class":
                requery["checklist"] = "default"
            requery["max_subq"] = int(args.requery_max_subq) if args.requery_max_subq != 2 else 3
            requery.update({
                "attncrop_layers": str(args.requery_attncrop_layers),
                "attncrop_thr_q": float(args.requery_attncrop_thr_q),
                "attncrop_expand": float(args.requery_attncrop_expand),
                "attncrop_side_lo": float(args.requery_attncrop_side_lo),
                "attncrop_side_hi": float(args.requery_attncrop_side_hi),
                "attncrop_conc_floor": float(args.requery_attncrop_conc_floor),
                "overlay_dir": args.requery_attncrop_overlay_dir,
            })
            if args.requery_attncrop_overlay_dir:
                os.makedirs(args.requery_attncrop_overlay_dir, exist_ok=True)
    if ab_tau is not None:
        if prompt_version != "v0":
            raise ValueError(
                f"--ab_tau={ab_tau} forces single-token A/B and must not be used with "
                f"text_src={text_src!r}. Use only with config:v0; for cotm_* omit --ab_tau."
            )
    print(
        f"decode: temperature={args.temperature}, "
        f"ab_tau={ab_tau if ab_tau is not None else 'off'}, "
        f"scheme_requery={requery if requery is not None else 'off'}"
    )
    
    if args.subset!="all":
        subsets=args.subset.split(",")
    else:
        subsets=None
    
    # load model
    worker=None
    if args.model_type=="triad":
        worker=SimpleModelWrapper(
            args.model_path, 
            args.model_base, 
            args.model_name, 
            'cuda',
            args.use_flash_attn,
            image_aspect_ratio=args.overwrite_image_aspect_ratio
        )
    elif args.model_type=="transformers":
        worker=TransformersModelWrapper(
            args.model_path,
            device='cuda',
        )
    elif args.model_type=="sglang":
        #worker=SGLangModelWrapper
        pass
    else:
        raise ValueError(f"unkown model_type: {args.model_type}")

    ecpr_index = None
    if getattr(args, "ecpr_sidecar", None):
        import importlib.util
        _slots = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "LLaVA-NeXT",
            "llava",
            "train",
            "ecpr_slots.py",
        )
        _spec = importlib.util.spec_from_file_location("ecpr_slots", _slots)
        _ecpr = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_ecpr)
        ecpr_index = _ecpr.load_sidecar_index(args.ecpr_sidecar)
        print(f"ECPR sidecar loaded: {args.ecpr_sidecar} ({len(ecpr_index)} keys)")
    
    from hyp_checklists import uncertainty_quantile_threshold

    def _eval_one(d, requery_cfg):
        """Run one sample; requery_cfg may be None or a per-pass requery dict."""
        subset = d["subset"]
        if type(qs_tmp) is dict:
            if subset not in qs_tmp:
                if "default" not in qs_tmp:
                    raise ValueError(f"subset {subset} in {ds_name} version {v} not in eval config")
                qs = qs_tmp["default"]
            else:
                qs = qs_tmp[subset]
        else:
            qs = qs_tmp

        if "src" in d:
            img_p = d["src"]
        elif "image" in d and d["image"] is not None:
            img_p = d["image"]
            if type(img_p) is str:
                img_p = [img_p]
        else:
            img_p = []

        imgs = [Image.open(os.path.join(data_dir, i)).convert("RGB") for i in img_p]

        if args.image_num is None:
            img_num = len(imgs)
        else:
            img_num = args.image_num

        if img_num < len(imgs):
            print(f"WARNING: image_num {img_num} is smaller than len(images) {len(imgs)}, will chunk")
            imgs = imgs[:img_num]
        elif img_num > len(imgs):
            print(f"WARNING: image_num {img_num} is larger than len(images) {len(imgs)}, will pad")
            img_e = Image.new("RGB", imgs[0].size, (0, 0, 0))
            imgs = imgs + [img_e] * (img_num - len(imgs))

        assert img_num == len(imgs)

        img_q = imgs[0]
        img_r = imgs[1:]
        if args.ref_mode == "all_empty":
            img_r = [Image.new("RGB", i.size, (0, 0, 0)) for i in img_r]
        elif args.ref_mode == "copy_query":
            img_r = [img_q] * (img_num - 1)
        elif args.ref_mode == "default":
            pass
        else:
            raise ValueError(f"unkown ref_mode {args.ref_mode}")
        imgs = [img_q] + img_r
        assert img_num == len(imgs), "after ref process"

        if "mask" in d and d["mask"] is not None:
            mask_p = d["mask"]
            if type(mask_p) is str:
                mask_p = [mask_p]
            masks = [read_mask(os.path.join(data_dir, i)) for i in mask_p]
        else:
            masks = None
        mask_o = masks
        if masks is None:
            mask_o = []
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
                img_np = img.numpy()
                masks.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
        elif args.mask_mode == "empty_when_missing":
            masks = []
            for idx, img in enumerate(imgs):
                if mask_o[idx] is None:
                    img_np = img.numpy()
                    masks.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
                else:
                    masks.append(mask_o[idx])
        elif args.mask_mode == "default":
            masks = mask_o
        else:
            raise ValueError(f"unkown mask_mode {args.mask_mode}")

        mask_q = masks[0]
        mask_r = masks[1:]
        mask_rnum = len(mask_r)
        if args.ref_mask_mode == "none_for_ref":
            mask_r = [None] * mask_rnum
        elif args.ref_mask_mode == "empty_for_ref":
            mask_r = []
            for img in imgs[1:]:
                img_np = img.numpy()
                mask_r.append(np.zeros(img_np.shape[:2], dtype=img_np.dtype))
        elif args.ref_mask_mode == "copy_query_mask":
            mask_r = [mask_q] * mask_rnum
        elif args.ref_mask_mode == "default":
            mask_r = []
            for idx, img in enumerate(imgs[1:], start=1):
                if mask_o[idx] is None:
                    mask_r.append(mask_q)
                else:
                    mask_r.append(mask_o[idx])
        elif args.ref_mask_mode == "follow_mask_mode":
            pass
        else:
            raise ValueError(f"unkown ref_mask_mode {args.ref_mask_mode}")
        masks = [mask_q] + mask_r

        if "bbox" in d and d["bbox"] is not None:
            boxes_list = d["bbox"]
        else:
            boxes_list = None

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

        inputs = {
            "text": qs,
            "temperature": args.temperature,
            "top_p": 0.7,
            "max_new_tokens": args.max_new_tokens,
            "use_pad_when_db": args.use_pad_when_db,
            "images": imgs,
            "masks": masks,
            "boxes_list": boxes_list,
            "ab_tau": ab_tau,
            "requery": requery_cfg,
            "subset": subset,
            "sample_id": d.get("id"),
        }

        assert worker is not None
        ans = deepcopy(d)
        if qs is None:
            qs = d["conversations"]
        if type(qs) is list:
            if args.qa_mode == "single_qa":
                convs = worker.multi_qa(inputs, qs, forget_state=True)
            elif args.qa_mode == "default":
                convs = worker.multi_qa(inputs, qs)
            else:
                raise ValueError(f"unkown qa_mode {args.qa_mode}")
            ans["conversations"] = convs
        elif type(qs) is str:
            _, out = worker.single_qa(inputs)
            ans["conversations"] = make_convs([qs, out])
        else:
            raise ValueError(f"unkown qs type {type(qs)}")

        sm = getattr(worker, "model", None)
        if sm is not None and getattr(sm, "last_ab_decision", None) is not None:
            ans["ab_decision"] = dict(sm.last_ab_decision)
        return ans

    # Filter subset once so two-pass indices stay aligned.
    eval_datas = []
    for d in all_datas[start:]:
        if subsets is not None and d["subset"] not in subsets:
            continue
        eval_datas.append(d)

    use_quantile = (
        requery is not None
        and str(requery.get("gate", "band")).lower() == "quantile"
    )
    all_ans = []

    if use_quantile:
        # Pass-1: p_A / uncertainty only (no sub-questions).
        pass1_cfg = dict(requery)
        pass1_cfg["score_only"] = True
        pass1_ans = []
        u_vals = []
        print(
            f"requery quantile pass-1 (score_only), "
            f"top_frac={requery.get('top_frac', 0.1)}"
        )
        for d in tqdm(eval_datas, desc="requery-pass1"):
            ans = _eval_one(d, pass1_cfg)
            pass1_ans.append(ans)
            u = None
            if "ab_decision" in ans:
                u = ans["ab_decision"].get("u")
            if u is None and "ab_decision" in ans:
                p_a = ans["ab_decision"].get("p_A")
                if p_a is not None:
                    from hyp_checklists import uncertainty_from_p_a
                    u = uncertainty_from_p_a(p_a)
            u_vals.append(0.0 if u is None else float(u))

        u_thr = uncertainty_quantile_threshold(u_vals, float(requery.get("top_frac", 0.1)))
        n_sel = sum(1 for u in u_vals if u >= u_thr)
        print(f"requery quantile: u_thr={u_thr:.6f}, selected={n_sel}/{len(u_vals)}")

        pass2_cfg = dict(requery)
        pass2_cfg["score_only"] = False
        pass2_cfg["u_thr"] = u_thr

        # Pass-2: only re-run selected hard examples; keep pass-1 for the rest.
        for d, ans1, u in tqdm(
            list(zip(eval_datas, pass1_ans, u_vals)), desc="requery-pass2"
        ):
            if u >= u_thr:
                ans = _eval_one(d, pass2_cfg)
            else:
                ans = ans1
                abd = dict(ans.get("ab_decision") or {})
                abd["in_band"] = False
                abd["u"] = u
                abd["u_thr"] = u_thr
                abd["gate"] = "quantile"
                abd["decision_src"] = "llm_argmax_out_of_band"
                abd["pred_letter"] = abd.get("llm_letter") or abd.get("pred_letter")
                ans["ab_decision"] = abd
            data_write(ans, answer_path, "jsonl", "a")
            all_ans.append(ans)
    else:
        for d in tqdm(eval_datas):
            ans = _eval_one(d, requery)
            data_write(ans, answer_path, "jsonl", "a")
            all_ans.append(ans)

    data_write(all_ans, answer_path, "json", "w")
    jsonl_tmp = answer_path.replace(".json", ".jsonl")
    if os.path.isfile(jsonl_tmp):
        os.remove(jsonl_tmp)
