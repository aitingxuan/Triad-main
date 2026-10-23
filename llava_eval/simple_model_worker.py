import json
import time
import math
import os
import re
import torch
from tqdm import tqdm
import requests
import shortuuid
import hashlib

# from evaluation_config import *

from PIL import Image

from llava.mm_utils import (
    process_images,
    load_image_from_base64,
    tokenizer_image_token,
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)

from hyp_checklists import format_subq, get_hyp_checklist, uncertainty_from_p_a
from hyp_aggregate_v2 import aggregate_hyp_v2
from hyp_selfq import (
    build_instruction as build_selfq_instruction,
    parse_selfq_generation,
    entropy_accept,
    sub_evidence_logit_for_a,
    bayes_fuse,
    DEFAULT_EVIDENCE_CLIP,
)


def _single_token_ids(tokenizer, texts):
    ids = []
    seen = set()
    unk = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            tid = int(enc[0])
            if tid not in seen and tid != unk:
                seen.add(tid)
                ids.append(tid)
        tid2 = tokenizer.convert_tokens_to_ids(text)
        if isinstance(tid2, int) and tid2 != unk and tid2 not in seen:
            if tokenizer.convert_ids_to_tokens(tid2) is not None:
                seen.add(tid2)
                ids.append(tid2)
    return ids


def resolve_ab_token_ids(tokenizer):
    """Single-token surface forms for A/B (with/without leading space)."""
    return {
        "A": _single_token_ids(tokenizer, ["A", " A", "A.", " A."]),
        "B": _single_token_ids(tokenizer, ["B", " B", "B.", " B."]),
    }


def two_way_softmax(z_a, z_b):
    m = max(z_a, z_b)
    ea = math.exp(z_a - m)
    eb = math.exp(z_b - m)
    s = ea + eb
    return ea / s, eb / s


class SimpleModelWorker:
    def __init__(self, model_path, model_base, model_name, device, 
                 load_8bit=False, load_4bit=False, use_flash_attn=False, image_aspect_ratio=None):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        self._ab_token_ids = None
        self.last_ab_decision = None
        print(f"Loading the model {self.model_name} ...")
        # @lyz(2025-01-03)
        # vocab_size = 152064 if '0.5b' not in self.model_name else 151936
        # ow_cfg={"vocab_size":vocab_size,}
        ow_cfg = {}
        if image_aspect_ratio is not None:
            print(f"overwrite image_aspect_ratio {image_aspect_ratio}")
            ow_cfg["image_aspect_ratio"]=image_aspect_ratio
            if image_aspect_ratio=="randomroi":
                ow_cfg["mm_patch_merge_type"]="spatial_avgpool_auto_unpad_add_newl"
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device, use_flash_attn=use_flash_attn,
            overwrite_config=ow_cfg)
        self.is_multimodal = 'llava' in self.model_name.lower()

    @torch.inference_mode()
    def generate_batch(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        masks = params.get("masks", None)
        boxes_list = params.get("boxes_list", None)
        # Keep the raw PIL inputs so attention-guided (attncrop) re-crops can
        # re-run process_images on the ORIGINAL image with a new ROI box.
        raw_images = images if isinstance(images, list) else None
        raw_masks = masks
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                # images = [load_image_from_base64(image) for image in images]
                image_sizes = [image.size for image in images]
                # 这里不做masks和boxes的选择，在process_images里做。
                if len(images)>1 and params.get('use_pad_when_db', False):
                    images, roi_bboxes = process_images(
                        images, image_processor, model.config, masks=masks, boxes_list=boxes_list,
                        overwrite_img_asp="pad", return_bboxes=True,
                    )
                else:
                    images, roi_bboxes = process_images(
                        images, image_processor, model.config, masks=masks, boxes_list=boxes_list,
                        return_bboxes=True,
                    )

                if type(images) is list:
                    if type(images[0]) is not dict:
                        images = [image.to(self.model.device, dtype=self.model.dtype) for image in images]
                else:
                    images = images.to(self.model.device, dtype=self.model.dtype)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
                image_sizes = None
                roi_bboxes = None
            image_args = {"images": images, "image_sizes": image_sizes, "roi_bboxes": roi_bboxes}
        else:
            images = None
            image_args = {}
        if '<image>' in prompt: assert images is not None
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        if max_new_tokens < 1:
            print(ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.")
            return ""

        # First-token A/B scoring (v0 only): ab_tau / scheme_requery.
        # Scheme A: ab_tau => P(A)>=tau ? A : B
        # Scheme requery (hyp): in-band => fixed-template H_A/H_B sub-Qs on full image;
        #   aggregate S_A/S_B; NEVER musc_scores. Fallback to llm_letter.
        ab_tau = params.get("ab_tau", None)
        requery = params.get("requery", None)
        use_ab_score = (
            (ab_tau is not None and float(ab_tau) >= 0)
            or (requery is not None)
        )
        if use_ab_score:
            if self._ab_token_ids is None:
                self._ab_token_ids = resolve_ab_token_ids(tokenizer)
            ab_ids = self._ab_token_ids
            if not ab_ids.get("A") or not ab_ids.get("B"):
                raise RuntimeError(f"Failed to resolve A/B token ids: {ab_ids}")

            def _ab_from_logits(logits_1d):
                z_a = float(max(logits_1d[i].item() for i in ab_ids["A"]))
                z_b = float(max(logits_1d[i].item() for i in ab_ids["B"]))
                p_a, p_b = two_way_softmax(z_a, z_b)
                return z_a, z_b, p_a, p_b

            def _ab_forward_prompt(prompt_str, ia=None):
                """Single-token A/B score on the same processed image_args as the main Q.

                Pass ``ia`` to score on a different view set (e.g. an attention-cropped ROI).
                """
                ids = tokenizer_image_token(
                    prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
                gen = model.generate(
                    inputs=ids,
                    do_sample=False,
                    max_new_tokens=1,
                    use_cache=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **(ia if ia is not None else image_args),
                )
                return _ab_from_logits(gen.scores[0][0].float())

            out = model.generate(
                inputs=input_ids,
                do_sample=False,
                max_new_tokens=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                **image_args,
            )
            z_a, z_b, p_a, p_b = _ab_from_logits(out.scores[0][0].float())
            llm_letter = "A" if p_a >= 0.5 else "B"
            letter = llm_letter
            decision_src = "llm_argmax"
            in_band = False
            S_A = 0.0
            S_B = 0.0
            sub_records = []
            hyp_v2_diag = None

            u_score = uncertainty_from_p_a(p_a)
            u_thr = None
            gate_name = "band"
            if requery is not None:
                mode = str(requery.get("mode", "hyp")).lower()
                if mode not in ("hyp", "hyp_v2", "selfq", "attncrop"):
                    raise ValueError(
                        f"Unsupported requery mode={mode!r}; expected 'hyp', 'hyp_v2', "
                        f"'selfq' or 'attncrop'."
                    )
                lo = float(requery.get("lo", 0.4))
                hi = float(requery.get("hi", 0.7))
                tau_ev = float(requery.get("tau_ev", 0.7))
                # hyp: hard-count delta (default 1). hyp_v2: soft-margin delta (default 0.5).
                delta_default = 0.5 if mode == "hyp_v2" else 1.0
                delta = float(requery.get("delta", delta_default))
                max_subq = int(requery.get("max_subq", 2))
                subset = params.get("subset") or requery.get("subset") or ""
                gate_name = str(requery.get("gate", "band")).lower()
                score_only = bool(requery.get("score_only", False))

                if score_only:
                    # Pass-1 for quantile gate: record p_A only, no sub-questions.
                    in_band = False
                    letter = llm_letter
                    decision_src = "llm_argmax_score_only"
                elif gate_name == "quantile":
                    if requery.get("u_thr") is None:
                        raise ValueError(
                            "requery gate='quantile' requires u_thr "
                            "(compute via evaluate.py two-pass)."
                        )
                    u_thr = float(requery["u_thr"])
                    in_band = u_score >= u_thr
                else:
                    gate_name = "band"
                    in_band = lo <= p_a <= hi

                if score_only:
                    pass
                elif not in_band:
                    letter = llm_letter
                    decision_src = "llm_argmax_out_of_band"
                elif not image_args:
                    letter = llm_letter
                    decision_src = "hyp_fallback_no_image"
                else:
                    tmpl = params.get("conv_template", "qwen_1_5")
                    if tmpl not in conv_templates:
                        tmpl = "qwen_1_5"
                    use_im_se = getattr(model.config, "mm_use_im_start_end", False)

                    def _build_sub_prompt(question_text):
                        conv = conv_templates[tmpl].copy()
                        body = DEFAULT_IMAGE_TOKEN + "\n" + format_subq(question_text)
                        conv.append_message(conv.roles[0], body)
                        conv.append_message(conv.roles[1], None)
                        pr = conv.get_prompt()
                        if use_im_se:
                            rt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                            pr = pr.replace(DEFAULT_IMAGE_TOKEN, rt)
                        return pr

                    if mode == "selfq":
                        # Scheme 4 (T2): the model writes its own yes/no check, answered by a
                        # forced two-choice readout (A=Yes/B=No) so the answer is always binary.
                        # Evidence is entropy-gated (sharper than main) and Bayesian-fused.
                        ev_clip = float(requery.get("selfq_clip", DEFAULT_EVIDENCE_CLIP))
                        gen_max_new = int(requery.get("selfq_gen_max_new_tokens", 64))
                        max_turns = max(0, int(requery.get("max_turns", 2)))
                        main_text = params.get("main_text") or params.get("text")
                        prior_qs = []
                        evidences = []
                        for _t in range(max_turns):
                            instr = build_selfq_instruction(main_text=main_text, prior_qs=prior_qs)
                            gen_conv = conv_templates[tmpl].copy()
                            gbody = DEFAULT_IMAGE_TOKEN + "\n" + instr
                            gen_conv.append_message(gen_conv.roles[0], gbody)
                            gen_conv.append_message(gen_conv.roles[1], None)
                            gprompt = gen_conv.get_prompt()
                            if use_im_se:
                                rt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                                gprompt = gprompt.replace(DEFAULT_IMAGE_TOKEN, rt)
                            g_ids = tokenizer_image_token(
                                gprompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                            ).unsqueeze(0).to(self.device)
                            g_out = model.generate(
                                inputs=g_ids,
                                do_sample=False,
                                max_new_tokens=gen_max_new,
                                use_cache=True,
                                **image_args,
                            )
                            gen_text = tokenizer.decode(g_out[0], skip_special_tokens=True)
                            rec = {
                                "turn": _t,
                                "gen_text": gen_text,
                                "q": None,
                                "yes_supports": None,
                                "no_supports": None,
                                "p_A": None,
                                "p_B": None,
                                "accept": False,
                                "kept": False,
                                "reason": None,
                            }
                            parsed = parse_selfq_generation(gen_text)
                            if parsed is None:
                                rec["reason"] = "parse_fail"
                                sub_records.append(rec)
                                if prior_qs:
                                    break
                                continue
                            q = parsed["question"]
                            rec["q"] = q
                            rec["yes_supports"] = parsed["yes_supports"]
                            rec["no_supports"] = parsed["no_supports"]
                            prior_qs.append(q)
                            _, _, sp_a, sp_b = _ab_forward_prompt(_build_sub_prompt(q))
                            rec["p_A"] = sp_a
                            rec["p_B"] = sp_b
                            rec["conf"] = max(sp_a, sp_b)
                            rec["letter"] = "A" if sp_a >= 0.5 else "B"
                            w, z = sub_evidence_logit_for_a(
                                sp_a, parsed["yes_supports"], parsed["no_supports"], clip=ev_clip
                            )
                            if w <= 0.0:
                                rec["reason"] = "uninformative"
                            elif entropy_accept(sp_a, p_a):
                                rec["accept"] = True
                                rec["kept"] = True
                                rec["evidence_logit"] = z
                                evidences.append((w, z))
                            else:
                                rec["reason"] = "entropy_gate_reject"
                            sub_records.append(rec)

                        fused = bayes_fuse(p_a, evidences, clip=ev_clip)
                        letter = "A" if fused["logit_final"] >= 0.0 else "B"
                        decision_src = "selfq_bayes" if fused["n_evidence"] > 0 else "hyp_fallback"
                        hyp_v2_diag = {"selfq": fused, "mode": "selfq"}
                        ev = float(fused["evidence_logit"])
                        S_A = ev if ev > 0 else 0.0
                        S_B = -ev if ev < 0 else 0.0
                    elif mode == "attncrop":
                        # Default_v4: attention-guided suspect box -> crop a high-fidelity
                        # ROI from the ORIGINAL image (reusing the EG-RoI view layout via
                        # process_images(boxes_list=...)), then run the default_v2 checklist
                        # (H_A/H_B) on that crop. The attention map only PROPOSES the box;
                        # the final decision is still the VLM A/B sub-answers (never MuSc).
                        from attn_crop import extract_attn_box_prepared, propose_box_margin_cam

                        ac_cl = str(requery.get("checklist", "default")).lower()
                        ac_checklist = get_hyp_checklist(subset, checklist=ac_cl)
                        ac_max_subq = int(requery.get("max_subq", 3))
                        ha_qs = list(ac_checklist.get("H_A") or [])[: max(0, ac_max_subq)]
                        hb_qs = list(ac_checklist.get("H_B") or [])[: max(0, ac_max_subq)]

                        overlay_dir = requery.get("overlay_dir")
                        overlay_path = None
                        if overlay_dir:
                            safe_id = re.sub(r"[^\w.\-]+", "_", str(params.get("sample_id") or "sample"))[:100]
                            safe_sub = re.sub(r"[^\w.\-]+", "_", str(subset or "sub"))[:40]
                            overlay_path = os.path.join(overlay_dir, f"{safe_sub}__{safe_id}.png")

                        attn_meta = None
                        box = None
                        if raw_images is not None and len(raw_images) == 1:
                            try:
                                proposer = str(requery.get("attncrop_proposer", "margin_cam")).lower()
                                if proposer in ("margin_cam", "cam", "gradcam"):
                                    attn_meta = propose_box_margin_cam(
                                        model,
                                        image_args["images"],
                                        raw_images[0],
                                        image_sizes,
                                        image_args.get("roi_bboxes"),
                                        input_ids,
                                        ab_ids["A"],
                                        ab_ids["B"],
                                        thr_q=float(requery.get("attncrop_cam_thr_q", 0.60)),
                                        expand=float(requery.get("attncrop_expand", 1.3)),
                                        side_lo=float(requery.get("attncrop_side_lo", 0.06)),
                                        side_hi=float(requery.get("attncrop_side_hi", 0.45)),
                                        conc_floor=float(requery.get("attncrop_cam_conc_floor", 0.25)),
                                        overlay_path=overlay_path,
                                        overlay_title=f"{params.get('sample_id')} | {subset} | pA={p_a:.3f}",
                                    )
                                else:
                                    gen_tid = int(ab_ids[llm_letter][0])
                                    attn_meta = extract_attn_box_prepared(
                                        model,
                                        image_args["images"],
                                        raw_images[0],
                                        image_sizes,
                                        image_args.get("roi_bboxes"),
                                        input_ids,
                                        gen_tid,
                                        layers=str(requery.get("attncrop_layers", "midband")),
                                        thr_q=float(requery.get("attncrop_thr_q", 0.90)),
                                        expand=float(requery.get("attncrop_expand", 1.3)),
                                        side_lo=float(requery.get("attncrop_side_lo", 0.06)),
                                        side_hi=float(requery.get("attncrop_side_hi", 0.45)),
                                        conc_floor=float(requery.get("attncrop_conc_floor", 0.30)),
                                        overlay_path=overlay_path,
                                        overlay_title=f"{params.get('sample_id')} | {subset} | pA={p_a:.3f}",
                                    )
                            except Exception as e:
                                attn_meta = {"kept": False, "reason": f"error:{e}", "box": None}
                        else:
                            attn_meta = {"kept": False, "reason": "multi_image", "box": None}

                        if attn_meta and attn_meta.get("kept"):
                            box = attn_meta.get("box")

                        if box is not None:
                            crop_images, crop_roi = process_images(
                                raw_images,
                                image_processor,
                                model.config,
                                masks=raw_masks,
                                boxes_list=[[box]],
                                return_bboxes=True,
                            )
                            if isinstance(crop_images, list):
                                crop_images = [im.to(model.device, dtype=model.dtype) for im in crop_images]
                            else:
                                crop_images = crop_images.to(model.device, dtype=model.dtype)
                            ia_crop = {
                                "images": crop_images,
                                "image_sizes": image_sizes,
                                "roi_bboxes": crop_roi,
                            }
                            for hyp_tag, qlist in (("H_A", ha_qs), ("H_B", hb_qs)):
                                for q in qlist:
                                    _, _, sp_a, sp_b = _ab_forward_prompt(_build_sub_prompt(q), ia_crop)
                                    conf = max(sp_a, sp_b)
                                    ans_letter = "A" if sp_a >= 0.5 else "B"
                                    sub_records.append({
                                        "hyp": hyp_tag,
                                        "q": q,
                                        "p_A": sp_a,
                                        "p_B": sp_b,
                                        "letter": ans_letter,
                                        "conf": conf,
                                        "kept": conf >= tau_ev,
                                        "view": "attncrop",
                                    })
                            for rec in sub_records:
                                if not rec.get("kept"):
                                    continue
                                if rec["hyp"] == "H_A":
                                    if rec["letter"] == "A":
                                        S_A += 1.0
                                    else:
                                        S_B += 1.0
                                else:
                                    if rec["letter"] == "A":
                                        S_B += 1.0
                                    else:
                                        S_A += 1.0
                            if S_A - S_B >= delta:
                                letter = "A"
                                decision_src = "attncrop_test"
                            elif S_B - S_A >= delta:
                                letter = "B"
                                decision_src = "attncrop_test"
                            else:
                                letter = llm_letter
                                decision_src = "attncrop_fallback"
                        else:
                            letter = llm_letter
                            decision_src = "attncrop_fallback_nobox"

                        hyp_v2_diag = {
                            "mode": "attncrop",
                            "proposer": str(requery.get("attncrop_proposer", "margin_cam")).lower(),
                            "box": box,
                            "used_crop": box is not None,
                            "attn": attn_meta,
                        }
                    else:
                        cl_key = str(requery.get("checklist", "per_class")).lower()
                        checklist = get_hyp_checklist(subset, checklist=cl_key)
                        ha_qs = list(checklist.get("H_A") or [])[: max(0, max_subq)]
                        hb_qs = list(checklist.get("H_B") or [])[: max(0, max_subq)]

                        # Collect sub-answers (same forwards for hyp / hyp_v2).
                        for hyp_tag, qlist in (("H_A", ha_qs), ("H_B", hb_qs)):
                            for q in qlist:
                                _, _, sp_a, sp_b = _ab_forward_prompt(_build_sub_prompt(q))
                                conf = max(sp_a, sp_b)
                                ans_letter = "A" if sp_a >= 0.5 else "B"
                                kept = conf >= tau_ev
                                sub_records.append({
                                    "hyp": hyp_tag,
                                    "q": q,
                                    "p_A": sp_a,
                                    "p_B": sp_b,
                                    "letter": ans_letter,
                                    "conf": conf,
                                    "kept": kept,
                                })

                        if mode == "hyp_v2":
                            letter, decision_src, hyp_v2_diag = aggregate_hyp_v2(
                                sub_records,
                                p_a,
                                llm_letter,
                                tau_ev=tau_ev,
                                delta=delta,
                                conflict_lambda=float(requery.get("conflict_lambda", 0.3)),
                                agree_lambda=float(requery.get("agree_lambda", 1.2)),
                                margin_ref=float(requery.get("margin_ref", 1.0)),
                                conflict_ratio=float(requery.get("conflict_ratio", 0.35)),
                                soft_temp=float(requery.get("soft_temp", 1.0)),
                            )
                            S_A = float(hyp_v2_diag.get("S_A", 0.0))
                            S_B = float(hyp_v2_diag.get("S_B", 0.0))
                        else:
                            # hyp v1: hard +1 counts (H_A Yes=>A, H_A No=>B; H_B Yes=>B, H_B No=>A).
                            for rec in sub_records:
                                if not rec["kept"]:
                                    continue
                                if rec["hyp"] == "H_A":
                                    if rec["letter"] == "A":
                                        S_A += 1.0
                                    else:
                                        S_B += 1.0
                                else:
                                    if rec["letter"] == "A":
                                        S_B += 1.0
                                    else:
                                        S_A += 1.0
                            if S_A - S_B >= delta:
                                letter = "A"
                                decision_src = "hyp_test"
                            elif S_B - S_A >= delta:
                                letter = "B"
                                decision_src = "hyp_test"
                            else:
                                letter = llm_letter
                                decision_src = "hyp_fallback"
            elif ab_tau is not None and float(ab_tau) >= 0:
                tau = float(ab_tau)
                letter = "A" if p_a >= tau else "B"
                decision_src = f"ab_tau_{tau}"

            self.last_ab_decision = {
                "p_A": p_a,
                "p_B": p_b,
                "z_A": z_a,
                "z_B": z_b,
                "pred_letter": letter,
                "llm_letter": llm_letter,
                "decision_src": decision_src,
                "in_band": in_band,
                "u": u_score if requery is not None else None,
                "u_thr": u_thr,
                "gate": gate_name if requery is not None else None,
                "ab_tau": float(ab_tau) if ab_tau is not None and float(ab_tau) >= 0 else None,
                "requery": dict(requery) if requery is not None else None,
                "subset": params.get("subset") or (requery or {}).get("subset"),
                "S_A": S_A if requery is not None else None,
                "S_B": S_B if requery is not None else None,
                "sub_answers": sub_records if requery is not None else None,
                "hyp_v2": hyp_v2_diag if requery is not None else None,
                "selfq": (
                    hyp_v2_diag.get("selfq")
                    if (requery is not None and str((requery or {}).get("mode", "")).lower() == "selfq"
                        and isinstance(hyp_v2_diag, dict))
                    else None
                ),
                "attncrop": (
                    hyp_v2_diag
                    if (requery is not None
                        and str((requery or {}).get("mode", "")).lower() == "attncrop"
                        and isinstance(hyp_v2_diag, dict))
                    else None
                ),
            }
            return letter

        self.last_ab_decision = None
        data = model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **image_args
        )
        return ''.join(tokenizer.batch_decode(
            data, 
            skip_special_tokens=True
        ))

class SimpleModelWrapper:
    def __init__(self,*args,**kwargs):
        self.model=SimpleModelWorker(*args,**kwargs)
        self.model_name=self.model.model_name
        self.template_name=self.select_template(self.model_name)
        print("using conversation template",self.template_name)
        
    def select_template(self, model_name):
        if "llava" in model_name.lower():
            if 'llama-2' in model_name.lower():
                template_name = "llava_llama_2"
            elif "qwen" in model_name.lower():
                template_name = "qwen_1_5"
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                if 'orca' in model_name.lower():
                    template_name = "mistral_orca"
                elif 'hermes' in model_name.lower():
                    template_name = "chatml_direct"
                else:
                    template_name = "mistral_instruct"
            elif 'llava-v1.6-34b' in model_name.lower():
                template_name = "chatml_direct"
            # elif 'llava-v1.6-vicuna-13b' in model_name.lower():
            #     template_name = "vicuna_v1"
            elif "v1" in model_name.lower():
                if 'mmtag' in model_name.lower():
                    template_name = "v1_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if 'mmtag' in model_name.lower():
                    template_name = "v0_mmtag"
                elif 'plain' in model_name.lower() and 'finetune' not in model_name.lower():
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        return template_name
    
    def add_text(self, state, text, image, image_process_mode="Default"):  # Follow gradio setting, the image_process_mode Radio is not visible and not changable.
        assert len(text) <= 8192, "测试的text超过了8192个字符."
        text = text[:8192]  # Hard cut-off # origin 1536
        if image is not None:
            assert len(text) <= 4096, "多模态时测试的text超过了4096个字符." 
            text = text[:4096]  # Hard cut-off for images # origin 1200
            if '<image>' not in text:
                # text = '<Image><image></Image>' + text
                # text = text + '\n<image>'
                text = '<image>\n' + text
            text = (text, image, image_process_mode)
        state.append_message(state.roles[0], text)
        state.append_message(state.roles[1], None)
        state.skip_next = False
        return state
    
    def new_state(self):
        return conv_templates[self.template_name].copy()
    
    def single_qa(self,inputs,state=None,multi_round=False):
        if state is None:
            state=self.new_state()
        image_num=0
        imgs=inputs.get("images",None)
        if imgs is not None:
            image_num=len(imgs)
        qs=inputs.get("text",None)
        state=self.add_text(state, qs, None if multi_round else imgs)
        prompt=state.get_prompt()
        if prompt.count("<image>")==1 and prompt.count("<image>") != image_num:
            img_s="\n".join(["<image>"]*image_num)
            prompt=prompt.replace("<image>",img_s)
        out=self.model.generate_batch({
                "model": self.model_name,
                "prompt": prompt,
                "temperature": inputs.get("temperature",0.0),
                "top_p":  inputs.get("top_p",0.7),
                "max_new_tokens": inputs.get("max_new_tokens",512),
                "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
                "use_pad_when_db": inputs.get("use_pad_when_db",False), 
                "images": imgs,
                "masks": inputs.get("masks",None),
                "boxes_list": inputs.get("boxes_list",None),
                "ab_tau": inputs.get("ab_tau", None),
                "requery": inputs.get("requery", None),
                "subset": inputs.get("subset", None),
                "sample_id": inputs.get("sample_id", None),
                "conv_template": self.template_name,
                "main_text": inputs.get("text", None),
            }).strip()
        state.messages[-1][-1]=out
        return state,out
    
    # 封装了多轮对话
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self.new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            if forget_state:
                state=self.new_state()
            state,out=self.single_qa(inputs,state,conv_idx!=0)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs
        

class TransformersModelWrapper:
    """
    Refer to [
        https://qwen.readthedocs.io/en/latest/inference/chat.html.  (LLM)
        https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct  (Qwen2VL)
        https://huggingface.co/docs/transformers/main/en/model_doc/llava_onevision#single-image-inference  (LLAVA-OneVision)
        https://huggingface.co/openbmb/MiniCPM-V-2_6  (MiniCPM-V 2.6)
    ]
    """
    def __init__(self, model_path, task='LMM', **kwargs):
        import torch
        self.task = task
        assert task in ['LLM', 'LMM'], f"TransformersModelWrapper不支持{task}类型."
        self.device = kwargs.get('device', None)
        if self.task == 'LLM':
            from transformers import pipeline
            
            if self.device is not None:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device=self.device)

            else:
                self.model = pipeline("text-generation", model_path, torch_dtype="auto", device_map="auto")

        if self.task == 'LMM':
            from transformers import AutoTokenizer, AutoProcessor, AutoModel
            from transformers import Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration

            if self.device is not None:
                device_map = self.device
            else:
                device_map = "auto"

            if "Qwen2-VL" in model_path:
                self.model_type = 'qwen2'
                model_class = Qwen2VLForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
                
            elif 'lmms-lab--llava-onevision' in model_path and '-hf' in model_path:  # 这种方式只能调用hf格式的llava-onevision模型。
                self.model_type = 'llava-onevision'
                model_class = LlavaOnevisionForConditionalGeneration
                # 这个里面把device_map和device合二为一了。
                self.model = model_class.from_pretrained(
                    model_path, torch_dtype="auto", device_map=device_map
                )
            
            elif 'MiniCPM-V' in model_path:
                self.model_type = 'minicpm-v'
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_path, trust_remote_code=True,
                    attn_implementation='sdpa', torch_dtype=torch.bfloat16
                )
                self.model.eval()
                if self.device is not None:
                    self.model.to(self.device)
                
            else:
                raise NotImplementError(f"不支持多模态模型: {model_path}")
            
            self.processor = AutoProcessor.from_pretrained(model_path)
    
    def add_text(self, state, text, new_images=None, **kwargs):
        if self.task == 'LLM':
            assert images is None, "语言模型不支持输入图像."
            state.append({"role": "user", "content": text})
            return state
        elif self.task == 'LMM':
            if new_images is not None:
                assert isinstance(new_images, list), "输入的图像应该是一个队列List."
                if self.model_type != 'minicpm-v':
                    state['state'].append(
                        {
                            "role": "user",
                            "content": [
                                    {
                                        "type": "image",
                                    }
                                    for _ in range(len(new_images))
                                ]
                                +
                                [
                                    {"type": "text", "text": text},
                                ],
                        }
                    )
                    state['images'] += new_images
                else:
                    state['state'].append(
                        {
                            "role": "user",
                            "content": new_images + [text]
                        }
                    )
                
            else:
                state['state'].append({"role": "user", "content": text})
            return state
        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")
    
    def new_state(self):
        if self.task == 'LLM':
            return []
        elif self.task == 'LMM':
            return {
                "state": [], 
                "images": []
            }
        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")

    def generate(self, new_text, images=None, state=None,  **generate_kwargs):
        if state is None:
            state = self.new_state()
        
        state = self.add_text(state, new_text, new_images=images)
        if self.task == 'LLM':
            response_message = self.model(state, **generate_kwargs)[0]["generated_text"][-1]
            state.append(response_message)
            return state, response_message['content']
        elif self.task == 'LMM':
            if self.model_type != 'minicpm-v':
                text_prompt = self.processor.apply_chat_template(state['state'], add_generation_prompt=True)
                # print(text_prompt)
                if self.device is None:  # 72B这里写的也是CUDA……有可能是device_map=auto可以这么弄
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to('cuda')
                else:
                    inputs = self.processor(
                        text=[text_prompt], images=state['images'], padding=True, return_tensors="pt"
                    ).to(self.device)

                output_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': output_text[0],
                })
                return state, output_text[0]
            else:
                answer = self.model.chat(
                    image=None,
                    msgs=state['state'], 
                    tokenizer=self.tokenizer
                )
                state['state'].append({
                    'role': 'assistant', 
                    'content': [answer]
                })
                # print(answer)
                return state, answer

        else:
            raise NotImplementError(f"不支持{self.task}的任务类型")
    
    def single_qa(self, inputs, state=None):
        if state is None:
            images=inputs.get("images",None)
            assert images is not None and len(images) == 1, "0-shot时必须有一张待测图像作为输入. "

        qs=inputs.get("text",None)

        generate_kwargs = {
            "temperature": inputs.get("temperature",0.0),
            "top_p":  inputs.get("top_p",0.7),
            "max_new_tokens": inputs.get("max_new_tokens",512),
        }

        return self.generate(qs, state=state, images=images, **generate_kwargs)
    
    def multi_qa(self,inputs,convs,forget_state=False): # maintain train data formation
        state=self.new_state()
        for conv_idx in range(0,len(convs),2):
            cur_qs=convs[conv_idx]["value"].strip()
            inputs["text"]=cur_qs
            if forget_state:
                state=self.new_state()
            state,out=self.single_qa(inputs,state,conv_idx!=0)
            convs[conv_idx+1]["old_value"]=convs[conv_idx+1]["value"]
            convs[conv_idx+1]["value"]=out
        return convs
