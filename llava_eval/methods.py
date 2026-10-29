"""Active decision methods: Defaultv2 and ReAL attention-crop.

Only these two experimental schemes remain.  Both share the same v0 main
question, band gate, generic checklist, hard vote and fallback.  ``real_crop_v2``
adds one thing: on hard examples, cotm_v3 is generated *only* to propose an
attention crop box; its Yes/No answer never decides the final label.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from attn_crop import propose_box_real
from egroi_box import expand_box_eg_roi
from hyp_checklists import DEFAULT_HYP_CHECKLIST, format_subq
from llava.constants import DEFAULT_IMAGE_TOKEN


@dataclass
class MethodConfig:
    band_lo: float = 0.4
    band_hi: float = 0.7
    tau_ev: float = 0.7
    delta: float = 1.0
    max_subq: int = 3
    cotm_max_new_tokens: int = 512

    crop_layers: str = "midband"
    crop_keep_frac: float = 0.5
    crop_alpha: float = 0.6
    crop_beta: float = 0.4
    crop_tau_t: float = 0.0
    crop_tau_i: float = 0.0
    crop_view: str = "anyres"
    crop_thr_q: float = 0.90
    crop_expand: float = 1.3
    crop_side_lo: float = 0.06
    crop_side_hi: float = 0.45
    crop_conc_floor: float = 0.30
    crop_box_expand: str = "none"
    crop_box_min_side: int = 384
    crop_overlay_dir: Optional[str] = None
    crop_dir: Optional[str] = None


DEFAULT_V2_CFG = MethodConfig()
REAL_CROP_V2_CFG = MethodConfig()


@dataclass
class Decision:
    letter: str
    source: str
    main_pA: float
    main_letter: str
    in_band: bool
    S_A: float = 0.0
    S_B: float = 0.0
    sub_answers: List[Dict[str, Any]] = field(default_factory=list)
    crop: Optional[Dict[str, Any]] = None
    cotm: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_name(value: Any, limit: int = 100) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value or "sample"))
    return text[:limit]


def _in_band(p_a: float, cfg: MethodConfig) -> bool:
    return float(cfg.band_lo) <= float(p_a) <= float(cfg.band_hi)


def _build_subq_prompt(worker, question: str) -> str:
    body = DEFAULT_IMAGE_TOKEN + "\n" + format_subq(question)
    return worker.build_prompt(body, add_image_if_missing=False)


def _vote(
    sub_answers: Sequence[Dict[str, Any]],
    main_letter: str,
    delta: float,
) -> tuple[str, str, float, float]:
    s_a = 0.0
    s_b = 0.0
    for rec in sub_answers:
        if not rec.get("kept"):
            continue
        if rec.get("hyp") == "H_A":
            if rec.get("letter") == "A":
                s_a += 1.0
            else:
                s_b += 1.0
        else:
            if rec.get("letter") == "A":
                s_b += 1.0
            else:
                s_a += 1.0
    if s_a - s_b >= float(delta):
        return "A", "A", s_a, s_b
    if s_b - s_a >= float(delta):
        return "B", "B", s_a, s_b
    return main_letter, "fallback", s_a, s_b


def _run_checklist(
    worker,
    image_args: Dict[str, Any],
    cfg: MethodConfig,
    view_tag: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for hyp, questions in (
        ("H_A", DEFAULT_HYP_CHECKLIST["H_A"]),
        ("H_B", DEFAULT_HYP_CHECKLIST["H_B"]),
    ):
        for question in list(questions)[: max(0, int(cfg.max_subq))]:
            result = worker.score_ab(_build_subq_prompt(worker, question), image_args)
            p_a = float(result["p_A"])
            p_b = float(result["p_B"])
            records.append(
                {
                    "hyp": hyp,
                    "q": question,
                    "p_A": p_a,
                    "p_B": p_b,
                    "letter": "A" if p_a >= 0.5 else "B",
                    "conf": max(p_a, p_b),
                    "kept": max(p_a, p_b) >= float(cfg.tau_ev),
                    "view": view_tag,
                }
            )
    return records


def run_default_v2(worker, sample: Dict[str, Any], cfg: MethodConfig) -> Decision:
    image_args, _, _ = worker.prepare_view(
        sample["images"],
        sample.get("masks"),
        sample.get("boxes_list"),
        use_pad_when_db=False,
    )
    main = worker.score_ab(
        worker.build_prompt(sample["v0_prompt"], add_image_if_missing=True),
        image_args,
    )
    main_p_a = float(main["p_A"])
    main_letter = str(main["letter"])

    if not _in_band(main_p_a, cfg):
        return Decision(
            letter=main_letter,
            source="main_out_of_band",
            main_pA=main_p_a,
            main_letter=main_letter,
            in_band=False,
        )

    sub_answers = _run_checklist(worker, image_args, cfg, "full")
    letter, vote_src, s_a, s_b = _vote(sub_answers, main_letter, cfg.delta)
    return Decision(
        letter=letter,
        source=f"checklist_{vote_src}",
        main_pA=main_p_a,
        main_letter=main_letter,
        in_band=True,
        S_A=s_a,
        S_B=s_b,
        sub_answers=sub_answers,
    )


def _parse_cotm_answer(text: str) -> Optional[str]:
    match = re.search(r"<answer>\s*(Yes|No)\s*</answer>", text or "", re.I)
    return match.group(1) if match else None


def _propose_crop(sample: Dict[str, Any], worker, image_args, cotm_input_ids, gen_ids, cfg):
    overlay_path = None
    if cfg.crop_overlay_dir:
        os.makedirs(cfg.crop_overlay_dir, exist_ok=True)
        overlay_path = os.path.join(
            cfg.crop_overlay_dir,
            f"{_safe_name(sample.get('subset'), 40)}__{_safe_name(sample.get('sample_id'))}.png",
        )
    try:
        meta = propose_box_real(
            worker.model.model,
            worker.model.tokenizer,
            image_args["images"],
            sample["images"][0],
            image_args["image_sizes"],
            image_args.get("roi_bboxes"),
            cotm_input_ids,
            gen_ids,
            layers=str(cfg.crop_layers),
            keep_frac=float(cfg.crop_keep_frac),
            alpha=float(cfg.crop_alpha),
            beta=float(cfg.crop_beta),
            tau_t=float(cfg.crop_tau_t),
            tau_i=float(cfg.crop_tau_i),
            view_name=str(cfg.crop_view),
            thr_q=float(cfg.crop_thr_q),
            expand=float(cfg.crop_expand),
            side_lo=float(cfg.crop_side_lo),
            side_hi=float(cfg.crop_side_hi),
            conc_floor=float(cfg.crop_conc_floor),
            overlay_path=overlay_path,
            overlay_title=f"{sample.get('sample_id')} | {sample.get('subset')} | real_crop_v2",
        )
    except Exception as exc:  # crop proposal must never break evaluation
        return {"kept": False, "box": None, "reason": f"error:{exc!r}", "proposer": "real"}

    box = meta.get("box") if meta.get("kept") else None
    if box is not None and str(cfg.crop_box_expand).lower() == "egroi":
        box = expand_box_eg_roi(
            box,
            (sample["images"][0].width, sample["images"][0].height),
            min_side=float(cfg.crop_box_min_side),
        )
        meta["box_expanded"] = [round(float(v), 2) for v in box]
    meta["box"] = box
    meta["used_crop"] = box is not None
    return meta


def _save_crop(sample: Dict[str, Any], box, cfg: MethodConfig) -> None:
    if not cfg.crop_dir or box is None:
        return
    try:
        os.makedirs(cfg.crop_dir, exist_ok=True)
        x0, y0, x1, y1 = [int(round(float(v))) for v in box[:4]]
        sample["images"][0].crop((x0, y0, x1, y1)).save(
            os.path.join(
                cfg.crop_dir,
                f"{_safe_name(sample.get('subset'), 40)}__{_safe_name(sample.get('sample_id'))}.png",
            )
        )
    except Exception:
        pass


def run_real_crop_v2(worker, sample: Dict[str, Any], cfg: MethodConfig) -> Decision:
    image_args, _, _ = worker.prepare_view(
        sample["images"],
        sample.get("masks"),
        sample.get("boxes_list"),
        use_pad_when_db=False,
    )
    main = worker.score_ab(
        worker.build_prompt(sample["v0_prompt"], add_image_if_missing=True),
        image_args,
    )
    main_p_a = float(main["p_A"])
    main_letter = str(main["letter"])

    if not _in_band(main_p_a, cfg):
        return Decision(
            letter=main_letter,
            source="main_out_of_band",
            main_pA=main_p_a,
            main_letter=main_letter,
            in_band=False,
        )

    cotm_text = ""
    cotm_answer = None
    gen_ids: List[int] = []
    box = None
    crop_meta: Dict[str, Any] = {
        "kept": False,
        "box": None,
        "reason": "cotm_not_run",
        "used_crop": False,
    }
    cotm_meta: Dict[str, Any] = {"used": False}

    try:
        cotm_prompt = worker.build_prompt(sample["cotm_prompt"], add_image_if_missing=False)
        cotm_text, gen_ids, cotm_input_ids = worker.generate_text(
            cotm_prompt,
            image_args,
            max_new_tokens=int(cfg.cotm_max_new_tokens),
        )
        cotm_answer = _parse_cotm_answer(cotm_text)
        cotm_meta = {
            "used": True,
            "answer": cotm_answer,
            "n_tokens": len(gen_ids),
            "hit_max_tokens": len(gen_ids) >= int(cfg.cotm_max_new_tokens),
        }
        crop_meta = _propose_crop(
            sample, worker, image_args, cotm_input_ids, gen_ids, cfg
        )
        box = crop_meta.get("box")
    except Exception as exc:
        crop_meta = {"kept": False, "box": None, "reason": f"cotm_error:{exc!r}", "proposer": "real"}
        cotm_meta = {"used": False, "error": repr(exc), "answer": cotm_answer}

    if box is not None:
        try:
            _save_crop(sample, box, cfg)
            crop_args, _, _ = worker.prepare_view(
                sample["images"],
                sample.get("masks"),
                [[box]],
                use_pad_when_db=False,
            )
            sub_answers = _run_checklist(worker, crop_args, cfg, "real_crop")
            letter, vote_src, s_a, s_b = _vote(sub_answers, main_letter, cfg.delta)
            return Decision(
                letter=letter,
                source=f"crop_checklist_{vote_src}",
                main_pA=main_p_a,
                main_letter=main_letter,
                in_band=True,
                S_A=s_a,
                S_B=s_b,
                sub_answers=sub_answers,
                crop=crop_meta,
                cotm=cotm_meta,
            )
        except Exception as exc:
            crop_meta = dict(crop_meta)
            crop_meta["crop_error"] = repr(exc)
            crop_meta["used_crop"] = False

    # No usable box: exactly Defaultv2 on the original view.
    sub_answers = _run_checklist(worker, image_args, cfg, "full")
    letter, vote_src, s_a, s_b = _vote(sub_answers, main_letter, cfg.delta)
    return Decision(
        letter=letter,
        source=f"crop_fail_checklist_{vote_src}",
        main_pA=main_p_a,
        main_letter=main_letter,
        in_band=True,
        S_A=s_a,
        S_B=s_b,
        sub_answers=sub_answers,
        crop=crop_meta,
        cotm=cotm_meta,
    )
