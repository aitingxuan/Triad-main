"""Utilities for MMAD MVTec-AD multiple-choice evaluation."""

from __future__ import annotations

import hashlib
import os
import random
import re
from difflib import get_close_matches
from typing import Dict, Iterator, List, Optional, Tuple

# MVTec-AD 15 classes (aligned with CoTM / cotm_v3.py).
MVTEC_AD_CLASSES = {
    "screw", "pill", "capsule", "carpet", "grid", "tile", "wood", "zipper",
    "cable", "toothbrush", "transistor", "metal_nut", "bottle", "hazelnut", "leather",
}

# Official MMAD 0-shot inspector instruction (evaluation/examples/GPT4/gpt4v.py).
MMAD_INSTRUCTION = (
    "You are an industrial inspector who checks products by images. "
    "You should judge whether there is a defect in the query image and answer the questions about it.\n"
    "Answer with the option's letter from the given choices directly."
)

# mmad.json keys use DS-MVTec/... (image/ layout). On disk you may have either:
#   DS-MVTec/bottle/image/broken_large/000.png   (Defect Spectrum, matches mmad.json)
#   MVTec-AD/bottle/test/broken_large/000.png    (official MVTec-AD, needs path mapping)
MMAD_MVTEC_JSON_ROOTS = frozenset({"DS-MVTec", "MVTec-AD", "MVTec"})
DEFAULT_IMAGE_ROOT = "auto"
ON_DISK_IMAGE_ROOTS = ("DS-MVTec", "MVTec-AD")

EXCLUDED_ROOTS = frozenset({
    "DS-MVTec-LOCO",
    "MVTec-LOCO",
    "VisA",
    "GoodsAD",
})


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def parse_mvtec_subset(image_path: str) -> Optional[str]:
    parts = normalize_path(image_path).split("/")
    if len(parts) < 2:
        return None
    root, subset = parts[0], parts[1]
    if root in EXCLUDED_ROOTS:
        return None
    if root not in MMAD_MVTEC_JSON_ROOTS:
        return None
    if subset not in MVTEC_AD_CLASSES:
        return None
    return subset


def is_mvtec_ad_image_path(image_path: str) -> bool:
    return parse_mvtec_subset(image_path) is not None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def iter_image_path_candidates(mmad_json_path: str) -> List[str]:
    """
    Candidate on-disk paths for an mmad.json image key.

    DS-MVTec keys use ``{cls}/image/{defect}/file.png``; official MVTec-AD uses
    ``{cls}/test/{defect}/file.png`` (good may also appear under train/good).
    """
    path = normalize_path(mmad_json_path)
    parts = path.split("/")
    if parse_mvtec_subset(path) is None:
        raise ValueError(f"not an MVTec-AD mmad path: {path}")

    json_root, cls = parts[0], parts[1]
    tail = parts[2:]
    tail_str = "/".join(tail)
    candidates: List[str] = []

    if json_root == "DS-MVTec":
        candidates.append(f"DS-MVTec/{cls}/{tail_str}")
        if len(tail) >= 2 and tail[0] == "image":
            after_image = "/".join(tail[1:])
            candidates.append(f"MVTec-AD/{cls}/test/{after_image}")
            if after_image.startswith("good/"):
                fname = after_image.split("/", 1)[1]
                candidates.append(f"MVTec-AD/{cls}/train/good/{fname}")
                candidates.append(f"MVTec-AD/{cls}/test/good/{fname}")
        elif tail and tail[0] in ("test", "train"):
            candidates.append(f"MVTec-AD/{cls}/{tail_str}")
    else:
        # MVTec-AD / MVTec keys in json
        candidates.append(f"MVTec-AD/{cls}/{tail_str}")
        if len(tail) >= 2 and tail[0] == "image":
            after_image = "/".join(tail[1:])
            candidates.append(f"MVTec-AD/{cls}/test/{after_image}")
            if after_image.startswith("good/"):
                fname = after_image.split("/", 1)[1]
                candidates.append(f"MVTec-AD/{cls}/train/good/{fname}")

    return _dedupe_preserve_order(candidates)


def _order_candidates(candidates: List[str], image_root: str) -> List[str]:
    if not image_root or image_root == "auto":
        return candidates
    preferred = [c for c in candidates if c.split("/")[0] == image_root]
    others = [c for c in candidates if c.split("/")[0] != image_root]
    return preferred + others


def resolve_existing_image_path(
    mmad_root: str,
    mmad_json_path: str,
    image_root: str = DEFAULT_IMAGE_ROOT,
) -> Tuple[Optional[str], List[str]]:
    """Return the first existing relative image path, plus all tried candidates."""
    candidates = _order_candidates(iter_image_path_candidates(mmad_json_path), image_root)
    for rel in candidates:
        if os.path.isfile(os.path.join(mmad_root, rel)):
            return rel, candidates
    return None, candidates


def detect_image_root(mmad_root: str) -> str:
    """Pick DS-MVTec if present (matches mmad.json), else MVTec-AD."""
    ds_probe = os.path.join(mmad_root, "DS-MVTec", "bottle")
    ad_probe = os.path.join(mmad_root, "MVTec-AD", "bottle")
    if os.path.isdir(ds_probe):
        return "DS-MVTec"
    if os.path.isdir(ad_probe):
        return "MVTec-AD"
    return "DS-MVTec"


def stable_shuffle_options(
    options: Dict[str, str], answer_key: str, seed_key: str
) -> Tuple[Dict[str, str], str]:
    """Shuffle options with a deterministic seed (reproducible, MMAD-style random order)."""
    items = list(options.items())
    seed = int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(items)

    shuffled: Dict[str, str] = {}
    new_answer = None
    for idx, (orig_key, value) in enumerate(items):
        letter = chr(65 + idx)
        shuffled[letter] = value
        if orig_key == answer_key:
            new_answer = letter
    if new_answer is None:
        raise ValueError(f"answer key {answer_key!r} not found in options {options}")
    return shuffled, new_answer


def build_options_text(options: Dict[str, str]) -> str:
    lines = []
    for letter in sorted(options.keys()):
        lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def build_human_prompt(question: str, options: Dict[str, str], zero_shot: bool = True) -> str:
    """Build the human turn for Triad evaluate.py (<image> token added by loader if missing)."""
    parts = [MMAD_INSTRUCTION, ""]
    if zero_shot:
        parts.append("Test image:")
    parts.append("<image>")
    parts.append(question.strip())
    parts.append(build_options_text(options))
    return "\n".join(parts)


def _match_option_by_keywords(inner: str, options: Dict[str, str], keywords: set) -> Optional[str]:
    inner_l = inner.lower().strip()
    if inner_l not in keywords:
        return None
    for letter, opt in options.items():
        opt_l = str(opt).lower()
        if any(k in opt_l for k in keywords):
            return letter
    return None


def parse_mmad_answer(response_text: str, options: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Parse model output to a choice letter (A–E).
    Supports MMAD official letter extraction, option-text fuzzy match, and CoTM/IAD-R1 tags.
    """
    if not response_text:
        return None

    text = response_text.strip()
    options = options or {}

    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        inner = tag_match.group(1).strip()
        for candidate in (inner, re.sub(r"<[^>]+>", "", inner).strip()):
            letter = _parse_letter_or_option_text(candidate, options)
            if letter:
                return letter

    letter = _parse_letter_or_option_text(text, options)
    if letter:
        return letter

    if options:
        values = [str(v) for v in options.values()]
        closest = get_close_matches(text, values, n=1, cutoff=0.0)
        if closest:
            target = closest[0]
            for k, v in options.items():
                if str(v) == target:
                    return k
    return None


def _parse_letter_or_option_text(text: str, options: Dict[str, str]) -> Optional[str]:
    text = text.strip()
    if not text:
        return None

    upper = text.upper()
    if len(upper) == 1 and upper in "ABCDE":
        return upper

    for pat in (
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-E])\b",
        r"\(([A-E])\)",
        r"^([A-E])[\.\:\)]",
    ):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    letters = re.findall(r"\b([A-E])\b", text.upper())
    if letters:
        return letters[0]

    text_l = text.lower()
    for letter, opt in options.items():
        opt_l = str(opt).lower().strip()
        if not opt_l:
            continue
        if text_l == opt_l or text_l in opt_l or opt_l in text_l:
            return letter

    yes_kw = {"yes", "abnormal", "defect", "defective", "anomaly", "fault", "faulty"}
    no_kw = {"no", "normal", "good", "non-defective", "non defective", "ok"}
    if text_l in yes_kw or any(w in text_l for w in yes_kw):
        hit = _match_option_by_keywords(text_l, options, yes_kw)
        if hit:
            return hit
    if text_l in no_kw or any(w in text_l for w in no_kw):
        hit = _match_option_by_keywords(text_l, options, no_kw)
        if hit:
            return hit

    return None


def iter_conversation_items(record: dict) -> Iterator[dict]:
    """Yield QA dicts from an MMAD mmad.json record."""
    for key, value in record.items():
        if key.startswith("conversation") and isinstance(value, list):
            for qa in value:
                if isinstance(qa, dict) and "Question" in qa:
                    yield qa


def normalize_question_type(qtype: str) -> str:
    if qtype in ("Object Structure", "Object Details"):
        return "Object Analysis"
    return qtype
