# -*- coding: utf-8 -*-
"""Parse CoTMv3-style tagged GPT outputs."""
from __future__ import absolute_import, division, print_function, unicode_literals

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_LOC_RE = re.compile(r"<location>(.*?)</location>", re.IGNORECASE | re.DOTALL)
_TYPE_RE = re.compile(r"<type>(.*?)</type>", re.IGNORECASE | re.DOTALL)
_ANS_RE = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?(?:think|location|type|answer)\b", re.IGNORECASE)


def _strip(s):
    if s is None:
        return None
    return s.strip()


def parse_gpt_output(text):
    """Parse a model reply into structured fields.

    Returns dict with keys:
      think, location, type, answer (0/1/None), answer_raw,
      has_think/location/type/answer, preamble, trailing,
      tag_order, starts_with_tag
    """
    if text is None:
        text = ""
    raw = text
    text = text.strip()

    think_m = _THINK_RE.search(text)
    loc_m = _LOC_RE.search(text)
    type_m = _TYPE_RE.search(text)
    ans_m = _ANS_RE.search(text)

    think = _strip(think_m.group(1)) if think_m else None
    location = _strip(loc_m.group(1)) if loc_m else None
    typ = _strip(type_m.group(1)) if type_m else None
    answer_raw = ans_m.group(1) if ans_m else None
    answer = None
    if answer_raw is not None:
        answer = 1 if answer_raw.lower() == "yes" else 0

    # preamble: text before first known tag
    first_tag = _ANY_TAG_RE.search(text)
    preamble = ""
    if first_tag is None:
        preamble = text
    elif first_tag.start() > 0:
        preamble = text[: first_tag.start()].strip()

    # trailing after last </answer> or last closing tag
    trailing = ""
    last_close = None
    for m in re.finditer(r"</(?:think|location|type|answer)>", text, re.IGNORECASE):
        last_close = m
    if last_close is not None and last_close.end() < len(text):
        trailing = text[last_close.end() :].strip()

    # tag order by first occurrence of opening tags
    order = []
    for name, m in (
        ("think", think_m),
        ("location", loc_m),
        ("type", type_m),
        ("answer", ans_m),
    ):
        if m is not None:
            order.append((m.start(), name))
    order.sort(key=lambda x: x[0])
    tag_order = [n for _, n in order]

    starts_with_tag = bool(re.match(r"\s*<", raw))

    return {
        "think": think,
        "location": location,
        "type": typ,
        "answer": answer,
        "answer_raw": answer_raw,
        "has_think": think_m is not None,
        "has_location": loc_m is not None and bool(location),
        "has_type": type_m is not None and bool(typ),
        "has_answer": ans_m is not None,
        "preamble": preamble,
        "trailing": trailing,
        "tag_order": tag_order,
        "starts_with_tag": starts_with_tag,
        "raw": raw,
    }


def extract_defect_folder(origin_path):
    """MVTec path like subset/test/defect_type/xxx.png -> defect_type."""
    if not origin_path:
        return None
    parts = origin_path.replace("\\", "/").split("/")
    # .../subset/test/<defect>/file.png
    if len(parts) >= 3 and parts[-3] == "test":
        return parts[-2]
    if len(parts) >= 2:
        return parts[-2]
    return None
