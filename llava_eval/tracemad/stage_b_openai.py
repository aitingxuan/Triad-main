# -*- coding: utf-8 -*-
"""Optional Stage-B judge via OpenAI-compatible Chat Completions HTTP API.

No openai SDK required (works with requests). Compatible with:
  - OpenAI official API
  - Azure OpenAI (set base URL + api-key header as configured by your gateway)
  - Local vLLM / Ollama OpenAI-compatible servers
  - Any gateway that implements POST /v1/chat/completions

This repository currently has NO built-in LLM-as-judge pipeline under llava_eval/;
Stage B is therefore optional and disabled unless --stage_b is passed.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import re
import time

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .judge_prompts import build_judge_messages


def _extract_json(text):
    if not text:
        raise ValueError("empty judge response")
    text = text.strip()
    # fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # raw object
    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        return json.loads(text[i : j + 1])
    raise ValueError("no JSON object in judge response")


class OpenAICompatibleJudge(object):
    def __init__(
        self,
        api_key=None,
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        timeout=120,
        max_retries=3,
        temperature=0.0,
    ):
        if requests is None:
            raise ImportError("requests is required for Stage B")
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("TRACEMAD_API_KEY")
        )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip(
            "/"
        )
        self.model = model or os.environ.get("TRACEMAD_JUDGE_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    def chat(self, messages):
        if not self.api_key and "api.openai.com" in self.base_url:
            raise RuntimeError(
                "OPENAI_API_KEY / TRACEMAD_API_KEY not set. "
                "For a local OpenAI-compatible server you may pass a dummy key."
            )
        url = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        "HTTP %s: %s" % (resp.status_code, resp.text[:500])
                    )
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("judge call failed: %s" % last_err)

    def judge_sample(self, sample, stage_a_row):
        messages = build_judge_messages(sample, stage_a_row)
        raw = self.chat(messages)
        scored = _extract_json(raw)
        scored["_raw_judge_text"] = raw
        return scored


def run_stage_b(samples_by_id, stage_a_rows, ids, judge, sleep_s=0.0):
    """Run judge on selected ids. Returns list of merged dicts."""
    row_by_id = {r["id"]: r for r in stage_a_rows}
    out = []
    for i, sid in enumerate(ids):
        sample = samples_by_id.get(sid)
        row = row_by_id.get(sid)
        if sample is None or row is None:
            continue
        try:
            judged = judge.judge_sample(sample, row)
            err = None
        except Exception as e:
            judged = None
            err = str(e)
        rec = {
            "id": sid,
            "subset": row.get("subset"),
            "stage_a_overall": row.get("overall"),
            "stage_b": judged,
            "error": err,
        }
        out.append(rec)
        if sleep_s:
            time.sleep(sleep_s)
        if (i + 1) % 20 == 0:
            print("[TraceMAD StageB] %d/%d" % (i + 1, len(ids)))
    return out


def aggregate_stage_b(stage_b_rows):
    keys = [
        "technical_accuracy",
        "comprehensiveness",
        "trace_logic",
        "schema_clarity",
        "overall",
    ]
    sums = {k: 0.0 for k in keys}
    cnt = {k: 0 for k in keys}
    errors = 0
    for r in stage_b_rows:
        if r.get("error") or not r.get("stage_b"):
            errors += 1
            continue
        jb = r["stage_b"]
        for k in keys:
            if k in jb and isinstance(jb[k], (int, float)):
                sums[k] += float(jb[k])
                cnt[k] += 1
    means = {}
    for k in keys:
        means[k] = (sums[k] / cnt[k]) if cnt[k] else None
    return {"n": len(stage_b_rows), "errors": errors, "means": means}
