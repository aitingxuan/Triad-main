# -*- coding: utf-8 -*-
"""Stage B LLM-judge prompts (RobustMAD-inspired, CoT-aware)."""
from __future__ import absolute_import, division, print_function, unicode_literals

JUDGE_SYSTEM = (
    "You are an expert industrial anomaly-inspection evaluator. "
    "Score the model's tagged diagnostic report against the ground truth. "
    "Be strict about hallucinations, format violations, and think/answer contradictions. "
    "Return ONLY valid JSON matching the required schema."
)

JUDGE_USER_TEMPLATE = """Evaluate the following industrial anomaly diagnosis.

## Object / subset
{subset}

## Ground truth
- defect present (gt): {gt_yes_no}
- defect folder / type label: {defect_folder}
- origin path: {origin_path}

## Production process context (from the user prompt)
{process_excerpt}

## Model raw output
{model_output}

## Parsed fields
- think: {think}
- location: {location}
- type: {type}
- answer: {answer}

## Stage-A automatic hints (may be imperfect)
- decision_correct: {decision_correct}
- schema_issues: {schema_issues}
- failure_mode_hints: {failure_modes}

## Scoring rubric (integers 1-5)
1. technical_accuracy: factual correctness vs image/GT (object, defect presence, type, location); punish hallucinations.
2. comprehensiveness: for defect=Yes, needs presence + location + type + enough evidence; for No, clear exclusion with observation.
3. trace_logic: think chain quality — evidence -> process attribution -> conclusion; think must entail location/type/answer.
4. schema_clarity: exact tag format, no extra text outside tags, correct Yes/No field usage.
5. overall: holistic score; if technical_accuracy or comprehensiveness <=2, overall must be <=3.

Also assign failure_modes from:
fragile_grounding, incomplete_diagnosis, weak_logical_grounding, schema_drift,
process_hallucination, overconfident_normal
(empty list if none).

Return JSON:
{{
  "technical_accuracy": 1,
  "comprehensiveness": 1,
  "trace_logic": 1,
  "schema_clarity": 1,
  "overall": 1,
  "think_answer_consistent": true,
  "process_attribution_ok": true,
  "location_plausible": true,
  "type_plausible": true,
  "failure_modes": [],
  "brief_rationale": "one or two sentences"
}}
"""


def extract_process_excerpt(human_prompt, max_chars=1800):
    if not human_prompt:
        return "(none)"
    # drop image token noise
    text = human_prompt.replace("<image>", "[image]")
    if "The following is the production process" in text:
        i = text.index("The following is the production process")
        text = text[i:]
    if "Question:" in text:
        text = text.split("Question:")[0]
    text = text.strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def build_judge_messages(sample, stage_a_row, parsed=None):
    human = ""
    gpt = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            human = turn.get("value") or ""
        elif turn.get("from") == "gpt":
            gpt = turn.get("value") or ""
    if parsed is None:
        parsed = (stage_a_row or {}).get("parsed") or {}

    gt = int(sample.get("gt", 0))
    user = JUDGE_USER_TEMPLATE.format(
        subset=sample.get("subset"),
        gt_yes_no="Yes" if gt == 1 else "No",
        defect_folder=(stage_a_row or {}).get("defect_folder"),
        origin_path=sample.get("origin_path"),
        process_excerpt=extract_process_excerpt(human),
        model_output=gpt,
        think=parsed.get("think"),
        location=parsed.get("location"),
        type=parsed.get("type"),
        answer=parsed.get("answer_raw"),
        decision_correct=(stage_a_row or {}).get("decision_correct"),
        schema_issues=(stage_a_row or {}).get("schema_issues"),
        failure_modes=(stage_a_row or {}).get("failure_modes"),
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
