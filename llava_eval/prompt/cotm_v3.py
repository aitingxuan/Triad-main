"""CoTMv3-aligned eval prompts: open-ended questions + production process (no A/B constraint).

Matches the human-turn format in CoTMv3.json so models fine-tuned on that data
can produce long IAD-R1 style outputs:
  <think>...</think><location>...</location><type>...</type><answer>Yes/No</answer>

Usage:
  python llava_eval/evaluate.py ... --text_src config:cotm_v3 --max_new_tokens 1024
"""

from .mvtec import MVTEC_PROMPT_V1_2_PROD, MVTEC_PROMPT_V1_PROD

# Subset keys used by evaluate.py (--subset cable / banana_juice / ...)
MVTEC_SUBSETS = (
    "screw", "pill", "capsule", "carpet", "grid", "tile", "wood", "zipper",
    "cable", "toothbrush", "transistor", "metal_nut", "bottle", "hazelnut", "leather",
)


def _mvtec_prod_context(subset: str) -> str:
    """Prefer v3 process text; fall back to v1 for carpet/grid/leather."""
    if subset in MVTEC_PROMPT_V1_2_PROD:
        return MVTEC_PROMPT_V1_2_PROD[subset]
    if subset in MVTEC_PROMPT_V1_PROD:
        return MVTEC_PROMPT_V1_PROD[subset]
    raise KeyError(f"no production process for mvtec subset: {subset}")

# Fixed intro/question pair aligned with the most frequent CoTMv3 training templates.
COTMV3_INTRO = (
    "Referencing the image and production process that are shown below, please answer the question:\n"
    "Image:\n"
    "<image>\n"
)

COTMV3_QUESTION = (
    "Question: Do you notice any faults or peculiarities with the product in this image?\n"
    "You must reply in this exact tag format (no extra text outside tags):\n"
    "<think>...</think><location>...</location><type>...</type><answer>Yes</answer>\n"
    "or, if there is no defect:\n"
    "<think>...</think><answer>No</answer>\n"
    "In <think>, reason from the production process and visible evidence. "
    "Use <location> and <type> only when the answer is Yes."
)

COTMV3_LOCO_PROD = {
    "banana_juice": (
        "The following is the production process of a defect-free banana juice:\n"
        "1. Banana Selection and Preparation:\n"
        "   - Select ripe bananas that will yield a rich flavor and consistent texture.\n"
        "   - Wash and peel the bananas to prepare them for juicing.\n"
        "2. Juicing:\n"
        "   - Crush and press the peeled bananas to extract the juice.\n"
        "   - Filter the juice to achieve a smooth, consistent texture without solid particles, ensuring it appears opaque and white.\n"
        "3. Pasteurization:\n"
        "   - Heat the banana juice to a specific temperature to sterilize and halt enzymatic activity, which helps in maintaining color and extending shelf life.\n"
        "4. Bottle Preparation:\n"
        "   - Clean and sterilize the glass bottles to prepare for filling. This ensures that the bottles are clear and free of contaminants, affecting the visual clarity and cleanliness.\n"
        "5. Filling:\n"
        "   - Fill the sterilized glass bottles with the pasteurized banana juice.\n"
        "   - Seal the bottles with screw-top lids to prevent contamination and preserve the juice.\n"
        "6. Label Design and Application:\n"
        "   - Design and print two labels: one featuring the banana illustration and another with the text \"100% Juice.\" Both labels have a yellow background.\n"
        "   - Apply the labels to the bottles, ensuring they are straight and well-aligned for a professional appearance.\n"
    ),
    "cherry_juice": (
        "The following is the production process of a defect-free cherry juice:\n"
        "1. Cherry Juice Production:\n"
        "   - Cherry Selection: Fresh cherries are selected for their color and ripeness..\n"
        "   - Juice Extraction: The cherries are crushed, and the juice is extracted, filtered, and pasteurized. These steps ensure the juice is clear, smooth, and visually appealing.\n"
        "2. Glass Bottle Production:\n"
        "   - Molding: The glass is molded into the transparent bottle shape, ensuring clarity so the juice is fully visible.\n"
        "   - Surface Finish: The glass is finished to be smooth and free of imperfections, contributing to the overall clean appearance of the bottle.\n"
        "3. Label Application:\n"
        "   - The labels are applied using an automatic labeling machine that precisely places each label onto the bottle's surface. The labels are smoothed to avoid wrinkles or bubbles. The top label is printed with the cherry image. The label with \"100% Juice\" text is attached on the bottom of the bottle.\n"
        "4. Filling the Bottle:\n"
        "   - Juice Filling: The cherry juice is filled to a consistent level, leaving a slight foam layer at the top. This maintains the clean visual presentation of the liquid.\n"
    ),
    "orange_juice": (
        "The following is the production process of a defect-free orange juice:\n"
        "1. Juice Production:\n"
        "- Juicing: Oranges are pressed to extract the juice, which has the distinct orange color.\n"
        "- Filtering: The juice is filtered to ensure a smooth and consistent appearance.\n"
        "- Blending: Juice from different batches is blended to maintain a uniform color and texture.\n"
        "- Cooling: The juice is cooled, preserving its visual freshness.\n"
        "2. Glass Bottle Production:\n"
        "- Bottle Forming: The molten glass is shaped into the clear bottle, influencing the transparency and appearance of the final product.\n"
        "- Annealing: The bottle is slowly cooled to ensure a smooth, durable, and clear finish.\n"
        "3. Label Production:\n"
        "- Label Printing: Two labels are printed—one featuring an orange illustration (top) and another with \"100% Juice\" text (bottom), both impacting the branding and visual design.\n"
        "4. Assembly Process:\n"
        "- Bottle Filling: The glass bottle is filled with the orange juice, showcasing the juice's color and consistency.\n"
        "- Label Application: The bottle moves through an automated labeling machine that applies the labels precisely. The machine uses rollers to ensure the labels are evenly adhered to the glass. The label with the orange image is applied to the upper portion of the bottle's body, centered for optimal visibility. The label with \"100% Juice\" text is applied slightly lower, aligned neatly beneath the top label. The machine applies uniform pressure to ensure that the labels are securely attached without wrinkles or misalignment, resulting in a smooth and visually appealing finish.\n"
    ),
}


def _build_cotmv3_prompt(context: str) -> str:
    return COTMV3_INTRO + context + COTMV3_QUESTION


# MVTec AD (15 classes): reuse v3 production process text, v1 fallback for carpet/grid/leather.
COTMV3_MVTEC_PROMPT = {
    k: _build_cotmv3_prompt(_mvtec_prod_context(k))
    for k in MVTEC_SUBSETS
}

# MVTec-LOCO bottle subsets used in CoTMv3.
COTMV3_LOCO_PROMPT = {
    k: _build_cotmv3_prompt(v) for k, v in COTMV3_LOCO_PROD.items()
}

# Unified lookup for evaluate.py subset routing (mvtec AD + LOCO juice).
COTMV3_PROMPT = {**COTMV3_MVTEC_PROMPT, **COTMV3_LOCO_PROMPT}
