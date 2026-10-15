"""
Fixed-template bidirectional checklists for scheme_requery hyp (方案 3).

H_A: visual checks that support "defective" when answered Yes.
H_B: visual checks that support "normal" when answered Yes.
Questions must be yes/no and verifiable on the image.
"""

SUBQ_SUFFIX = (
    "A. Yes\n"
    "B. No\n"
    "Answer with the option's letter from the given choices directly."
)

# MVTec AD subsets (keys match evaluation subset names).
MVTEC_HYP_CHECKLISTS = {
    "bottle": {
        "H_A": [
            "Is there a crack, chip, or broken edge on the glass bottle?",
            "Is there contamination or foreign residue on the bottle surface?",
        ],
        "H_B": [
            "Is the bottle body smooth and free of cracks or chips?",
            "Is the bottle surface clean without stains or foreign matter?",
        ],
    },
    "cable": {
        "H_A": [
            "Is any wire bent, cut, missing, or swapped compared with a normal cable?",
            "Is the insulation poked, cut, or otherwise damaged?",
        ],
        "H_B": [
            "Are all conductors present and neatly arranged without cuts?",
            "Is the insulation continuous and intact along the cable?",
        ],
    },
    "capsule": {
        "H_A": [
            "Is there a crack, poke, scratch, or squeeze mark on the capsule?",
            "Is the imprint faulty, missing, or clearly distorted?",
        ],
        "H_B": [
            "Is the capsule shell smooth and unbroken?",
            "Is the printed imprint clear and correctly aligned?",
        ],
    },
    "carpet": {
        "H_A": [
            "Is there a cut, hole, or thread defect in the carpet weave?",
            "Is there a color stain or metal/foreign contamination on the carpet?",
        ],
        "H_B": [
            "Is the carpet weave continuous and evenly textured?",
            "Is the carpet color uniform without stains or foreign objects?",
        ],
    },
    "grid": {
        "H_A": [
            "Is any grid wire bent, broken, or interrupted?",
            "Is there glue residue, thread, or metal contamination on the grid?",
        ],
        "H_B": [
            "Is the grid pattern regular and unbroken?",
            "Is the grid surface clean without glue or foreign debris?",
        ],
    },
    "hazelnut": {
        "H_A": [
            "Is there a crack, cut, or hole on the hazelnut shell?",
            "Is there an abnormal print mark or surface damage on the nut?",
        ],
        "H_B": [
            "Is the hazelnut shell intact without cracks or holes?",
            "Is the shell surface clean and free of abnormal marks?",
        ],
    },
    "leather": {
        "H_A": [
            "Is there a cut, fold, poke, or glue mark on the leather?",
            "Is there a clear color stain or discoloration on the leather?",
        ],
        "H_B": [
            "Is the leather surface smooth without cuts, folds, or punctures?",
            "Is the leather color uniform without stains or glue marks?",
        ],
    },
    "metal_nut": {
        "H_A": [
            "Is the metal nut bent, flipped, or geometrically distorted?",
            "Is there a scratch or abnormal color spot on the nut?",
        ],
        "H_B": [
            "Is the nut shape regular with a flat, undistorted face?",
            "Is the metal surface clean without scratches or discoloration?",
        ],
    },
    "pill": {
        "H_A": [
            "Is there a crack, scratch, or contamination on the pill?",
            "Is the imprint faulty, or does the pill color/type look abnormal?",
        ],
        "H_B": [
            "Is the pill body intact without cracks or scratches?",
            "Is the pill color and imprint clear and consistent?",
        ],
    },
    "screw": {
        "H_A": [
            "Is there a scratch on the screw head or neck?",
            "Is the thread damaged, incomplete, or manipulated at the tip?",
        ],
        "H_B": [
            "Is the screw thread continuous and evenly spaced?",
            "Is the head and shank free of scratches or dents?",
        ],
    },
    "tile": {
        "H_A": [
            "Is there a crack, rough patch, oil stain, or gray stroke on the tile?",
            "Is there a glue strip or foreign mark on the tile surface?",
        ],
        "H_B": [
            "Is the tile surface smooth and free of cracks or rough areas?",
            "Is the tile pattern clean without oil, glue, or stroke marks?",
        ],
    },
    "toothbrush": {
        "H_A": [
            "Are any bristles bent, broken, irregular, or missing?",
            "Is the toothbrush head deformed or clearly defective?",
        ],
        "H_B": [
            "Are the bristles straight, evenly spaced, and complete?",
            "Is the toothbrush head shape regular and undamaged?",
        ],
    },
    "transistor": {
        "H_A": [
            "Is any lead bent, cut, or missing on the transistor?",
            "Is the case damaged, or is the transistor misplaced on the board?",
        ],
        "H_B": [
            "Are the leads straight, intact, and properly attached?",
            "Is the transistor case intact and correctly positioned?",
        ],
    },
    "wood": {
        "H_A": [
            "Is there a hole, scratch, or liquid stain on the wood?",
            "Is there an abnormal color spot or combined surface defect?",
        ],
        "H_B": [
            "Is the wood surface intact without holes or deep scratches?",
            "Is the wood color/grain consistent without liquid stains?",
        ],
    },
    "zipper": {
        "H_A": [
            "Are any zipper teeth broken, split, squeezed, or misaligned?",
            "Is the zipper fabric torn, or are the teeth rough/damaged?",
        ],
        "H_B": [
            "Are the zipper teeth complete, aligned, and evenly spaced?",
            "Is the zipper tape intact without tears or rough damage?",
        ],
    },
}

# Fallback / force_default: generic visual checks (no class name, no dataset defect taxonomy).
# Covers surface damage, contamination, and geometry/arrangement (bent, missing, misplaced).
# Prefer max_subq>=3 so the geometry pair is actually asked (see scheme_requery_default_v2).
DEFAULT_HYP_CHECKLIST = {
    "H_A": [
        "Is there a scratch, crack, dent, or hole on the object surface?",
        "Is there contamination, stain, or foreign matter on the object?",
        "Is any part of the object missing, misplaced, bent, or geometrically distorted?",
    ],
    "H_B": [
        "Is the object surface continuous and free of local damage?",
        "Is the object clean and uniform without stains or foreign matter?",
        "Are all parts present, correctly placed, and free of geometric distortion?",
    ],
}

# Defv3 (v3-b): contrastive / relative checks — no class names, no defect-type taxonomy
# (avoid scratch/crack/stain words). H_A vs H_B are opposing continuity judgments.
# Prefer max_subq>=3 (see scheme_requery_default_v3).
DEFAULT_HYP_CHECKLIST_V3 = {
    "H_A": [
        "Does any local region look discontinuous or broken relative to its surroundings?",
        "Does any local region look inconsistent in color or texture compared with nearby areas?",
        "Does any part look missing, misplaced, bent, or out of alignment with the rest?",
    ],
    "H_B": [
        "Do local regions look continuous and consistent with their surroundings?",
        "Is color and texture locally consistent with nearby areas?",
        "Do all visible parts look present, aligned, and free of geometric distortion?",
    ],
}


def uncertainty_from_p_a(p_a: float) -> float:
    """Margin-based uncertainty in [0, 1]; max at p_A=0.5."""
    return 1.0 - abs(2.0 * float(p_a) - 1.0)


def uncertainty_quantile_threshold(u_scores, top_frac: float) -> float:
    """Return u_thr so that samples with u >= u_thr are ~top_frac most uncertain.

    top_frac in (0, 1]: fraction of the run to requery. No absolute p_A band.
    """
    import numpy as np

    vals = [float(u) for u in (u_scores or [])]
    if not vals:
        return float("inf")
    tf = float(top_frac)
    if tf <= 0.0:
        return float("inf")
    if tf >= 1.0:
        return float(min(vals))
    return float(np.quantile(np.asarray(vals, dtype=np.float64), 1.0 - tf))


def get_hyp_checklist(subset: str, force_default: bool = False, checklist: str = None):
    """Return {"H_A": [...], "H_B": [...]} for subset.

    checklist: "per_class" | "default" | "default_v3".
    force_default=True is legacy alias for checklist="default".
    """
    if checklist is None:
        key = "default" if force_default else "per_class"
    else:
        key = str(checklist).strip().lower()
    if key in ("default_v3", "v3"):
        return DEFAULT_HYP_CHECKLIST_V3
    if key == "default" or force_default:
        return DEFAULT_HYP_CHECKLIST
    sub = (subset or "").strip().lower()
    if sub in MVTEC_HYP_CHECKLISTS:
        return MVTEC_HYP_CHECKLISTS[sub]
    return DEFAULT_HYP_CHECKLIST


def format_subq(question: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q = q + "?"
    return f"{q}\n{SUBQ_SUFFIX}"
