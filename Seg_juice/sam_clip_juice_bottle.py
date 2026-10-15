"""
SAM + CLIP open-vocabulary segmentation for MVTec LOCO juice_bottle.

Approximates LogSAD's patch-CLIP + SAM majority-vote merge for juice_bottle.
NOT a full LogSAD replica: no DINOv2, no k-means path, no anomaly scoring,
and CLIP weight is OpenAI ViT-L/14@336 (LogSAD uses DataComp.XL).

Important: LogSAD's open_clip_local hardcodes last-layer n_patches=(32, 32),
so CLIP input must be 448x448 (448/14=32), matching LogSAD.

Expects LogSAD checkout at Seg_juice/LogSAD-master/ (open_clip_local, prompt_ensemble).

Example:
  cd /home/yz/code/Triad-main/Seg_juice
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u sam_clip_juice_bottle.py \
    --split train --max_images 3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGSAD_ROOT = _SCRIPT_DIR / "LogSAD-master"
_DEFAULT_DATASET = "/home/yz/xxy/data/datasets/MMAD/MVTec-LOCO"
_DEFAULT_SAM_CKPT = str(_SCRIPT_DIR / "checkpoints" / "sam_vit_h_4b8939.pth")
_DEFAULT_CLIP_CACHE = os.path.expanduser("~/.cache/clip")
# RoBiS vendors SAM under third_party_library (imports use that package path)
_ROBIS_ROOT = Path("/home/yz/code/RoBiS")

print("boot: starting imports...", flush=True)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

print("boot: torch/cv2 ok", flush=True)

# LogSAD local CLIP (modified encode_image with intermediate layers)
if not (_LOGSAD_ROOT / "open_clip_local").is_dir():
    raise FileNotFoundError(
        f"Missing {_LOGSAD_ROOT / 'open_clip_local'}. "
        "Put the LogSAD repo under Seg_juice/LogSAD-master/"
    )
sys.path.insert(0, str(_LOGSAD_ROOT))
import open_clip_local as open_clip
from prompt_ensemble import encode_obj_text

print("boot: open_clip_local ok", flush=True)

# Prefer RoBiS-vendored SAM; fall back to pip-installed segment_anything
if _ROBIS_ROOT.is_dir():
    sys.path.insert(0, str(_ROBIS_ROOT))
    from third_party_library.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
else:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

print("boot: segment_anything ok", flush=True)

# LogSAD juice_bottle patch queries (semantic classes for merge)
DEFAULT_PATCH_QUERIES = [
    ["glass"],
    ["liquid in bottle"],
    ["fruit"],
    ["label", "tag"],
    ["black background", "background"],
]
CLASS_NAMES = ["glass", "liquid", "fruit", "label", "background"]

# Distinct colors for visualization (BGR for OpenCV)
COLOR_MAP = np.array(
    [
        [180, 120, 40],   # glass
        [40, 40, 220],    # liquid
        [40, 180, 40],    # fruit
        [40, 200, 220],   # label
        [40, 40, 40],     # background
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM+CLIP segmentation for juice_bottle")
    p.add_argument(
        "--dataset_path",
        type=str,
        default=_DEFAULT_DATASET,
        help="MVTec LOCO root (contains juice_bottle/)",
    )
    p.add_argument("--category", type=str, default="juice_bottle")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "validation", "all"],
        help="Which LOCO split(s) to run",
    )
    p.add_argument(
        "--clip_cache_dir",
        type=str,
        default=_DEFAULT_CLIP_CACHE,
        help="Directory containing ViT-L-14-336px.pt",
    )
    p.add_argument(
        "--sam_checkpoint",
        type=str,
        default=_DEFAULT_SAM_CKPT,
        help="SAM ViT-H checkpoint path",
    )
    p.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Must be 448: LogSAD CLIP last-layer NACLIP uses n_patches=(32,32)=448/14",
    )
    p.add_argument("--feat_size", type=int, default=64, help="Upsampled patch grid (LogSAD uses 64)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_images", type=int, default=-1, help="-1 = all")
    p.add_argument(
        "--save_dir",
        type=str,
        default=str(_SCRIPT_DIR / "outputs_juice_seg"),
    )
    p.add_argument("--min_region", type=int, default=32, help="Filter small regions after merge")
    return p.parse_args()


def collect_images(dataset_path: str, category: str, split: str) -> list[str]:
    root = Path(dataset_path) / category
    paths: list[str] = []

    def add_split(name: str) -> None:
        split_dir = root / name
        if not split_dir.is_dir():
            return
        # train/good or test/{good,logical_anomalies,structural_anomalies}
        for sub in sorted(split_dir.iterdir()):
            if sub.is_dir():
                paths.extend(sorted(str(p) for p in sub.glob("*.png")))
            elif sub.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                paths.append(str(sub))

    if split == "all":
        for s in ("train", "validation", "test"):
            add_split(s)
    else:
        add_split(split)

    return paths


def to_np_img(m: torch.Tensor) -> np.ndarray:
    """Denormalize CLIP tensor [3,H,W] -> uint8 RGB."""
    m = m.permute(1, 2, 0).detach().cpu().numpy()
    mean = np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    std = np.array([[[0.26862954, 0.26130258, 0.27577711]]])
    m = m * std + mean
    return np.clip(m * 255.0, 0, 255).astype(np.uint8)


def merge_segmentations(sam_label: np.ndarray, semantic: np.ndarray, background_class: int) -> np.ndarray:
    """Majority-vote: assign each SAM instance the dominant CLIP semantic label (LogSAD)."""
    unique_labels = np.unique(sam_label)
    max_label = int(unique_labels.max()) if unique_labels.size else 0
    label_map = np.full(max_label + 1, background_class, dtype=int)
    max_sem = int(semantic.max()) if semantic.size else background_class

    for label_a in unique_labels:
        mask_a = sam_label == label_a
        labels_b = semantic[mask_a]
        if labels_b.size > 0:
            count_b = np.bincount(labels_b.astype(np.int64), minlength=max_sem + 1)
            label_map[label_a] = int(np.argmax(count_b))
    return label_map[sam_label]


def sam_instances_to_label_map(masks: list[dict]) -> np.ndarray:
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    h, w = sorted_masks[0]["segmentation"].shape
    out = np.zeros((h, w), dtype=np.int32)
    for i, ann in enumerate(sorted_masks, start=1):
        out[ann["segmentation"]] = i
    return out


def colorize(label_map: np.ndarray) -> np.ndarray:
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(COLOR_MAP):
        rgb[label_map == i] = color
    return rgb


def filter_small_regions(label_map: np.ndarray, background: int, min_area: int) -> np.ndarray:
    binary = (label_map != background).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = label_map.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= min_area:
            out[labels == i] = background
    return out


def resize_visual_pos_embed(model, image_size: int) -> int:
    """Interpolate OpenAI ViT-L/14@336 pos-embed to LogSAD's 448 input (32x32 patches)."""
    visual = model.visual
    patch = visual.patch_size[0] if isinstance(visual.patch_size, tuple) else visual.patch_size
    new_grid = image_size // patch
    old_grid = int(math.sqrt(visual.positional_embedding.shape[0] - 1))
    if old_grid == new_grid:
        visual.image_size = (image_size, image_size)
        visual.grid_size = (new_grid, new_grid)
        return new_grid

    print(f"Resizing CLIP pos-embed {old_grid}x{old_grid} -> {new_grid}x{new_grid}", flush=True)
    pos = visual.positional_embedding.data
    pos_tok, pos_img = pos[:1], pos[1:]
    pos_img = pos_img.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    pos_img = F.interpolate(pos_img.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    pos_img = pos_img.permute(0, 2, 3, 1).reshape(new_grid * new_grid, -1)
    new_pos = torch.cat([pos_tok, pos_img.to(pos.dtype)], dim=0)
    visual.positional_embedding = torch.nn.Parameter(new_pos)
    visual.image_size = (image_size, image_size)
    visual.grid_size = (new_grid, new_grid)
    return new_grid


def build_models(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects CUDA (LogSAD CLIP attention uses cuda tensors).")
    if args.image_size != 448:
        raise ValueError(
            f"image_size={args.image_size} is unsafe: LogSAD open_clip_local hardcodes "
            "n_patches=(32, 32) on the last ViT layer. Use 448."
        )

    clip_weight = Path(args.clip_cache_dir) / "ViT-L-14-336px.pt"
    if not clip_weight.is_file():
        raise FileNotFoundError(
            f"CLIP weight not found: {clip_weight}\n"
            "Expected OpenAI ViT-L-14-336px.pt in --clip_cache_dir."
        )
    if not Path(args.sam_checkpoint).is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    print(f"Loading CLIP from {clip_weight} ...", flush=True)
    model_clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336",
        pretrained="openai",
        cache_dir=args.clip_cache_dir,
        device=device,
    )
    resize_visual_pos_embed(model_clip, args.image_size)
    model_clip.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14-336")

    print(f"Loading SAM from {args.sam_checkpoint} ...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    transform = T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    return model_clip, tokenizer, mask_generator, transform, device


@torch.no_grad()
def segment_one(
    image_path: str,
    model_clip,
    patch_query_obj: torch.Tensor,
    mask_generator,
    transform,
    device: torch.device,
    image_size: int,
    feat_size: int,
    ori_feat_size: int,
    feature_list: list[int],
    min_region: int,
) -> dict:
    pil = Image.open(image_path).convert("RGB")
    pil_resized = pil.resize((image_size, image_size), Image.BICUBIC)
    tensor = torch.from_numpy(np.array(pil_resized)).permute(2, 0, 1).float() / 255.0
    batch = transform(tensor).unsqueeze(0).to(device)

    # CLIP patch features (LogSAD-style)
    _image_features, patch_tokens, proj_patch_tokens = model_clip.encode_image(batch, feature_list)
    # proj_patch_tokens: (1, N, embed_dim) without CLS after internal pool — LogSAD uses tokens @ proj
    # encode returns pooled, patch_tokens(list), proj = tokens @ proj  where tokens include spatial after pool
    # From VisionTransformer: return pooled, patch_tokens, tokens @ self.proj
    # tokens here are WITHOUT cls (from _global_pool), shape [B, grid^2, width] then @ proj -> [B, grid^2, embed]

    if proj_patch_tokens.dim() == 3:
        proj = proj_patch_tokens[0]  # (N, C)
    else:
        proj = proj_patch_tokens

    if proj.shape[0] != ori_feat_size * ori_feat_size:
        # safety: infer grid
        n = proj.shape[0]
        g = int(n**0.5)
        proj = proj[: g * g]
        ori_feat_size = g

    proj = proj.view(1, ori_feat_size, ori_feat_size, -1).permute(0, 3, 1, 2)
    if feat_size != ori_feat_size:
        proj = F.interpolate(proj, size=(feat_size, feat_size), mode="bilinear", align_corners=True)
    proj = proj.permute(0, 2, 3, 1).reshape(feat_size * feat_size, -1)
    proj = F.normalize(proj, p=2, dim=-1)

    patch_similarity = proj @ patch_query_obj.T
    patch_mask = patch_similarity.argmax(-1).view(feat_size, feat_size).cpu().numpy().astype(np.int32)

    raw_image = to_np_img(batch[0])  # RGB uint8 at CLIP resolution
    masks = mask_generator.generate(raw_image)
    if len(masks) == 0:
        sam_label = np.zeros(raw_image.shape[:2], dtype=np.int32)
        merge = np.full(raw_image.shape[:2], len(CLASS_NAMES) - 1, dtype=np.int32)
    else:
        sam_label = sam_instances_to_label_map(masks)
        h, w = raw_image.shape[:2]
        resized_patch = cv2.resize(patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bg = len(CLASS_NAMES) - 1
        merge = merge_segmentations(sam_label, resized_patch, background_class=bg)
        merge = filter_small_regions(merge, background=bg, min_area=min_region)

    return {
        "raw_rgb": raw_image,
        "patch_mask": patch_mask,
        "sam_label": sam_label,
        "merge": merge,
        "overlay": cv2.addWeighted(raw_image, 0.55, colorize(merge), 0.45, 0),
    }


def save_results(out_dir: Path, stem: str, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["raw_rgb"]).save(out_dir / f"{stem}_image.png")
    Image.fromarray(colorize(result["merge"])).save(out_dir / f"{stem}_merge.png")
    Image.fromarray(result["overlay"]).save(out_dir / f"{stem}_overlay.png")
    # upsampled patch-only preview
    pm = result["patch_mask"]
    pm_big = cv2.resize(pm, (result["raw_rgb"].shape[1], result["raw_rgb"].shape[0]), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(colorize(pm_big)).save(out_dir / f"{stem}_clip_patch.png")
    np.save(out_dir / f"{stem}_merge.npy", result["merge"])


def main() -> None:
    args = parse_args()
    model_clip, tokenizer, mask_generator, transform, device = build_models(args)

    patch_query_obj = encode_obj_text(model_clip, DEFAULT_PATCH_QUERIES, tokenizer, device)
    assert patch_query_obj.shape[0] == len(CLASS_NAMES)

    image_paths = collect_images(args.dataset_path, args.category, args.split)
    if not image_paths:
        raise FileNotFoundError(
            f"No images under {args.dataset_path}/{args.category} (split={args.split})"
        )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    ori_feat_size = args.image_size // 14  # 448/14=32, matches LogSAD
    feature_list = [6, 12, 18, 24]  # 1-indexed layers, same as LogSAD
    save_root = Path(args.save_dir) / args.category
    save_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Config: image_size={args.image_size}, ori_feat={ori_feat_size}, "
        f"feat_size={args.feat_size}, images={len(image_paths)}",
        flush=True,
    )

    # legend
    legend = np.zeros((40, 40 * len(CLASS_NAMES), 3), dtype=np.uint8)
    for i, name in enumerate(CLASS_NAMES):
        legend[:, i * 40 : (i + 1) * 40] = COLOR_MAP[i]
    Image.fromarray(legend).save(save_root / "legend.png")
    with open(save_root / "classes.txt", "w", encoding="utf-8") as f:
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{i}\t{name}\n")

    print(f"Segmenting {len(image_paths)} images -> {save_root}")
    cat_root = Path(args.dataset_path) / args.category
    for path in tqdm(image_paths):
        try:
            stem = str(Path(path).relative_to(cat_root).with_suffix("")).replace(os.sep, "__")
        except ValueError:
            stem = Path(path).stem
        result = segment_one(
            path,
            model_clip,
            patch_query_obj,
            mask_generator,
            transform,
            device,
            args.image_size,
            args.feat_size,
            ori_feat_size,
            feature_list,
            args.min_region,
        )
        save_results(save_root, stem, result)

    print("Done.")
    print("Classes:", ", ".join(f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)))
    print("Note: LogSAD does not split upper/lower label or bottle cap; edit DEFAULT_PATCH_QUERIES to customize.")


if __name__ == "__main__":
    main()
