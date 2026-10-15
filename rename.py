#!/usr/bin/env python3
from pathlib import Path
import shutil
import itertools

ROOT = Path("/home/atx/code/Triad-main/saved_patches2")
files = sorted(ROOT.glob("*.png"), key=lambda x: x.stat().st_mtime)

# 按前缀聚组
groups = {}
for f in files:
    prefix = f.name.split('_')[0]          # 12345
    groups.setdefault(prefix, []).append(f)

# 按前缀首次出现顺序给组号
ordered = sorted(groups.items(), key=lambda kv: min(f.stat().st_mtime for f in kv[1]))

for group_id, (_, group) in enumerate(ordered):
    for f in group:
        patch = f.name.split('_', 1)[1]    # patch00.png ...
        new_path = ROOT / f"{group_id:02d}_{patch}"
        if not new_path.exists():
            shutil.move(str(f), str(new_path))

print(f"renamed {len(files)} files into {len(groups)} groups")