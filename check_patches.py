#!/usr/bin/env python3
"""Kiểm tra nhanh số sample có/không có patches."""
import json
import sys

metadata_path = sys.argv[1]
has_patches = 0
no_patches = 0
total = 0

with open(metadata_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        data = json.loads(line)
        if data.get("num_patches", 0) > 0:
            has_patches += 1
        else:
            no_patches += 1

print(f"Total metadata lines: {total}")
print(f"Has patches (num_patches > 0): {has_patches}")
print(f"No patches (num_patches == 0): {no_patches}")
