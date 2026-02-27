"""
Revert Colab.ipynb — bỏ FORCE_FRESH_STAGE2 vì user chưa train cũ.
Giữ lại cell cảnh báo về 4 fixes đã sửa.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(r"c:\Users\Admin\Desktop\TryHairStyle\backend\training\Colab.ipynb")

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# 1. Xóa FORCE_FRESH_STAGE2 khỏi cell cấu hình
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "FORCE_FRESH_STAGE2" in src and "RESUME_TRAINING" in src:
        new_source = [line for line in cell["source"] if "FORCE_FRESH" not in line]
        cell["source"] = new_source
        print(f"  ✅ Cell {i}: Xóa FORCE_FRESH_STAGE2")
        break

# 2. Đơn giản hóa Stage 2 cell — dùng RESUME_TRAINING trực tiếp
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "Stage2Trainer" in src and "train_loop" in src:
        cell["source"] = [
            "import os, torch\n",
            "os.chdir(str(WORK_DIR))\n",
            "\n",
            "print(\"=\"*60)\n",
            "print(\"🧬 STAGE 2: HAIR INPAINTING\")\n",
            "print(\"=\"*60)\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"💾 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB used\")\n",
            "\n",
            "from backend.training.train_stage2 import Stage2Trainer\n",
            "t2 = Stage2Trainer()\n",
            "t2.train_loop(\n",
            "    num_epochs=STAGE2_EPOCHS, batch_size=STAGE2_BATCH_SIZE,\n",
            "    max_samples=STAGE2_MAX_SAMPLES,\n",
            "    target_size=(STAGE2_RESOLUTION, STAGE2_RESOLUTION),\n",
            "    accumulation_steps=STAGE2_ACCUMULATION, resume=RESUME_TRAINING\n",
            ")\n",
            "del t2; torch.cuda.empty_cache()\n",
            "print(\"✅ Stage 2 xong!\")\n",
        ]
        print(f"  ✅ Cell {i}: Đơn giản hóa Stage 2 (dùng RESUME_TRAINING)")
        break

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print(f"\n🎉 Done — đã bỏ FORCE_FRESH_STAGE2, Colab dùng RESUME_TRAINING bình thường.")
