#!/usr/bin/env python3
"""
split_dataset.py - Chia dataset processed thành các container nhỏ hơn.

Mỗi container chứa tối đa N samples (mặc định 5000).
Chỉ bao gồm sample có đầy đủ file trong 5 thư mục bắt buộc + metadata.
hair_patches là tùy chọn: sample hợp lệ nếu num_patches == 0 (không có patch).
Sample thiếu file ở bất kỳ folder bắt buộc nào sẽ bị loại và ghi log.

Chạy trên WSL/Linux:
    python3 split_dataset.py \
        --source '/mnt/c/Users/Admin/My Drive/TryHairStyle/processed' \
        --output-dir '/mnt/c/Users/Admin/My Drive/TryHairStyle' \
        --batch-size 5000

Thêm --dry-run để kiểm tra trước khi chạy thật.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("Cài tqdm để có thanh tiến trình: pip install tqdm")
    # Fallback nếu không có tqdm
    def tqdm(iterable, **kwargs):
        return iterable

# Tất cả thư mục con (bao gồm cả hair_patches)
ALL_SUBDIRS = [
    "bald_images",
    "ground_truth_images",
    "hair_only_images",
    "hair_patches",       # Đặc biệt: nhiều file / sample ({ID}_patch_XXX.png)
    "identity_embeddings",
    "style_vectors",
]

# 5 thư mục BẮT BUỘC - mỗi sample PHẢI có file trong tất cả thư mục này
REQUIRED_SUBDIRS = [
    "bald_images",
    "ground_truth_images",
    "hair_only_images",
    "identity_embeddings",
    "style_vectors",
]

# Các thư mục có đúng 1 file / sample (filename = {ID}.ext)
SINGLE_FILE_DIRS = {
    "bald_images":        ".png",
    "ground_truth_images": ".png",
    "hair_only_images":   ".png",
    "identity_embeddings": ".npy",
    "style_vectors":      ".png",
}

# hair_patches có nhiều file / sample: {ID}_patch_000.png, {ID}_patch_001.png, ...
PATCH_DIR = "hair_patches"


def setupLogging(logFile: str) -> logging.Logger:
    """Thiết lập logging ra cả console và file."""
    logger = logging.getLogger("split_dataset")
    logger.setLevel(logging.DEBUG)

    # Console handler - INFO trở lên
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File handler - DEBUG trở lên (ghi chi tiết)
    fh = logging.FileHandler(logFile, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


def scanSampleIds(sourceDir: Path, logger: logging.Logger) -> dict:
    """
    Quét tất cả 6 thư mục để tìm sample IDs có mặt trong mỗi thư mục.
    Trả về dict: {sample_id: {subdir: [list_of_files]}}
    """
    logger.info("Đang quét sample IDs từ tất cả thư mục con...")

    # Dict chứa: sample_id -> {subdir_name -> [filenames]}
    sampleFiles = defaultdict(lambda: defaultdict(list))

    # Quét các thư mục single-file (1 file / sample)
    for dirName, ext in SINGLE_FILE_DIRS.items():
        dirPath = sourceDir / dirName
        if not dirPath.exists():
            logger.error(f"Thư mục không tồn tại: {dirPath}")
            continue

        count = 0
        for f in tqdm(list(dirPath.iterdir()), desc=f"  Scan {dirName}", unit="file"):
            if f.is_file() and f.suffix.lower() == ext:
                sampleId = f.stem  # filename không có extension = sample ID
                sampleFiles[sampleId][dirName].append(f.name)
                count += 1

        logger.info(f"  {dirName}: {count} files")

    # Quét hair_patches (nhiều file / sample)
    patchDir = sourceDir / PATCH_DIR
    if patchDir.exists():
        count = 0
        for f in tqdm(list(patchDir.iterdir()), desc=f"  Scan {PATCH_DIR}", unit="file"):
            if f.is_file() and f.suffix.lower() == ".png":
                # Tên file: {ID}_patch_XXX.png → cần tách lấy ID
                name = f.stem  # e.g. "AP054876-002_patch_000"
                # Tìm vị trí "_patch_" để tách sample ID
                patchIdx = name.find("_patch_")
                if patchIdx != -1:
                    sampleId = name[:patchIdx]
                    sampleFiles[sampleId][PATCH_DIR].append(f.name)
                    count += 1
                else:
                    logger.warning(f"  File không đúng format patch: {f.name}")
        logger.info(f"  {PATCH_DIR}: {count} files")
    else:
        logger.error(f"Thư mục không tồn tại: {patchDir}")

    logger.info(f"Tổng unique sample IDs phát hiện: {len(sampleFiles)}")
    return sampleFiles


def loadMetadata(metadataPath: Path, logger: logging.Logger) -> dict:
    """Đọc metadata.jsonl, trả về dict {sample_id: json_line_string}."""
    logger.info(f"Đang đọc metadata từ: {metadataPath}")

    metadataMap = {}
    lineCount = 0
    errorCount = 0

    with open(metadataPath, "r", encoding="utf-8") as f:
        for lineNum, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sampleId = data.get("id")
                if sampleId:
                    metadataMap[sampleId] = line
                    lineCount += 1
                else:
                    logger.warning(f"  Metadata dòng {lineNum}: thiếu field 'id'")
                    errorCount += 1
            except json.JSONDecodeError as e:
                logger.warning(f"  Metadata dòng {lineNum}: JSON parse error - {e}")
                errorCount += 1

    logger.info(f"Metadata: {lineCount} entries hợp lệ, {errorCount} lỗi")
    return metadataMap


def filterCompleteSamples(
    sampleFiles: dict,
    metadataMap: dict,
    logger: logging.Logger,
) -> list:
    """
    Lọc chỉ giữ sample có ĐẦY ĐỦ file trong 5 thư mục bắt buộc + metadata.
    hair_patches: hợp lệ nếu có patch files HOẶC metadata ghi num_patches == 0.
    Trả về danh sách sample IDs đã sort.
    """
    logger.info("Đang lọc sample có đầy đủ dữ liệu (5 thư mục bắt buộc + metadata)...")
    logger.info("  (hair_patches tùy chọn: hợp lệ nếu num_patches == 0)")

    completeSamples = []
    incompleteCount = 0
    missingDetails = defaultdict(int)
    patchStats = {"has_patches": 0, "no_patches": 0}

    # Hợp nhất tất cả sample IDs từ cả file scan và metadata
    allIds = set(sampleFiles.keys()) | set(metadataMap.keys())

    for sampleId in sorted(allIds):
        missingDirs = []

        # Kiểm tra metadata (bắt buộc)
        if sampleId not in metadataMap:
            missingDirs.append("metadata.jsonl")

        # Kiểm tra 5 thư mục bắt buộc
        for subdir in REQUIRED_SUBDIRS:
            if subdir not in sampleFiles.get(sampleId, {}):
                missingDirs.append(subdir)

        # Kiểm tra hair_patches đặc biệt:
        # - Hợp lệ nếu có patch files trong thư mục
        # - HOẶC metadata ghi num_patches == 0 (sample không có patch)
        hasPatchFiles = PATCH_DIR in sampleFiles.get(sampleId, {})
        if sampleId in metadataMap:
            try:
                meta = json.loads(metadataMap[sampleId])
                numPatches = meta.get("num_patches", -1)
            except json.JSONDecodeError:
                numPatches = -1

            if hasPatchFiles:
                patchStats["has_patches"] += 1
            elif numPatches == 0:
                # Sample hợp lệ - metadata ghi rõ không có patch
                patchStats["no_patches"] += 1
            else:
                # Có metadata nhưng num_patches > 0 mà không có file patch
                missingDirs.append(f"{PATCH_DIR}(expected {numPatches} patches)")
        elif not hasPatchFiles:
            missingDirs.append(PATCH_DIR)

        if missingDirs:
            incompleteCount += 1
            for d in missingDirs:
                missingDetails[d] += 1
            logger.debug(f"  SKIP {sampleId}: thiếu [{', '.join(missingDirs)}]")
        else:
            completeSamples.append(sampleId)

    logger.info(f"Kết quả lọc:")
    logger.info(f"  ✅ Đầy đủ (sẽ xử lý): {len(completeSamples)} samples")
    logger.info(f"     - Có hair_patches:  {patchStats['has_patches']}")
    logger.info(f"     - Không patch (OK): {patchStats['no_patches']}")
    logger.info(f"  ❌ Thiếu (sẽ bỏ qua):  {incompleteCount} samples")

    if missingDetails:
        logger.info("  Chi tiết thiếu theo thư mục:")
        for dirName, count in sorted(missingDetails.items(), key=lambda x: -x[1]):
            logger.info(f"    - {dirName}: {count} samples thiếu")

    return completeSamples


def splitIntoContainers(
    completeSamples: list,
    batchSize: int,
    logger: logging.Logger,
) -> list:
    """Chia danh sách sample IDs thành các batch."""
    containers = []
    for i in range(0, len(completeSamples), batchSize):
        batch = completeSamples[i : i + batchSize]
        containers.append(batch)

    numContainers = len(containers)
    logger.info(f"Chia thành {numContainers} containers (batch size = {batchSize})")
    for idx, batch in enumerate(containers):
        containerName = f"processed_{idx + 1:03d}"
        logger.info(f"  {containerName}: {len(batch)} samples")

    return containers


def copyFilesForContainer(
    containerIdx: int,
    sampleIds: list,
    sampleFiles: dict,
    metadataMap: dict,
    sourceDir: Path,
    outputDir: Path,
    dryRun: bool,
    logger: logging.Logger,
) -> dict:
    """
    Copy files của 1 container.
    Trả về thống kê: {subdir: file_count}
    """
    containerName = f"processed_{containerIdx + 1:03d}"
    containerDir = outputDir / containerName
    stats = defaultdict(int)

    # Tạo thư mục container và các thư mục con
    if not dryRun:
        containerDir.mkdir(parents=True, exist_ok=True)
        for subdir in ALL_SUBDIRS:
            (containerDir / subdir).mkdir(exist_ok=True)

    logger.info(f"[{containerName}] Xử lý {len(sampleIds)} samples...")

    # Copy files cho mỗi sample
    for sampleId in tqdm(sampleIds, desc=f"  {containerName}", unit="sample"):
        fileMap = sampleFiles[sampleId]

        # Copy single-file dirs
        for dirName, ext in SINGLE_FILE_DIRS.items():
            srcFile = sourceDir / dirName / f"{sampleId}{ext}"
            dstFile = containerDir / dirName / f"{sampleId}{ext}"

            if not dryRun:
                shutil.copy2(str(srcFile), str(dstFile))
            stats[dirName] += 1

        # Copy hair_patches (tất cả patch files của sample này)
        patchFiles = fileMap.get(PATCH_DIR, [])
        for patchName in patchFiles:
            srcFile = sourceDir / PATCH_DIR / patchName
            dstFile = containerDir / PATCH_DIR / patchName

            if not dryRun:
                shutil.copy2(str(srcFile), str(dstFile))
            stats[PATCH_DIR] += 1

    # Ghi metadata.jsonl cho container này
    metadataPath = containerDir / "metadata.jsonl"
    if not dryRun:
        with open(metadataPath, "w", encoding="utf-8") as f:
            for sampleId in sampleIds:
                f.write(metadataMap[sampleId] + "\n")
    stats["metadata_lines"] = len(sampleIds)

    # Log thống kê
    totalFiles = sum(v for k, v in stats.items() if k != "metadata_lines")
    logger.info(f"  [{containerName}] {totalFiles} files copied, {stats['metadata_lines']} metadata lines")

    return dict(stats)


def verifyIntegrity(
    containers: list,
    allStats: list,
    completeSamples: list,
    logger: logging.Logger,
):
    """Kiểm tra tính toàn vẹn sau khi chia."""
    logger.info("=" * 60)
    logger.info("KIỂM TRA TÍNH TOÀN VẸN")
    logger.info("=" * 60)

    # Tổng samples
    totalSamples = sum(len(c) for c in containers)
    assert totalSamples == len(completeSamples), \
        f"Mismatch: tổng samples trong containers ({totalSamples}) != complete samples ({len(completeSamples)})"
    logger.info(f"✅ Tổng samples: {totalSamples} == {len(completeSamples)} (khớp)")

    # Kiểm tra không trùng lặp
    allIds = []
    for batch in containers:
        allIds.extend(batch)
    uniqueIds = set(allIds)
    assert len(uniqueIds) == len(allIds), \
        f"Phát hiện trùng lặp: {len(allIds)} total != {len(uniqueIds)} unique"
    logger.info(f"✅ Không trùng lặp: {len(uniqueIds)} unique IDs")

    # Bảng thống kê
    logger.info("")
    logger.info(f"{'Container':<16} {'Samples':>8} {'bald':>8} {'gt':>8} {'hair':>8} {'patch':>8} {'embed':>8} {'style':>8}")
    logger.info("-" * 84)

    totalsByDir = defaultdict(int)
    for idx, (batch, stats) in enumerate(zip(containers, allStats)):
        name = f"processed_{idx + 1:03d}"
        logger.info(
            f"{name:<16} {len(batch):>8} "
            f"{stats.get('bald_images', 0):>8} "
            f"{stats.get('ground_truth_images', 0):>8} "
            f"{stats.get('hair_only_images', 0):>8} "
            f"{stats.get('hair_patches', 0):>8} "
            f"{stats.get('identity_embeddings', 0):>8} "
            f"{stats.get('style_vectors', 0):>8}"
        )
        for k, v in stats.items():
            totalsByDir[k] += v

    logger.info("-" * 84)
    logger.info(
        f"{'TOTAL':<16} {totalSamples:>8} "
        f"{totalsByDir.get('bald_images', 0):>8} "
        f"{totalsByDir.get('ground_truth_images', 0):>8} "
        f"{totalsByDir.get('hair_only_images', 0):>8} "
        f"{totalsByDir.get('hair_patches', 0):>8} "
        f"{totalsByDir.get('identity_embeddings', 0):>8} "
        f"{totalsByDir.get('style_vectors', 0):>8}"
    )
    logger.info("")
    logger.info("✅ HOÀN TẤT - Dataset đã chia thành công!")


def main():
    parser = argparse.ArgumentParser(
        description="Chia dataset processed thành các container nhỏ (mỗi container tối đa N samples)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Đường dẫn tới thư mục processed gốc",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Thư mục chứa các containers output (processed_001, processed_002, ...)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Số sample tối đa mỗi container (mặc định: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra, không copy file thật",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="split_dataset.log",
        help="Đường dẫn file log (mặc định: split_dataset.log)",
    )

    args = parser.parse_args()

    sourceDir = Path(args.source)
    outputDir = Path(args.output_dir)

    # Thiết lập logging
    logger = setupLogging(args.log_file)

    logger.info("=" * 60)
    logger.info("SPLIT DATASET - BẮT ĐẦU")
    logger.info("=" * 60)
    logger.info(f"Source:     {sourceDir}")
    logger.info(f"Output:    {outputDir}")
    logger.info(f"Batch:     {args.batch_size}")
    logger.info(f"Dry-run:   {args.dry_run}")
    logger.info("")

    # Validate source
    if not sourceDir.exists():
        logger.error(f"Thư mục source không tồn tại: {sourceDir}")
        sys.exit(1)

    metadataPath = sourceDir / "metadata.jsonl"
    if not metadataPath.exists():
        logger.error(f"Không tìm thấy metadata: {metadataPath}")
        sys.exit(1)

    # === BƯỚC 1: Quét files ===
    sampleFiles = scanSampleIds(sourceDir, logger)
    logger.info("")

    # === BƯỚC 2: Đọc metadata ===
    metadataMap = loadMetadata(metadataPath, logger)
    logger.info("")

    # === BƯỚC 3: Lọc sample đầy đủ 6 thư mục + metadata ===
    completeSamples = filterCompleteSamples(sampleFiles, metadataMap, logger)
    logger.info("")

    if not completeSamples:
        logger.error("Không có sample nào đầy đủ! Kiểm tra lại dataset.")
        sys.exit(1)

    # === BƯỚC 4: Chia batch ===
    containers = splitIntoContainers(completeSamples, args.batch_size, logger)
    logger.info("")

    # === BƯỚC 5: Copy files ===
    if args.dry_run:
        logger.info("🔍 DRY-RUN MODE - Không copy file thật")
        logger.info("")

    allStats = []
    for idx, batch in enumerate(containers):
        stats = copyFilesForContainer(
            containerIdx=idx,
            sampleIds=batch,
            sampleFiles=sampleFiles,
            metadataMap=metadataMap,
            sourceDir=sourceDir,
            outputDir=outputDir,
            dryRun=args.dry_run,
            logger=logger,
        )
        allStats.append(stats)

    logger.info("")

    # === BƯỚC 6: Verify ===
    verifyIntegrity(containers, allStats, completeSamples, logger)


if __name__ == "__main__":
    main()
