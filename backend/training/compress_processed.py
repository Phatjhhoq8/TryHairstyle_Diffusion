"""
Script nén thư mục processed thành các file ZIP để upload lên Google Drive.

Tính năng:
- Nén từng thư mục con riêng biệt thành file .zip
- Hỗ trợ chia nhỏ (split) file zip nếu vượt quá giới hạn (mặc định 4GB)
- Hiển thị progress bar khi nén
- Có thể chọn nén toàn bộ hoặc chỉ thư mục cụ thể
- Tự động bỏ qua những thư mục đã nén rồi (kiểm tra file .zip tồn tại)

Sử dụng:
    python compress_processed.py                          # Nén tất cả
    python compress_processed.py --folders bald_images    # Nén 1 thư mục cụ thể
    python compress_processed.py --split-size 2           # Chia nhỏ mỗi phần 2GB
    python compress_processed.py --force                  # Nén lại cả thư mục đã nén
    python compress_processed.py --output ./compressed    # Chỉ định thư mục output
"""

import os
import sys
import zipfile
import argparse
import math
import shutil
from pathlib import Path
from datetime import datetime

# Thư mục gốc chứa dữ liệu đã xử lý
PROCESSED_DIR = Path(__file__).parent / "processed"
# Thư mục mặc định để lưu file nén
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "compressed_for_drive"
# Kích thước mặc định mỗi phần khi chia nhỏ (GB)
DEFAULT_SPLIT_SIZE_GB = 4
# Google Drive giới hạn upload 5TB/ngày, nhưng file đơn lẻ tối đa 5TB
# Tuy nhiên, để upload nhanh và ổn định, nên giữ mỗi file dưới 4GB


def getSizeReadable(sizeBytes):
    """Chuyển đổi bytes sang đơn vị dễ đọc."""
    if sizeBytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(sizeBytes, 1024)))
    p = math.pow(1024, i)
    s = round(sizeBytes / p, 2)
    return f"{s} {units[i]}"


def getDirSize(dirPath):
    """Tính tổng kích thước thư mục (bytes)."""
    totalSize = 0
    for dirpath, dirnames, filenames in os.walk(dirPath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                totalSize += os.path.getsize(fp)
    return totalSize


def countFiles(dirPath):
    """Đếm tổng số file trong thư mục."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(dirPath):
        count += len(filenames)
    return count


def compressFolder(folderPath, outputDir, splitSizeBytes=None, forceOverwrite=False):
    """
    Nén 1 thư mục thành file .zip, hỗ trợ chia nhỏ.

    Args:
        folderPath: Đường dẫn thư mục cần nén
        outputDir: Thư mục output
        splitSizeBytes: Kích thước tối đa mỗi phần (bytes). None = không chia nhỏ
        forceOverwrite: Nén lại nếu file đã tồn tại
    """
    folderName = folderPath.name
    zipPath = outputDir / f"{folderName}.zip"

    # Kiểm tra đã nén chưa (trừ khi force)
    if not forceOverwrite and zipPath.exists():
        print(f"  ⏭️  Bỏ qua '{folderName}' - đã có file nén. Dùng --force để nén lại.")
        return

    # Lấy thông tin thư mục
    dirSize = getDirSize(folderPath)
    fileCount = countFiles(folderPath)
    print(f"\n  📁 Thư mục: {folderName}")
    print(f"     Kích thước gốc: {getSizeReadable(dirSize)}")
    print(f"     Số file: {fileCount:,}")

    if fileCount == 0:
        print(f"  ⚠️  Thư mục rỗng, bỏ qua.")
        return

    # Thu thập tất cả file cần nén
    allFiles = []
    for dirpath, dirnames, filenames in os.walk(folderPath):
        for filename in filenames:
            fullPath = Path(dirpath) / filename
            # Tạo relative path so với thư mục processed
            # Giữ lại tên folder cha để khi giải nén có đúng cấu trúc
            relativePath = fullPath.relative_to(folderPath.parent)
            allFiles.append((fullPath, relativePath))

    # Sắp xếp theo tên để dễ theo dõi
    allFiles.sort(key=lambda x: str(x[1]))

    # Nén không chia nhỏ hoặc chia nhỏ tùy theo cài đặt
    if splitSizeBytes and dirSize > splitSizeBytes:
        # Chia nhỏ thành nhiều phần
        numParts = math.ceil(dirSize / splitSizeBytes)
        print(f"     🔀 Chia thành ~{numParts} phần (mỗi phần ≤ {getSizeReadable(splitSizeBytes)})")
        compressSplit(allFiles, folderName, outputDir, splitSizeBytes)
    else:
        # Nén thành 1 file duy nhất
        print(f"     📦 Đang nén thành 1 file...")
        compressSingle(allFiles, zipPath)

    print(f"  ✅ Hoàn tất nén '{folderName}'!")


def compressSingle(allFiles, zipPath):
    """Nén tất cả file vào 1 file zip duy nhất."""
    totalFiles = len(allFiles)
    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx, (fullPath, relativePath) in enumerate(allFiles, 1):
            zf.write(fullPath, relativePath)
            # Hiển thị progress mỗi 1000 file hoặc file cuối
            if idx % 1000 == 0 or idx == totalFiles:
                pct = (idx / totalFiles) * 100
                print(f"     📊 Progress: {idx:,}/{totalFiles:,} ({pct:.1f}%)", end='\r')

    # Hiển thị kích thước file nén
    compressedSize = zipPath.stat().st_size
    print(f"\n     💾 Kích thước nén: {getSizeReadable(compressedSize)} -> {zipPath.name}")


def compressSplit(allFiles, folderName, outputDir, splitSizeBytes):
    """Nén và chia nhỏ thành nhiều file zip."""
    partNum = 1
    currentSize = 0
    currentFiles = []
    totalFiles = len(allFiles)
    processedCount = 0

    def flushPart(files, partNumber):
        """Ghi 1 phần zip."""
        partName = f"{folderName}_part{partNumber:03d}.zip"
        partPath = outputDir / partName
        with zipfile.ZipFile(partPath, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for fullPath, relativePath in files:
                zf.write(fullPath, relativePath)
        partSize = partPath.stat().st_size
        print(f"     💾 {partName}: {getSizeReadable(partSize)}")

    for fullPath, relativePath in allFiles:
        fileSize = fullPath.stat().st_size
        processedCount += 1

        # Nếu thêm file này vào sẽ vượt quá giới hạn -> ghi phần hiện tại
        if currentFiles and (currentSize + fileSize > splitSizeBytes):
            flushPart(currentFiles, partNum)
            partNum += 1
            currentFiles = []
            currentSize = 0

        currentFiles.append((fullPath, relativePath))
        currentSize += fileSize

        # Hiển thị progress
        if processedCount % 2000 == 0 or processedCount == totalFiles:
            pct = (processedCount / totalFiles) * 100
            print(f"     📊 Progress: {processedCount:,}/{totalFiles:,} ({pct:.1f}%)", end='\r')

    # Ghi phần cuối cùng
    if currentFiles:
        flushPart(currentFiles, partNum)

    print(f"\n     📦 Tổng: {partNum} phần")


def compressMetadata(processedDir, outputDir, forceOverwrite=False):
    """Nén file metadata.jsonl riêng."""
    metadataFile = processedDir / "metadata.jsonl"
    if not metadataFile.exists():
        print("  ⚠️  Không tìm thấy metadata.jsonl")
        return

    zipPath = outputDir / "metadata.zip"
    if not forceOverwrite and zipPath.exists():
        print(f"  ⏭️  Bỏ qua 'metadata.jsonl' - đã có file nén.")
        return

    fileSize = metadataFile.stat().st_size
    print(f"\n  📄 metadata.jsonl: {getSizeReadable(fileSize)}")
    print(f"     📦 Đang nén...")

    with zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(metadataFile, "metadata.jsonl")

    compressedSize = zipPath.stat().st_size
    ratio = (1 - compressedSize / fileSize) * 100 if fileSize > 0 else 0
    print(f"     💾 {getSizeReadable(compressedSize)} (giảm {ratio:.1f}%) -> {zipPath.name}")
    print(f"  ✅ Hoàn tất nén metadata!")


def main():
    parser = argparse.ArgumentParser(
        description="Nén thư mục processed để upload lên Google Drive"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="Chỉ nén các thư mục con cụ thể (vd: bald_images hair_patches)",
        default=None
    )
    parser.add_argument(
        "--split-size",
        type=float,
        default=DEFAULT_SPLIT_SIZE_GB,
        help=f"Kích thước tối đa mỗi phần khi chia nhỏ, đơn vị GB (mặc định: {DEFAULT_SPLIT_SIZE_GB}GB)"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Không chia nhỏ file, nén thành 1 file duy nhất cho mỗi thư mục"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nén lại cả những thư mục đã nén trước đó"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Thư mục output (mặc định: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Bỏ qua metadata.jsonl"
    )

    args = parser.parse_args()

    # Kiểm tra thư mục processed
    if not PROCESSED_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {PROCESSED_DIR}")
        sys.exit(1)

    # Thiết lập output directory
    outputDir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    outputDir.mkdir(parents=True, exist_ok=True)

    # Tính kích thước split
    splitSizeBytes = None if args.no_split else int(args.split_size * 1024 * 1024 * 1024)

    # Lấy danh sách thư mục con cần nén
    if args.folders:
        subDirs = [PROCESSED_DIR / f for f in args.folders]
        # Kiểm tra tồn tại
        for d in subDirs:
            if not d.exists():
                print(f"❌ Không tìm thấy: {d}")
                sys.exit(1)
    else:
        subDirs = sorted([d for d in PROCESSED_DIR.iterdir() if d.is_dir()])

    # Header
    print("=" * 60)
    print("🗜️  COMPRESS PROCESSED DATA FOR GOOGLE DRIVE")
    print("=" * 60)
    print(f"📂 Source: {PROCESSED_DIR}")
    print(f"📤 Output: {outputDir}")
    if splitSizeBytes:
        print(f"✂️  Split size: {getSizeReadable(splitSizeBytes)}")
    else:
        print(f"✂️  Split: Tắt (nén thành 1 file mỗi thư mục)")
    print(f"🔄 Force overwrite: {'Có' if args.force else 'Không'}")
    print(f"📁 Thư mục cần nén: {len(subDirs)}")
    print("-" * 60)

    startTime = datetime.now()

    # Nén metadata.jsonl
    if not args.skip_metadata:
        compressMetadata(PROCESSED_DIR, outputDir, args.force)

    # Nén từng thư mục con
    for subDir in subDirs:
        compressFolder(subDir, outputDir, splitSizeBytes, args.force)

    # Tổng kết
    elapsed = datetime.now() - startTime
    print("\n" + "=" * 60)
    print("🎉 HOÀN TẤT!")
    print(f"⏱️  Thời gian: {elapsed}")
    print(f"📤 Output: {outputDir}")
    print("-" * 60)

    # Liệt kê file đã tạo
    outputFiles = sorted(outputDir.glob("*.zip"))
    totalCompressed = 0
    for f in outputFiles:
        size = f.stat().st_size
        totalCompressed += size
        print(f"   📦 {f.name}: {getSizeReadable(size)}")

    print(f"\n   💾 Tổng kích thước nén: {getSizeReadable(totalCompressed)}")
    print("=" * 60)
    print("\n💡 Để upload lên Google Drive, bạn có thể:")
    print("   1. Upload thủ công qua drive.google.com")
    print("   2. Dùng rclone: rclone copy ./compressed_for_drive remote:TryHairStyle/processed")
    print("   3. Dùng gdrive CLI: gdrive upload ./compressed_for_drive/*.zip")


if __name__ == "__main__":
    main()
