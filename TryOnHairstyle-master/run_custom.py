"""
HairFusion — Custom Inference CLI.
Run hair transfer on custom face + hair images.
"""
import argparse
import os
import shutil
import sys

# Initialize backend (sys.path setup)
import backend  # noqa: F401

from backend.app.config import DATA_DIR
from backend.app.services.preprocessing import PreprocessingService
from backend.app.services.diffusion import HairDiffusionService
from backend.app.utils.image_utils import tensor2img
import cv2


def run_custom_inference(face_path, hair_path, output_dir="results/custom"):
    print(f"--- Running HairFusion on Custom Input ---")
    print(f"Face: {face_path}")
    print(f"Hair: {hair_path}")

    if not os.path.exists(face_path):
        print(f"Error: Face image not found: {face_path}")
        return
    if not os.path.exists(hair_path):
        print(f"Error: Hair image not found: {hair_path}")
        return

    # 1. Setup Session
    session_id = "custom_run"
    base_dir = os.path.join(DATA_DIR, session_id)

    if os.path.exists(base_dir):
        try:
            shutil.rmtree(base_dir)
        except Exception as e:
            print(f"Warning: Could not clean temp dir {base_dir}: {e}")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)

    # Copy images
    img_face = cv2.imread(face_path)
    img_hair = cv2.imread(hair_path)

    cv2.imwrite(os.path.join(base_dir, "images", "face.png"), img_face)
    cv2.imwrite(os.path.join(base_dir, "images", "hair.png"), img_hair)

    # Create test pairs
    with open(os.path.join(base_dir, "test_pairs.txt"), "w") as f:
        f.write("face.png hair.png\n")

    # 2. Preprocess
    print("--- Preprocessing ---")
    preprocessor = PreprocessingService()
    try:
        preprocessor.run_preprocess(os.path.join(base_dir, "images", "face.png"), session_id)
        preprocessor.run_preprocess(os.path.join(base_dir, "images", "hair.png"), session_id)
        preprocessor.run_make_agnostic("face.png", session_id)
    except Exception as e:
        print(f"Preprocessing Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Inference
    print("--- Running Inference ---")
    try:
        diffusion = HairDiffusionService()
        results = diffusion.run_inference(data_root_dir=base_dir)

        if results:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "result.png")
            cv2.imwrite(save_path, results[0][:, :, ::-1])
            print(f"Success! Image saved to: {save_path}")
        else:
            print("No results generated.")
    except Exception as e:
        print(f"Inference Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HairFusion on custom images")
    parser.add_argument("--face", type=str, required=True, help="Path to input face image")
    parser.add_argument("--hair", type=str, required=True, help="Path to reference hair image")
    parser.add_argument("--output", type=str, default="results/custom", help="Output directory")

    args = parser.parse_args()
    run_custom_inference(args.face, args.hair, args.output)
