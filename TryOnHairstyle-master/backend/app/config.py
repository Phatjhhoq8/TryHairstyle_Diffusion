"""
HairFusion — Centralized Configuration.
All paths are relative to the project root (HairFusion-main/).
"""
import os

# ── Directories ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # HairFusion-main/
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")
LOG_DIR = os.path.join(BACKEND_DIR, "logs")
CONFIG_DIR = os.path.join(BACKEND_DIR, "configs")

# ── Model Weights ───────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(LOG_DIR, "hairfusion", "models",
                               "[Train]_[epoch=599]_[train_loss_epoch=0.3666].ckpt")
VAE_PATH = os.path.join(MODEL_DIR, "realisticVisionV60B1_v51VAE.safetensors")
SEG_MODEL_PATH = os.path.join(MODEL_DIR, "face_segment16.pth")
LANDMARKS_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

# ── Config Files ────────────────────────────────────────────
CONFIG_YAML_PATH = os.path.join(CONFIG_DIR, "config.yaml")

# ── Runtime ─────────────────────────────────────────────────
DEVICE = "cuda"
IMG_H = 512
IMG_W = 512
SAVE_MEMORY = False
