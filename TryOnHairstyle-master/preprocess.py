"""
HairFusion — Standalone Preprocessor.
Preprocess a given image (face detection, crop, segmentation masks).
"""
import os
import argparse

# Initialize backend (sys.path setup)
import backend  # noqa: F401

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from kornia.morphology import dilation, erosion
from torchvision.utils import save_image
from skimage import io, img_as_float32

from ffhq_dataset.landmarks_detector import LandmarksDetector
from face_parsing.model import BiSeNet
from backend.app.utils.image_utils import (
    get_seg, get_seg_mask, get_crop_coords_crop, get_nth
)
from backend.app.config import SEG_MODEL_PATH, LANDMARKS_MODEL_PATH, DATA_DIR
from torchvision.ops import masks_to_boxes

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, required=True,
                        help='image path to preprocess')
    parser.add_argument('--save_dir_name', type=str, default='Custom',
                        help='dir name to save preprocessed data')
    parser.add_argument('--crop_scale', type=float, default=4,
                        help='scale to crop, the bigger, the more zoom out')

    return parser.parse_args()

def run():
    args = parse_args()

    data_dir = args.save_dir_name

    KEYPOINTS_DIR = os.path.join(DATA_DIR, data_dir, 'keypoints')
    IMAGES_DIR = os.path.join(DATA_DIR, data_dir, 'images')
    HAIR_MASK_DIR = os.path.join(DATA_DIR, data_dir, 'mask_hair')
    FACE_MASK_DIR = os.path.join(DATA_DIR, data_dir, 'mask_face')
    NTH_DIR = os.path.join(DATA_DIR, data_dir, 'nth')

    os.makedirs(KEYPOINTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(HAIR_MASK_DIR, exist_ok=True)
    os.makedirs(FACE_MASK_DIR, exist_ok=True)
    os.makedirs(NTH_DIR, exist_ok=True)

    # keypoint extractor
    if not os.path.isfile(LANDMARKS_MODEL_PATH):
        print(f'{LANDMARKS_MODEL_PATH} does not exist!')
        exit()
    landmarks_detector = LandmarksDetector(LANDMARKS_MODEL_PATH)

    seg_model = BiSeNet(n_classes=16)
    seg_model.cuda()
    if not os.path.isfile(SEG_MODEL_PATH):
        print(f'{SEG_MODEL_PATH} does not exist!')
        exit()
    seg_model.load_state_dict(torch.load(SEG_MODEL_PATH))
    seg_model.eval()

    raw_img_path = args.img_path

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_resize = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(os.path.basename(raw_img_path))[0], i)
        crop_face_path = os.path.join(IMAGES_DIR, face_img_name)
        raw_img = img_as_float32(io.imread(raw_img_path))
        raw_img_cropped = get_crop_coords_crop(np.array(face_landmarks), (raw_img.shape[1], raw_img.shape[0]), raw_img, scale=args.crop_scale)

        raw_img_cropped = (raw_img_cropped * 255).astype(np.uint8)
        raw_img_cropped_pil = Image.fromarray(raw_img_cropped)
        raw_img_cropped_pil.save(crop_face_path)

        face_landmarks_cropped = landmarks_detector.get_landmarks(crop_face_path)[0]

        keypoints_path = os.path.join(KEYPOINTS_DIR, face_img_name.replace('png', 'txt'))
        np.savetxt(keypoints_path, np.array(face_landmarks_cropped), fmt='%d', delimiter=',')

        image_raw = Image.open(crop_face_path).convert('RGB')
        img_size = image_raw.size[0]

        kp = np.loadtxt(keypoints_path, delimiter=',')
        kp = torch.tensor(kp) * (512 / img_size)
        kp[kp < 0] = 0

        kp_array = np.array(kp, dtype='float32')
        nth = get_nth(kp_array, [512, 512, 3], black=True)
        nth = transform(nth)
        save_image(nth, os.path.join(NTH_DIR, face_img_name), normalize=True)

        image_seg_input = transform_resize(image_raw)
        image_seg_input = image_seg_input.unsqueeze(0).cuda()

        image_seg_output, image_seg_sigmoid = get_seg(seg_model, image_seg_input, image_seg_input.shape[2:], sigmoid=True)

        mask_hair = get_seg_mask(image_seg_output, region='hair')[0]
        hair_path = os.path.join(HAIR_MASK_DIR, face_img_name)
        save_image(mask_hair, hair_path, normalize=True)

        mask_face = get_seg_mask(image_seg_output, region='face')[0]
        face_path = os.path.join(FACE_MASK_DIR, face_img_name)
        save_image(mask_face, face_path, normalize=True)


if __name__ == '__main__':
    run()
