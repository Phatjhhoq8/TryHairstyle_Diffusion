"""
Preprocessing Service for HairFusion.
Moved from root pipeline_utils.py → backend/app/services/preprocessing.py
Handles face detection, segmentation, cropping, and mask generation.
"""
import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from kornia.morphology import dilation, erosion
from torchvision.utils import save_image
from skimage import io, img_as_float32

# Library imports (resolved via sys.path in backend/__init__.py)
from ffhq_dataset.landmarks_detector import LandmarksDetector
from face_parsing.model import BiSeNet

# App imports
from backend.app.utils.image_utils import (
    get_seg, get_seg_mask, get_crop_coords_crop,
    get_binary_from_img, get_forehead
)
from backend.app.config import SEG_MODEL_PATH, LANDMARKS_MODEL_PATH, DATA_DIR


class PreprocessingService:
    """Handles face preprocessing: landmarks, segmentation, cropping."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Landmarks Detector
        if not os.path.isfile(LANDMARKS_MODEL_PATH):
            raise FileNotFoundError(f"{LANDMARKS_MODEL_PATH} not found!")
        self.landmarks_detector = LandmarksDetector(LANDMARKS_MODEL_PATH)

        # Initialize Segmentation Model
        self.seg_model = BiSeNet(n_classes=16)
        self.seg_model.to(self.device)
        if not os.path.isfile(SEG_MODEL_PATH):
            raise FileNotFoundError(f"{SEG_MODEL_PATH} not found!")
        self.seg_model.load_state_dict(torch.load(SEG_MODEL_PATH))
        self.seg_model.eval()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_resize = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def get_nth(self, img, frame_shape, head=None, hairline=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dpi = 100
        fig = plt.figure(figsize=(frame_shape[0] / dpi, frame_shape[1] / dpi), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
        plt.imshow(img)
        fig.canvas.draw()
        try:
            s, (width, height) = fig.canvas.print_to_buffer()
            nth = Image.frombuffer('RGBA', (width, height), s, 'raw', 'RGBA', 0, 1).convert('RGB')
        except:
            try:
                rgb = fig.canvas.tostring_rgb()
            except AttributeError:
                rgb = fig.canvas.tobytes()
            nth = Image.frombuffer('RGB', fig.canvas.get_width_height(), rgb, 'raw', 'RGB', 0, 1)

        plt.close(fig)
        return nth

    def run_preprocess(self, raw_img_path, save_dir_name):
        """
        Preprocess a single image: detect face, crop, extract landmarks,
        generate NTH, hair mask, face mask, and DensePose placeholder.
        """
        base = os.path.join(DATA_DIR, save_dir_name)
        IMAGES_DIR = os.path.join(base, 'images')
        KEYPOINTS_DIR = os.path.join(base, 'keypoints')
        HAIR_MASK_DIR = os.path.join(base, 'mask_hair')
        FACE_MASK_DIR = os.path.join(base, 'mask_face')
        NTH_DIR = os.path.join(base, 'nth')
        DENSEPOSE_DIR = os.path.join(base, 'images-densepose')

        for d in [IMAGES_DIR, KEYPOINTS_DIR, HAIR_MASK_DIR, FACE_MASK_DIR, NTH_DIR, DENSEPOSE_DIR]:
            os.makedirs(d, exist_ok=True)

        face_img_name = os.path.basename(raw_img_path)
        face_img_name_base = os.path.splitext(face_img_name)[0]
        face_img_name = face_img_name_base + '.png'

        # 1. Landmarks & Cropping
        landmarks = self.landmarks_detector.get_landmarks(raw_img_path)
        if len(landmarks) == 0:
            raise Exception("No face detected in image!")

        face_landmarks = landmarks[0]

        crop_face_path = os.path.join(IMAGES_DIR, face_img_name)
        raw_img = img_as_float32(io.imread(raw_img_path))

        raw_img_cropped = get_crop_coords_crop(
            np.array(face_landmarks),
            (raw_img.shape[1], raw_img.shape[0]),
            raw_img, scale=4.0
        )
        raw_img_cropped = (raw_img_cropped * 255).astype(np.uint8)
        raw_img_cropped_pil = Image.fromarray(raw_img_cropped)
        raw_img_cropped_pil.save(crop_face_path)

        # Re-detect landmarks on crop
        face_landmarks_cropped = self.landmarks_detector.get_landmarks(crop_face_path)[0]

        # Save Keypoints
        keypoints_path = os.path.join(KEYPOINTS_DIR, face_img_name.replace('.png', '.txt'))
        np.savetxt(keypoints_path, np.array(face_landmarks_cropped), fmt='%d', delimiter=',')

        # 2. NTH Generation
        image_raw = Image.open(crop_face_path).convert('RGB')
        img_size = image_raw.size[0]

        kp = np.array(face_landmarks_cropped) * (512 / img_size)
        kp[kp < 0] = 0
        kp_array = np.array(kp, dtype='float32')

        nth = self.get_nth(kp_array, [512, 512, 3])
        nth_tensor = self.transform(nth)
        save_image(nth_tensor, os.path.join(NTH_DIR, face_img_name), normalize=True)

        # 3. Segmentation (Hair & Face Masks)
        image_seg_input = self.transform_resize(image_raw).unsqueeze(0).to(self.device)
        image_seg_output, _ = get_seg(self.seg_model, image_seg_input, image_seg_input.shape[2:], sigmoid=True)

        mask_hair = get_seg_mask(image_seg_output, region='hair')[0]
        save_image(mask_hair, os.path.join(HAIR_MASK_DIR, face_img_name), normalize=True)

        mask_face = get_seg_mask(image_seg_output, region='face')[0]
        save_image(mask_face, os.path.join(FACE_MASK_DIR, face_img_name), normalize=True)

        # 4. DensePose Placeholder
        dp_placeholder = torch.zeros(3, 512, 512)
        save_image(dp_placeholder, os.path.join(DENSEPOSE_DIR, face_img_name_base + '.jpg'), normalize=True)

        return face_img_name

    def run_make_agnostic(self, face_img_name, save_dir_name):
        """
        Generate agnostic image (face preserved, hair removed) for inpainting.
        """
        base = os.path.join(DATA_DIR, save_dir_name)

        KEYPOINTS_DIR = os.path.join(base, 'keypoints')
        IMAGES_DIR = os.path.join(base, 'images')
        HAIR_MASK_DIR = os.path.join(base, 'mask_hair')
        FACE_MASK_DIR = os.path.join(base, 'mask_face')
        AGN_DIR = os.path.join(base, 'agnostic')
        AGN_MASK_DIR = os.path.join(base, 'agnostic-mask')

        os.makedirs(AGN_DIR, exist_ok=True)
        os.makedirs(AGN_MASK_DIR, exist_ok=True)

        k_face_size_mean = 0.26

        # Load Image
        crop_face_path = os.path.join(IMAGES_DIR, face_img_name)
        image_raw = Image.open(crop_face_path).convert('RGB')
        img_size = image_raw.size[0]
        image = self.transform(image_raw)

        # Load Keypoints
        keypoints_path = os.path.join(KEYPOINTS_DIR, face_img_name.replace('.png', '.txt'))
        kp = np.loadtxt(keypoints_path, delimiter=',')
        kp = torch.tensor(kp) * (512 / img_size)

        x_diff = abs(kp[:, 0].max() - kp[:, 0].min()) / 512
        y_diff = abs(kp[:, 1].max() - kp[:, 1].min()) / 512
        face_size_mean = (x_diff + y_diff) / 2

        # Hair Mask Dilation
        hair_path = os.path.join(HAIR_MASK_DIR, face_img_name)
        mask_hair = get_binary_from_img(hair_path)
        dil_size = int(50 * (face_size_mean / k_face_size_mean))
        mask_hair_dil = dilation(mask_hair.unsqueeze(0), torch.ones((dil_size, dil_size)))[0]

        # Face Mask Processing
        face_path = os.path.join(FACE_MASK_DIR, face_img_name)
        mask_face = get_binary_from_img(face_path)

        try:
            mask_forehead = get_forehead(mask_face[0:1, ], kp.unsqueeze(0))[0]
        except Exception:
            mask_forehead = torch.zeros_like(mask_face)

        mask_forehead_dil = dilation(mask_forehead.unsqueeze(0), torch.ones((5, 5)))[0]
        mask_face_wo_fh = mask_face * (1 - mask_forehead_dil)
        mask_face_wo_fh = erosion(mask_face_wo_fh.unsqueeze(0), torch.ones((5, 5)))[0]
        mask_face_wo_fh = dilation(mask_face_wo_fh.unsqueeze(0), torch.ones((3, 3)))[0]

        # Body estimation using landmarks (DensePose bypass)
        l_end_x, l_mid_x = kp[0, 0], (kp[6, 0] + kp[7, 0]) / 2
        r_end_x, r_mid_x = kp[16, 0], (kp[9, 0] + kp[10, 0]) / 2

        chin_y = kp[8, 1]
        jaw_width = abs(kp[16, 0] - kp[0, 0])
        start_y = chin_y + (jaw_width * 0.2)
        end_y = 512

        start_y = max(0, min(start_y, 512))

        l_wid = 3 * abs(l_end_x - l_mid_x)
        r_wid = 3 * abs(r_end_x - r_mid_x)

        l_start_x = max(0, l_mid_x - l_wid)
        l_end_x = l_mid_x
        r_end_x = min(r_mid_x + r_wid, 512)
        r_start_x = r_mid_x

        start_y, end_y = int(start_y), int(end_y)
        l_start_x, l_end_x = int(l_start_x), int(l_end_x)
        r_start_x, r_end_x = int(r_start_x), int(r_end_x)

        # Adjust using hair mask
        hair_mask_original_1ch = torch.sum(mask_hair_dil, axis=0, keepdims=True)
        hair_mask_1ch = (hair_mask_original_1ch > 0.2) * 1

        nonzero_hair = hair_mask_1ch.nonzero()
        if len(nonzero_hair) > 0:
            start_x_hair = torch.min(nonzero_hair[:, 2])
            end_x_hair = torch.max(nonzero_hair[:, 2])
            l_start_x = min(l_start_x, start_x_hair)
            r_end_x = max(r_end_x, end_x_hair)

        agnostic_premask = torch.zeros_like(image)
        if start_y < 512:
            agnostic_premask[:, start_y:, l_start_x:r_end_x] = 1

        agnostic = image.clone()
        agnostic[agnostic_premask == 1] = 0
        agnostic *= (1 - mask_hair_dil)
        agnostic[mask_face_wo_fh > 0] = image[mask_face_wo_fh > 0]

        agnostic_mask = (agnostic != 0) * 1.0
        agnostic_mask = dilation(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic_mask = erosion(agnostic_mask.unsqueeze(0), torch.ones((10, 10)))[0]
        agnostic = image * agnostic_mask

        save_image(agnostic, os.path.join(AGN_DIR, face_img_name), normalize=True)
        save_image(agnostic_mask, os.path.join(AGN_MASK_DIR, face_img_name), normalize=True)
