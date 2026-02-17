import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

try:
    from backend.training.face_geometry import FaceGeometry, generate_scalp_mask, detect_bald, create_hybrid_mask
except ImportError:
    from face_geometry import FaceGeometry, generate_scalp_mask, detect_bald, create_hybrid_mask

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
MODELS_DIR = BACKEND_DIR / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"
INSIGHTFACE_ROOT = BACKEND_DIR
ADAFACE_MODEL_PATH = MODELS_DIR / "adaface_ir101_webface4m.ckpt"

print(f"--- DEVICE: {DEVICE} ---")
print(f"--- MODEL DIR: {MODELS_DIR} ---")

# --- PHẦN 1: KIẾN TRÚC MODEL ADAFACE (ResNet) ---
# Copy từ adaface_ir.py để file này độc lập

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

def iresnet100(pretrained=False, progress=True, **kwargs):
    return IResNet(IBasicBlock, [3, 13, 30, 3], **kwargs)


# --- PHẦN 2: CÁC SERVICE XỬ LÝ ---

class IndependentAdaFace:
    def __init__(self, model_path=ADAFACE_MODEL_PATH):
        self.device = torch.device(DEVICE)
        self.model = None
        self.mtcnn = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(
                image_size=112, margin=0, min_face_size=20,
                thresholds=[0.5, 0.6, 0.6], factor=0.709,
                post_process=False, device=self.device
            )
            
            if os.path.exists(self.model_path):
                self.model = iresnet100()
                checkpoint = torch.load(self.model_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                else:
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                print(f"[AdaFace] Loaded model from {self.model_path}")
            else:
                print(f"[AdaFace] Model not found: {self.model_path}")

        except ImportError as e:
            print(f"[AdaFace] Missing dependency or import error: {e}")
            print("[AdaFace] Try: pip install facenet-pytorch")
        except Exception as e:
            print(f"[AdaFace] Error loading model: {e}")
            import traceback
            traceback.print_exc()

    def _to_input(self, aligned_face_rgb):
        if isinstance(aligned_face_rgb, Image.Image):
            img = np.array(aligned_face_rgb)
        else:
            img = aligned_face_rgb
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        img_bgr = img[:, :, ::-1].copy() # RGB -> BGR
        img_normalized = (img_bgr.astype(np.float32) - 127.5) / 127.5
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1))
        img_tensor = img_tensor.unsqueeze(0).float()
        return img_tensor.to(self.device)

    def get_embedding(self, image_cv2, bbox=None):
        if self.model is None or self.mtcnn is None: return None
        
        # Align (MTCNN expects RGB)
        image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        if bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            h, w = image_rgb.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            image_rgb = image_rgb[y1:y2, x1:x2]

        pil_image = Image.fromarray(image_rgb)
        try:
            aligned = self.mtcnn(pil_image)
            if aligned is not None:
                aligned_np = aligned.permute(1, 2, 0).cpu().numpy()
                aligned_np = np.clip(aligned_np, 0, 255).astype(np.uint8)
                
                # Extract Embedding
                input_tensor = self._to_input(aligned_np)
                with torch.no_grad():
                    embedding = self.model(input_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"[AdaFace] Error extracting: {e}")
        return None


class IndependentYOLOFace:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                print(f"[YOLO] Model not found: {self.model_path}")
                return
            self.model = YOLO(self.model_path)
            print(f"[YOLO] Loaded model from {self.model_path}")
        except ImportError:
            print("[YOLO] Missing ultralytics. pip install ultralytics")

    def detect(self, image_cv2, conf=0.4):
        if self.model is None: return []
        try:
            image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
            results = self.model(image_rgb, verbose=False, conf=conf)
            faces = []
            for result in results:
                if result.boxes is None: continue
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    faces.append({'bbox': bbox, 'confidence': conf})
            return sorted(faces, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        except Exception as e:
            print(f"[YOLO] Error: {e}")
            return []


import torchvision.transforms as transforms
import torchvision

# --- PHẦN 1.5: KIẾN TRÚC MODEL BISENET (Face Parsing) ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return feat * atten

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        return feat * atten + feat

class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32

class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv_out(x)

class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False) # Use ResNet18 backbone
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        feat4 = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(feat4) 
        feat16 = self.resnet.layer3(feat8) 
        feat32 = self.resnet.layer4(feat16) 

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = F.interpolate(feat32_sum, size=feat16.size()[2:], mode='bilinear', align_corners=True)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.size()[2:], mode='bilinear', align_corners=True)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up

class IndependentBiSeNet:
    def __init__(self, model_path, device=DEVICE):
        self.device = device
        self.model = None
        self.load(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def load(self, model_path):
        try:
            print(f"[BiSeNet] Loading model from {model_path}...")
            # N_CLASSES = 19 for CelebAMask-HQ
            self.model = BiSeNet(n_classes=19)
            if self.device == 'cuda':
                self.model.cuda()
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Handle DataParallel prefix 'module.'
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Use strict=False because ResNet backbone has 'fc' layer but weights don't (and we don't use it)
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            print("[BiSeNet] Model loaded successfully.")
        except Exception as e:
            print(f"[BiSeNet] Error loading model: {e}")
            self.model = None

    def parse(self, image_bgr):
        if self.model is None:
            return None
        
        # Resize to 512x512 for inference
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        img_pil = Image.fromarray(img_resized)
        
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out, out16, out32 = self.model(input_tensor)
            
        # Get mask
        out_mask = out.squeeze(0).cpu().numpy().argmax(0)
        
        # Resize mask back to original size
        h, w = image_bgr.shape[:2]
        out_mask_full = cv2.resize(out_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        return out_mask_full

class TrainingFaceSystem:
    def __init__(self):
        print("--- Khởi tạo hệ thống Face Training độc lập ---")
        
        # 1. Khởi tạo YOLO Face
        self.yolo_detector = IndependentYOLOFace(YOLO_MODEL_PATH)
        
        # 2. Khởi tạo InsightFace (cho landmarks 2D 106)
        self.insight_analyser = None
        try:
            from insightface.app import FaceAnalysis
            self.insight_analyser = FaceAnalysis(name='antelopev2', root=str(INSIGHTFACE_ROOT), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.insight_analyser.prepare(ctx_id=0, det_size=(640, 640))
            print("[InsightFace] Initialized")
        except Exception as e:
            print(f"[InsightFace] Error: {e}")

        # 3. Khởi tạo AdaFace
        self.adaface = IndependentAdaFace(ADAFACE_MODEL_PATH)
        
        # 4. Khởi tạo BiSeNet (Segmentation)
        self.bisenet = IndependentBiSeNet(MODELS_DIR / "bisenet" / "79999_iter.pth", device=DEVICE)

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def process(self, image_path):
        if not Path(image_path).exists():
            print(f"Error: Image not found {image_path}")
            return None
        
        # Đọc ảnh
        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            print("Error: Cannot read image (cv2)")
            return None
            
        # --- BƯỚC 1: Detection (YOLO) ---
        detections = self.yolo_detector.detect(img_cv2)
        
        if not detections:
            return None
            
        # Lấy khuôn mặt lớn nhất
        best_face = max(detections, key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]))
        bbox = best_face['bbox']
        score = best_face['confidence']
        kps_yolo = None # YOLO model in original code does not return kps directly in 'detect' method.

        # --- BƯỚC 2: InsightFace (Lấy Landmarks 106 & Embedding) ---
        target_embedding = None
        target_kps = None
        target_landmark_106 = None
        source = "yolo"
        
        if self.insight_analyser:
            faces = self.insight_analyser.get(img_cv2)
            if faces:
                # Tìm face khớp với bbox của YOLO nhất
                # Tính IOU đơn giản hoặc khoảng cách tâm
                def get_center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
                yolo_center = get_center(bbox)
                
                best_ins_face = min(faces, key=lambda f: np.linalg.norm(np.array(get_center(f.bbox)) - np.array(yolo_center)))
                
                # Check nếu face này hợp lý (gần nhau)
                dist = np.linalg.norm(np.array(get_center(best_ins_face.bbox)) - np.array(yolo_center))
                
                if dist < 50: # Ngưỡng chấp nhận
                    target_embedding = best_ins_face.embedding
                    target_kps = best_ins_face.kps
                    if hasattr(best_ins_face, 'landmark_2d_106'):
                        target_landmark_106 = best_ins_face.landmark_2d_106
                    source = "insightface"
                # else:
                #    print(f"[Debug] Distance too large (>50). Skipping InsightFace match.")

                # else:
                #    print(f"[Debug] Distance too large (>50). Skipping InsightFace match.")

        # --- BƯỚC 3: AdaFace (Nếu InsightFace fail hoặc muốn dùng AdaFace extract) ---
        # Ở đây ta ưu tiên dùng AdaFace để extract embedding cho chuẩn theo yêu cầu cũ,
        # hoặc dùng InsightFace nếu AdaFace chưa sẵn sàng.
        # Nhưng AdaFace mạnh hơn cho các góc khó.
        
        # Crop ảnh cho AdaFace (dùng bbox yolo đã mở rộng)
        # AdaFace cần align chuẩn. Ta dùng landmarks 5 điểm từ YOLO (nếu có) hoặc InsightFace để align.
        
        final_embedding = target_embedding
        final_kps = target_kps if target_kps is not None else kps_yolo
        
        # Nếu có kps, dùng AdaFace extract lại cho xịn
        if final_kps is not None and self.adaface.model is not None:
             # Convert kps về format chuẩn list/array
             # The original AdaFace get_embedding expects bbox, not kps.
             # If kps are available, MTCNN inside AdaFace will use them for alignment.
             # For now, we pass the bbox from YOLO.
             emb_ada = self.adaface.get_embedding(img_cv2, bbox=bbox)
             if emb_ada is not None:
                 final_embedding = emb_ada
                 source = "adaface"

        if final_embedding is None:
            return None

        # --- BƯỚC 4: Face Geometry + Scalp Mask + Hybrid Mask ---
        # 4a. Tính Face Geometry từ landmarks
        geometry = None
        if final_kps is not None:
            geometry = FaceGeometry(final_kps, landmark_106=target_landmark_106)
            print(f"[FaceGeometry] {geometry}")

        # 4b. Sinh scalp mask từ geometry
        scalp_mask = None
        if geometry is not None:
            scalp_mask = generate_scalp_mask(geometry, img_cv2.shape)
            print(f"[ScalpMask] Generated: {np.sum(scalp_mask > 0)} pixels")

        # 4c. BiSeNet parse → hair mask
        bisenet_mask = None
        hair_mask_raw = None
        seg_valid = False
        if self.bisenet.model is not None:
            bisenet_mask = self.bisenet.parse(img_cv2)
            if bisenet_mask is not None:
                hair_mask_raw = (bisenet_mask == 17).astype(np.uint8) * 255
                # Filter: loại bỏ hair false positive trong vùng mặt
                if geometry is not None:
                    face_exclude = np.zeros_like(hair_mask_raw)
                    if geometry.jawline is not None:
                        hull = cv2.convexHull(geometry.jawline.astype(np.int32))
                        cv2.fillConvexPoly(face_exclude, hull, 255)
                    # Cắt thêm: mọi hair dưới eyebrow line
                    if geometry.has_detailed:
                        eyebrow_y = int(np.min(geometry.landmark_106[33:43, 1]))
                    else:
                        eyebrow_y = int(geometry.eye_center[1] - geometry.eye_distance * 0.10)
                    face_exclude[eyebrow_y:, :] = 255
                    # Loại bỏ hair trong face region
                    hair_mask_raw = cv2.bitwise_and(hair_mask_raw, cv2.bitwise_not(face_exclude))
                seg_valid = True

        # 4d. Detect bald
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if geometry is not None and geometry.jawline is not None:
            jaw_area = geometry.get_face_area_from_jawline()
            if jaw_area is not None and jaw_area > 0:
                face_area = jaw_area

        is_bald, hair_ratio = detect_bald(hair_mask_raw, face_area)
        print(f"[BaldDetect] is_bald={is_bald}, hair_ratio={hair_ratio:.3f}")

        # 4e. Hybrid mask
        hybrid_mask = None
        hybrid_method = 'none'
        if scalp_mask is not None:
            hybrid_mask, hybrid_method = create_hybrid_mask(
                scalp_mask, hair_mask_raw, is_bald, seg_valid, geometry=geometry
            )
            print(f"[HybridMask] Method: {hybrid_method}")

        return {
            'bbox': bbox,
            'kps': final_kps,
            'landmark_2d_106': target_landmark_106,
            'embedding': final_embedding,
            'score': score,
            'source': source,
            'geometry': geometry,
            'scalp_mask': scalp_mask,
            'hair_mask': hair_mask_raw,
            'bisenet_mask': bisenet_mask,
            'is_bald': is_bald,
            'hair_ratio': hair_ratio,
            'hybrid_mask': hybrid_mask,
            'hybrid_method': hybrid_method,
        }

    def visualize_embedding(self, embedding, width=512, height=50):
        # Normalize embedding về 0..255 để vẽ
        norm_emb = np.clip(embedding, -0.08, 0.08)
        norm_emb = (norm_emb + 0.08) / 0.16 * 255
        norm_emb = norm_emb.astype(np.uint8)
        
        # Reshape thành ảnh (1, 512) rồi resize
        emb_img = norm_emb.reshape(1, -1)
        emb_img = np.tile(emb_img, (height, 1))
        
        # Tạo heatmap màu
        heatmap = cv2.applyColorMap(emb_img, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return heatmap

    def embedding_to_color(self, embedding):
        """
        Chuyển đổi embedding (512D) thành một màu RGB duy nhất (Identity Color).
        Cách đơn giản: PCA giảm chiều hoặc lấy trung bình các đoạn.
        Ở đây dùng cách hash đơn giản để tạo màu cố định cho mỗi vector.
        """
        # Normalize vector
        emb = embedding / np.linalg.norm(embedding)
        
        # Chia 512 chiều thành 3 phần (R, G, B)
        # 512 không chia hết cho 3, lấy dư
        r = np.sum(emb[0:170])
        g = np.sum(emb[170:340])
        b = np.sum(emb[340:510])
        
        # Map giá trị sum về 0..255
        # Giá trị sum thường nằm trong khoảng -10..10
        def map_val(v):
            return int(np.clip((v + 2) / 4 * 255, 0, 255))
        
        color = (map_val(b), map_val(g), map_val(r)) # BGR cho OpenCV
        return color

    def visualize_result(self, image_cv2, result):
        img_vis = image_cv2.copy()
        
        if result is None:
            return img_vis

        bbox = [int(x) for x in result['bbox']]
        
        # Tạo màu danh tính từ embedding
        color = self.embedding_to_color(result['embedding'])
        
        # Vẽ bbox
        cv2.rectangle(img_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        
        # Vẽ thông tin
        text = f"{result['source']} ({result['score']:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img_vis, (bbox[0], bbox[1] - 25), (bbox[0] + text_size[0], bbox[1]), color, -1)
        cv2.putText(img_vis, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Vẽ Landmarks (5 points)
        if 'kps' in result and result['kps'] is not None:
            for kp in result['kps']:
                cv2.circle(img_vis, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

        # Vẽ Embedding Heatmap
        embedding = result['embedding']
        heatmap = self.visualize_embedding(embedding, width=bbox[2]-bbox[0])
        
        h_heatmap, w_heatmap = heatmap.shape[:2]
        y_start = bbox[3] + 5
        y_end = y_start + h_heatmap
        
        if y_end < img_vis.shape[0]:
            heatmap_resized = cv2.resize(heatmap, (bbox[2]-bbox[0], h_heatmap))
            img_vis[y_start:y_end, bbox[0]:bbox[2]] = heatmap_resized
            cv2.rectangle(img_vis, (bbox[0], y_start), (bbox[2], y_end), (255, 255, 255), 1)

        return img_vis

    def visualize_red_face(self, image_cv2, result):
        """
        Tạo ảnh overlay màu đỏ lên vùng khuôn mặt VÀ TÓC tìm được.
        Sử dụng BiSeNet Parsing nếu có.
        """
        img_red = image_cv2.copy()
        overlay = image_cv2.copy()
        
        if result is None:
            return img_red
            
        # 1. Thử dùng BiSeNet Segmentation trước
        if self.bisenet.model is not None:
            mask = self.bisenet.parse(image_cv2)
            if mask is not None:
                # Mask labels: 1=skin, 2-13=features, 17=hair
                # Target: Skin + Features + Hair
                # Tạo mask binary: (mask >= 1 & mask <= 13) | (mask == 17)
                target_mask = ((mask >= 1) & (mask <= 13)) | (mask == 17)
                
                # Overlay màu đỏ lên vùng True
                overlay[target_mask] = (0, 0, 255)
                
                # Blend
                alpha = 0.5
                # Chỉ blend ở vùng target_mask
                cv2.addWeighted(overlay, alpha, img_red, 1 - alpha, 0, img_red)
                return img_red

        # 2. Fallback: Dùng landmarks (chỉ có khuôn mặt, ko có tóc)
        print("[Warning] BiSeNet failed or not loaded. Using landmarks fallback.")
        points = None
        
        if 'landmark_2d_106' in result and result['landmark_2d_106'] is not None:
             points = result['landmark_2d_106'].astype(np.int32)
        elif 'kps' in result and result['kps'] is not None:
             points = result['kps'].astype(np.int32)
        
        if points is not None:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(overlay, hull, (0, 0, 255))
        else:
            bbox = [int(x) for x in result['bbox']]
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), -1)
            
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img_red, 1 - alpha, 0, img_red)
        
        return img_red

    def visualize_segmentation(self, image_cv2, result):
        """
        Tạo ảnh Segmentation 3 màu dùng hybrid_mask từ process():
        - Môi trường (Background): Màu Xám
        - Tóc/Scalp (Hair): Màu Trắng
        - Người (Body/Face): Màu Đen
        """
        h, w = image_cv2.shape[:2]
        seg_img = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)

        if result is None:
            return seg_img

        # Face region: từ BiSeNet person labels hoặc scalp complement
        bisenet_mask = result.get('bisenet_mask')
        if bisenet_mask is not None:
            person_mask = ((bisenet_mask >= 1) & (bisenet_mask <= 16)) | (bisenet_mask == 18)
            seg_img[person_mask] = (0, 0, 0)

        # Fill gaps bằng landmarks convex hull
        if result.get('landmark_2d_106') is not None:
            hull = cv2.convexHull(result['landmark_2d_106'].astype(np.int32))
            cv2.fillConvexPoly(seg_img, hull, (0, 0, 0))
        elif result.get('kps') is not None:
            hull = cv2.convexHull(result['kps'].astype(np.int32))
            cv2.fillConvexPoly(seg_img, hull, (0, 0, 0))

        # Hybrid mask (scalp + hair) — vẽ SAU face để đè lên
        hybrid_mask = result.get('hybrid_mask')
        if hybrid_mask is not None:
            seg_img[hybrid_mask > 0] = (255, 255, 255)
        elif result.get('hair_mask') is not None:
            seg_img[result['hair_mask'] > 0] = (255, 255, 255)

        method = result.get('hybrid_method', 'unknown')
        print(f"[Segmentation] Rendered: {method}")
        return seg_img

    def visualize_geometry_debug(self, image_cv2, result):
        """
        Debug visualization:
        - Scalp ellipse (xanh lá)
        - Hair mask contour (trắng)
        - Hybrid mask overlay (đỏ mờ)
        - Thông số: eye_dist, yaw, hair_ratio, is_bald, method
        """
        img_debug = image_cv2.copy()

        if result is None:
            return img_debug

        geometry = result.get('geometry')
        if geometry is None:
            cv2.putText(img_debug, "No FaceGeometry", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img_debug

        # 1. Vẽ scalp ellipse (xanh lá)
        scalp_mask = result.get('scalp_mask')
        if scalp_mask is not None:
            contours, _ = cv2.findContours(scalp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_debug, contours, -1, (0, 255, 0), 2)

        # 2. Vẽ hair mask contour (trắng)
        hair_mask = result.get('hair_mask')
        if hair_mask is not None and np.sum(hair_mask > 0) > 0:
            contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_debug, contours, -1, (255, 255, 255), 1)

        # 3. Hybrid mask overlay (đỏ mờ)
        hybrid_mask = result.get('hybrid_mask')
        if hybrid_mask is not None:
            overlay = img_debug.copy()
            overlay[hybrid_mask > 0] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.25, img_debug, 0.75, 0, img_debug)

        # 4. Vẽ eye markers + nose
        cv2.circle(img_debug, tuple(geometry.left_eye.astype(int)), 4, (255, 0, 0), -1)
        cv2.circle(img_debug, tuple(geometry.right_eye.astype(int)), 4, (255, 0, 0), -1)
        cv2.circle(img_debug, tuple(geometry.nose.astype(int)), 4, (0, 255, 255), -1)
        cv2.circle(img_debug, tuple(geometry.eye_center.astype(int)), 3, (0, 255, 0), 2)

        # 5. Vẽ jawline nếu có (106pt)
        if geometry.jawline is not None:
            pts = geometry.jawline.astype(np.int32)
            cv2.polylines(img_debug, [pts], False, (0, 200, 255), 1)

        # 6. Thông tin text
        info_y = 25
        infos = [
            f"Eye Dist: {geometry.eye_distance:.1f}px",
            f"Yaw: {geometry.yaw:.1f} deg",
            f"Hair Ratio: {result.get('hair_ratio', 0):.3f}",
            f"Bald: {result.get('is_bald', False)}",
            f"Method: {result.get('hybrid_method', 'N/A')}",
            f"Source: {'106pt' if geometry.has_detailed else '5pt'}",
        ]
        for text in infos:
            cv2.putText(img_debug, text, (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            info_y += 18

        return img_debug

    def save_embedding(self, embedding, save_path):
        np.save(save_path, embedding)
        print(f"Saved embedding to: {save_path}")

    def process_and_save(self, image_path, output_dir=None, target_embedding=None, threshold=0.4):
        if output_dir is None:
            output_dir = Path(image_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"\nScanning: {image_path}")
        result = self.process(image_path)
        
        img_name = Path(image_path).stem
        
        if result:
            print(f"Found face: {result['source']} (Score: {result['score']:.2f})")
            
            # Tính màu đại diện (Output log)
            color = self.embedding_to_color(result['embedding'])
            print(f"Identity Color (BGR): {color}")
            
            # 1. Lưu embedding
            emb_path = output_dir / f"{img_name}_embedding.npy"
            self.save_embedding(result['embedding'], emb_path)
            
            # 2. Visualize (Standard BBox + Heatmap)
            orig_img = cv2.imread(str(image_path))
            vis_img = self.visualize_result(orig_img, result)
            vis_path = output_dir / f"{img_name}_visualized.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            print(f"Saved visualization to: {vis_path}")
            
            # 3. Visualize Red Face (Mask Đỏ - Giữ lại cho user tham khảo)
            red_img = self.visualize_red_face(orig_img, result)
            red_path = output_dir / f"{img_name}_red.jpg"
            cv2.imwrite(str(red_path), red_img)
            print(f"Saved red face visualization to: {red_path}")

            # 4. Visualize Segmentation (dùng hybrid mask)
            seg_img = self.visualize_segmentation(orig_img, result)
            seg_path = output_dir / f"{img_name}_segmented.jpg"
            cv2.imwrite(str(seg_path), seg_img)
            print(f"Saved segmented visualization to: {seg_path}")

            # 5. Visualize Geometry Debug
            geom_img = self.visualize_geometry_debug(orig_img, result)
            geom_path = output_dir / f"{img_name}_geometry.jpg"
            cv2.imwrite(str(geom_path), geom_img)
            print(f"Saved geometry debug to: {geom_path}")
            
            return result
        else:
            print("No face detected.")
            return None

def convert_wsl_path(raw_path):
    if sys.platform.startswith('linux') and (raw_path.startswith('C:') or raw_path.startswith('c:')):
        return raw_path.replace('\\', '/').replace('C:', '/mnt/c').replace('c:', '/mnt/c')
    return raw_path

if __name__ == "__main__":
    # Test block
    processor = TrainingFaceSystem()
    
    # Args
    target_img_path = None
    
    args = sys.argv[1:]
    
    if len(args) > 0:
        # Xử lý tham số 1: Ảnh input
        target_img_path = Path(convert_wsl_path(args[0]))
    else:
        # Tự tìm
        upload_dir = BACKEND_DIR / "data" / "uploads"
        if upload_dir.exists():
            for f in upload_dir.iterdir():
                if f.suffix.lower() in ['.jpg', '.png']:
                    target_img_path = f
                    break
    
    # Chạy
    if target_img_path and target_img_path.exists():
        out_dir = BACKEND_DIR / "training" / "output_test"
        processor.process_and_save(target_img_path, output_dir=out_dir)
    else:
        print("Usage:")
        print("  python backend/training/training_face.py <path_to_image>")
        print("Example:")
        print("  python backend/training/training_face.py image.jpg")
