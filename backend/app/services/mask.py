
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from backend.app.config import model_paths, settings

# Định nghĩa model BiSeNet (rút gọn để chạy inference)
# Do cấu trúc BiSeNet khá dài, ta sẽ dùng code tối giản hoặc load state dictionary vào structure.
# Để đơn giản và chính xác, ta cần định nghĩa class BiSeNet tương tự như repo gốc.
# Tuy nhiên, để tránh file quá dài, ở đây ta sẽ dùng một thủ thuật: 
# Load model script từ một file utils hoặc define class ngay đây nếu không quá phức tạp.
# Dưới đây là định nghĩa class BiSeNet cơ bản cần thiết để load weights 79999_iter.pth.

# --- BiSeNet Arch Definitions (Simplified for Inference) ---
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        # Context Path implementattion skipped for brevity in prompt, 
        # BUT CRITICAL: We need the actual architecture to load weights.
        # Since I cannot import it from a library, I must rely on the user providing it 
        # OR implementation details included here.
        # FOR NOW: I will implementing a PLACEHOLDER wrapper that assumes the model structure is standard.
        # If this fails, we will need to copy the full BiSeNet.py code.
        pass
    
    # ... (Full architecture is too long for this single block)

# Để an toàn và thực tế, thay vì define lại toàn bộ mạng (dễ lỗi sai structure),
# ta sẽ dùng thư viện 'transformers' nếu có hỗ trợ SegFormer (mới hơn, tốt hơn).
# NHƯNG yêu cầu là BiSeNet.
# Giải pháp: Ta sẽ viết 1 file utils/bisenet_arch.py chứa kiến trúc mạng (Copy từ GitHub official),
# sau đó import vào đây.
# Tuy nhiên, hiện tại tôi sẽ viết logic xử lý ảnh trước, 
# và giả định ta có thể load model bằng torch.load nếu nó là full model (ko phải chỉ state_dict).
# Nếu là state_dict (file .pth thường là vậy), ta BẮT BUỘC phải có class definition.

# -> Quyết định: Tạo file `backend/app/services/bisenet_arch.py` trước containing full arch.
# Nhưng tôi không thể switch task ngay. 
# -> Tôi sẽ viết file `mask.py` này import từ `bisenet_arch`, và sau bước này tôi sẽ tạo file `bisenet_arch.py`.

from .bisenet_arch import BiSeNet # Sẽ tạo file này ngay sau

class SegmentationService:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.load_state_dict(torch.load(model_paths.BISENET_CHECKPOINT, map_location=self.device), strict=False)
        self.net.to(self.device)
        self.net.eval()
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_mask(self, image_pil, target_class=17): # 17 usually Hair in CelebAMask-HQ
        """
        Input: PIL Image
        Output: Binary Mask (PIL Image) của vùng tóc
        """
        w, h = image_pil.size
        # Resize nhỏ để infer nhanh hơn (e.g. 512x512)
        img_resized = image_pil.resize((512, 512), Image.BILINEAR)
        img_tensor = self.to_tensor(img_resized)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(self.device)

        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
        # Parsing là ma trận 512x512 chứa class ID
        # Lọc class tóc (thường là 17)
        mask = np.zeros_like(parsing).astype(np.uint8)
        mask[parsing == target_class] = 255
        
        # Resize mask về kích thước gốc
        mask_cv2 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Dilate mask để mở rộng vùng biên một chút (giúp inpainting tốt hơn)
        kernel = np.ones((5,5), np.uint8)
        mask_dilated = cv2.dilate(mask_cv2, kernel, iterations=2)
        
        return Image.fromarray(mask_dilated)

