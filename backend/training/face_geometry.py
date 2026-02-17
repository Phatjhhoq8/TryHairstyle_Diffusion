"""
Face Geometry Module — Landmark-based Scalp Mask

Tính toán Face Geometry từ landmarks (5-point kps hoặc 106-point),
sinh scalp mask, detect bald, tạo hybrid mask.

Không phụ thuộc model nào — chỉ dùng numpy + cv2.
"""

import numpy as np
import cv2


# ============================================================
# LANDMARK INDEX MAP (InsightFace 106-point)
# ============================================================
# 0-32:   Face contour / Jawline (33 điểm)
#         0=tai phải, 16=cằm, 32=tai trái
# 33-42:  Lông mày (10 điểm)
# 43-46:  Sống mũi (4 điểm)
# 47-51:  Đáy mũi / cánh mũi (5 điểm)
# 52-57:  Mắt phải — upper (6 điểm)
# 58-63:  Mắt phải — lower (6 điểm)
# 64-69:  Mắt trái — upper (6 điểm)
# 70-75:  Mắt trái — lower (6 điểm)
# 76-87:  Miệng outer (12 điểm)
# 88-95:  Miệng inner (8 điểm)
# 96-97:  Đồng tử (2 điểm)
# 98-105: Iris (8 điểm)

# InsightFace 5-point kps:
# kps[0] = mắt phải (viewer's right)
# kps[1] = mắt trái (viewer's left)
# kps[2] = mũi
# kps[3] = khóe miệng phải
# kps[4] = khóe miệng trái


class FaceGeometry:
    """
    Tính toán Face Geometry từ InsightFace landmarks.
    
    Hỗ trợ 2 nguồn dữ liệu:
    - kps_5 (bắt buộc): 5-point keypoints — luôn có
    - landmark_106 (optional): 106-point landmarks — ưu tiên khi có
    """

    def __init__(self, kps_5, landmark_106=None):
        """
        Args:
            kps_5: ndarray (5, 2) — InsightFace 5-point keypoints
            landmark_106: ndarray (106, 2) hoặc None — InsightFace 106-point landmarks
        """
        self.kps_5 = np.array(kps_5, dtype=np.float64)
        self.landmark_106 = np.array(landmark_106, dtype=np.float64) if landmark_106 is not None else None
        self.has_detailed = self.landmark_106 is not None

        # Tính các property cơ bản
        self._compute_eyes()
        self._compute_nose()
        self._compute_jawline()
        self._compute_derived()

    def _compute_eyes(self):
        """Tính tâm mắt — ưu tiên 106 landmarks (contour mắt đầy đủ)."""
        if self.has_detailed:
            lm = self.landmark_106
            # Mắt phải: trung bình upper(52-57) + lower(58-63)
            self.right_eye = np.mean(lm[52:64], axis=0)
            # Mắt trái: trung bình upper(64-69) + lower(70-75)
            self.left_eye = np.mean(lm[64:76], axis=0)
        else:
            # Fallback: 5-point kps
            self.right_eye = self.kps_5[0].copy()
            self.left_eye = self.kps_5[1].copy()

    def _compute_nose(self):
        """Tính vị trí mũi — ưu tiên 106 landmarks."""
        if self.has_detailed:
            # Index 46: sống mũi cuối (gần đầu mũi nhất trong nhóm 43-46)
            self.nose = self.landmark_106[46].copy()
        else:
            self.nose = self.kps_5[2].copy()

    def _compute_jawline(self):
        """Lấy jawline contour — chỉ có từ 106 landmarks."""
        if self.has_detailed:
            self.jawline = self.landmark_106[0:33].copy()
            self.chin = self.landmark_106[16].copy()
        else:
            self.jawline = None
            self.chin = None

    def _compute_derived(self):
        """Tính các giá trị dẫn xuất: eye_center, eye_distance, yaw."""
        self.eye_center = (self.left_eye + self.right_eye) / 2.0
        self.eye_distance = float(np.linalg.norm(self.right_eye - self.left_eye))

        # Yaw: ước lượng góc quay đầu từ vị trí mũi so với eye center
        # Dùng tỷ lệ khoảng cách ngang mũi-eye vs khoảng cách 2 mắt
        # Khi frontal: nose gần eye_center → yaw ≈ 0
        # Khi profile 45°: nose lệch ~0.5 eye_dist
        # Khi profile 90°: nose lệch ~1.0 eye_dist (nhưng InsightFace hiếm detect được)
        nose_offset_x = self.nose[0] - self.eye_center[0]
        # Dùng atan2 cho smooth mapping, scale factor = eye_distance
        # Tỷ lệ nose_offset / eye_distance ≈ sin(yaw) * 0.8 (anthropometry)
        ratio = nose_offset_x / (self.eye_distance + 1e-6)
        # Clamp ratio: max ±0.7 (profile 90° thực tế không quá 0.7 với InsightFace)
        ratio = np.clip(ratio, -0.7, 0.7)
        # Map to angle: ratio 0.7 → ~45°
        self.yaw = float(np.degrees(np.arctan2(ratio, 0.7)))

    def get_face_area_from_bbox(self, bbox):
        """Tính diện tích mặt từ bbox [x1, y1, x2, y2]."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def get_face_area_from_jawline(self):
        """Tính diện tích mặt từ jawline contour (chính xác hơn bbox)."""
        if self.jawline is None:
            return None
        hull = cv2.convexHull(self.jawline.astype(np.int32))
        return float(cv2.contourArea(hull))

    def __repr__(self):
        return (
            f"FaceGeometry(eye_dist={self.eye_distance:.1f}, "
            f"yaw={self.yaw:.1f}°, "
            f"detailed={'106pt' if self.has_detailed else '5pt'})"
        )


def generate_scalp_mask(geometry, image_shape):
    """
    Sinh scalp mask = ellipse phía trên lông mày.
    
    Anchor: eyebrow top (từ 106 landmarks) hoặc eye_center (fallback 5pt).
    Bottom edge luôn ở eyebrow line — KHÔNG BAO GIỜ lọt vào vùng mặt.
    Sau đó subtract face convex hull để loại bỏ overlap.

    Args:
        geometry: FaceGeometry instance
        image_shape: (h, w) hoặc (h, w, c)

    Returns:
        mask: ndarray (h, w), uint8, 0 hoặc 255
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    eye_dist = geometry.eye_distance
    if eye_dist < 10:
        return mask

    yaw_rad = np.radians(geometry.yaw)

    # === Xác định anchor_y (đỉnh lông mày) ===
    if geometry.has_detailed:
        lm = geometry.landmark_106
        # Lông mày: indices 33-42
        eyebrow_pts = lm[33:43]
        # Lấy Y cao nhất (nhỏ nhất) của lông mày
        anchor_y = float(np.min(eyebrow_pts[:, 1]))
        # Center X: trung bình X lông mày (ổn hơn eye_center cho profile)
        anchor_x = float(np.mean(eyebrow_pts[:, 0]))
    else:
        # Fallback: trên eye_center 20% eye_dist (ước lượng eyebrow)
        anchor_y = geometry.eye_center[1] - eye_dist * 0.20
        anchor_x = geometry.eye_center[0]

    # === Ellipse axes ===
    # Chiều rộng: rộng hơn eye_dist để cover thái dương
    yaw_cos = max(0.60, np.cos(yaw_rad))
    axis_x = int(eye_dist * 1.10 * yaw_cos)

    # Chiều cao: scalp cao ~75% eye_dist
    axis_y = int(eye_dist * 0.75)

    # === Ellipse center ===
    # Bottom edge = anchor_y (eyebrow line)
    # Center = anchor_y - axis_y (chỉ extend lên trên)
    center_y = int(anchor_y - axis_y)
    center_x = int(anchor_x)

    # Dịch ngang nhẹ theo yaw
    center_x += int(eye_dist * 0.05 * np.sin(yaw_rad))

    # Rotation nhẹ theo yaw
    rotation_angle = geometry.yaw * 0.08

    # Clamp center
    center_x = max(axis_x, min(w - axis_x, center_x))
    center_y = max(0, min(h - 1, center_y))

    # Vẽ ellipse
    if axis_x > 0 and axis_y > 0:
        cv2.ellipse(
            mask,
            (center_x, center_y),
            (axis_x, axis_y),
            rotation_angle,
            0, 360,
            255, -1
        )

    # === Subtract vùng mặt ===
    # Cắt bỏ mọi pixel từ trên lông mày trở xuống (margin 15% eye_dist)
    cutoff_y = int(anchor_y - eye_dist * 0.15)
    if cutoff_y < h:
        mask[cutoff_y:, :] = 0

    # Nếu có jawline, subtract convex hull thêm (cho trường hợp ellipse cong)
    if geometry.jawline is not None:
        face_hull = cv2.convexHull(geometry.jawline.astype(np.int32))
        cv2.fillConvexPoly(mask, face_hull, 0)

    return mask


def detect_bald(hair_mask, face_area, threshold=0.10):
    """
    Detect bald: nếu hair_ratio < threshold → coi là bald.

    Args:
        hair_mask: binary mask (0/255) từ BiSeNet, hoặc None
        face_area: diện tích khuôn mặt (từ bbox hoặc jawline)
        threshold: ngưỡng bald (default 10%)

    Returns:
        (is_bald: bool, hair_ratio: float)
    """
    if hair_mask is None or face_area <= 0:
        return True, 0.0

    hair_area = float(np.sum(hair_mask > 0))
    hair_ratio = hair_area / face_area

    is_bald = hair_ratio < threshold
    return is_bald, hair_ratio


def create_hybrid_mask(scalp_mask, hair_mask=None, is_bald=False, seg_valid=True, geometry=None):
    """
    Tạo hybrid mask kết hợp scalp + hair.

    Logic:
    - Có tóc + segmentation OK → scalp_mask ∪ hair_mask
    - Bald (hair_ratio < threshold) → scalp_mask only
    - Segmentation lỗi / None  → scalp_mask only (fallback)
    
    Cuối cùng: subtract face region (jawline hull) để đảm bảo 
    hybrid mask KHÔNG BAO GIỜ chứa pixel trong vùng mặt.

    Args:
        scalp_mask: binary mask (0/255) từ generate_scalp_mask()
        hair_mask: binary mask (0/255) từ BiSeNet, hoặc None
        is_bald: True nếu detect_bald() trả về True
        seg_valid: False nếu BiSeNet parse thất bại
        geometry: FaceGeometry instance (optional, dùng để subtract face)

    Returns:
        (hybrid_mask: ndarray, method: str)
    """
    # Fallback: chỉ dùng scalp
    if not seg_valid or hair_mask is None:
        result = scalp_mask.copy()
        method = 'scalp_only (no_segmentation)'
    elif is_bald:
        result = scalp_mask.copy()
        method = 'scalp_only (bald)'
    else:
        # Có tóc: kết hợp scalp + hair
        result = cv2.bitwise_or(scalp_mask, hair_mask)
        method = 'scalp+hair'

    # === SUBTRACT FACE REGION ===
    # Đảm bảo hybrid mask không bao giờ chứa pixel trong vùng mặt
    if geometry is not None:
        # 1. Cắt bỏ dưới eyebrow line
        if geometry.has_detailed:
            eyebrow_y = int(np.min(geometry.landmark_106[33:43, 1]))
        else:
            eyebrow_y = int(geometry.eye_center[1] - geometry.eye_distance * 0.10)
        result[eyebrow_y:, :] = 0

        # 2. Subtract jawline convex hull 
        if geometry.jawline is not None:
            face_hull = cv2.convexHull(geometry.jawline.astype(np.int32))
            cv2.fillConvexPoly(result, face_hull, 0)

    return result, method
