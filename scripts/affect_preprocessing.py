import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. FACEMESH SUBSET DEFINITIONS
# =========================================================
FACEMESH_LIPS = frozenset([
    (61,146),(146,91),(91,181),(181,84),(84,17),
    (17,314),(314,405),(405,321),(321,375),
    (375,291),(61,185),(185,40),(40,39),(39,37),
    (37,0),(0,267),(267,269),(269,270),(270,409),
    (409,291),(78,95),(95,88),(88,178),(178,87),
    (87,14),(14,317),(317,402),(402,318),(318,324),
    (324,308),(78,191),(191,80),(80,81),(81,82),
    (82,13),(13,312),(312,311),(311,310),
    (310,415),(415,308)
])

FACEMESH_LEFT_EYE = frozenset([
    (263,249),(249,390),(390,373),(373,374),
    (374,380),(380,381),(381,382),(382,362),
    (263,466),(466,388),(388,387),(387,386),
    (386,385),(385,384),(384,398),(398,362)
])

FACEMESH_RIGHT_EYE = frozenset([
    (33,7),(7,163),(163,144),(144,145),
    (145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),
    (159,158),(158,157),(157,173),(173,133)
])

FACEMESH_LEFT_EYEBROW = frozenset([
    (276,283),(283,282),(282,295),(295,285),
    (300,293),(293,334),(334,296),(296,336)
])

FACEMESH_RIGHT_EYEBROW = frozenset([
    (46,53),(53,52),(52,65),(65,55),
    (70,63),(63,105),(105,66),(66,107)
])

FACEMESH_FACE_OVAL = frozenset([
    (10,338),(338,297),(297,332),(332,284),
    (284,251),(251,389),(389,356),(356,454),
    (454,323),(323,361),(361,288),(288,397),
    (397,365),(365,379),(379,378),(378,400),
    (400,377),(377,152),(152,148),(148,176),
    (176,149),(149,150),(150,136),(136,172),
    (172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),
    (54,103),(103,67),(67,109),(109,10)
])

FACEMESH_NOSE = frozenset([
    (168,6),(6,197),(197,195),(195,5),
    (5,4),(4,1),(1,19),(19,94),(94,2),
    (98,97),(97,2),(2,326),(326,327),
    (327,294),(294,278),(278,344),
    (344,440),(440,275),(275,4),
    (4,45),(45,220),(220,115),
    (115,48),(48,64),(64,98)
])

SELECTED_CONNECTIONS = (
    FACEMESH_LIPS
    | FACEMESH_LEFT_EYE
    | FACEMESH_RIGHT_EYE
    | FACEMESH_LEFT_EYEBROW
    | FACEMESH_RIGHT_EYEBROW
    | FACEMESH_FACE_OVAL
    | FACEMESH_NOSE
)

SUBSET_INDICES = [
    0, 1, 2, 4, 5, 6, 7, 10, 13, 14, 17, 19, 21, 33, 37, 39, 40, 45, 48,
    54, 58, 61, 63, 64, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 94,
    95, 97, 98, 103, 105, 107, 109, 115, 127, 132, 133, 136, 144, 145, 146,
    148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
    168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 220, 234, 246, 249,
    251, 263, 267, 269, 270, 275, 278, 284, 288, 291, 293, 294, 296, 297,
    300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 326, 327, 332,
    334, 336, 338, 344, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379,
    380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402,
    405, 409, 415, 440, 454, 466
]


# =========================================================
# 2. LANDMARK NORMALIZATION
# =========================================================
def normalize_landmark_subset(
    landmarks,
    subset_indices,
    image_width,
    image_height,
    left_eye_idx=33,
    right_eye_idx=263
):
    points = {}
    for i, lm in enumerate(landmarks):
        x = lm.x * image_width
        y = lm.y * image_height
        points[i] = np.array([x, y], dtype=np.float32)

    p_left = points[left_eye_idx]
    p_right = points[right_eye_idx]

    center = (p_left + p_right) / 2.0
    cx, cy = center

    scale = np.linalg.norm(p_left - p_right)
    if scale == 0:
        raise ValueError("Scale is zero.")

    normalized_points = []
    for idx in subset_indices:
        p = points[idx]
        x_norm = (p[0] - cx) / scale
        y_norm = (p[1] - cy) / scale
        normalized_points.append([x_norm, y_norm])

    return np.array(normalized_points, dtype=np.float32)


# =========================================================
# 3. FACE CROP EXTRACTION
# =========================================================
def extract_face_crop(image_rgb, landmarks, output_size=(224, 224), margin=0.20):
    h, w, _ = image_rgb.shape

    xs = np.array([lm.x * w for lm in landmarks], dtype=np.float32)
    ys = np.array([lm.y * h for lm in landmarks], dtype=np.float32)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_min -= margin * box_w
    x_max += margin * box_w
    y_min -= margin * box_h
    y_max += margin * box_h

    x_min = max(0, int(np.floor(x_min)))
    y_min = max(0, int(np.floor(y_min)))
    x_max = min(w, int(np.ceil(x_max)))
    y_max = min(h, int(np.ceil(y_max)))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid face crop box.")

    crop = image_rgb[y_min:y_max, x_min:x_max]
    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return crop


# =========================================================
# 4. FACE LANDMARKER WRAPPER
# =========================================================
class AffectPreprocessor:
    def __init__(
        self,
        model_path: str,
        subset_indices=None,
        output_face_size=(224, 224),
        margin=0.20,
    ):
        self.model_path = model_path
        self.subset_indices = subset_indices if subset_indices is not None else SUBSET_INDICES
        self.output_face_size = output_face_size
        self.margin = margin

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        """
        Input:
            frame_bgr: webcam frame in BGR
        Returns:
            dict with keys:
                success: bool
                face_crop: np.ndarray or None
                landmarks: np.ndarray or None
                error: str or None
        """
        try:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = image_rgb.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self.detector.detect(mp_image)

            if len(result.face_landmarks) == 0:
                return {
                    "success": False,
                    "face_crop": None,
                    "landmarks": None,
                    "error": "no_face_detected"
                }

            landmarks = result.face_landmarks[0]

            normalized_points = normalize_landmark_subset(
                landmarks=landmarks,
                subset_indices=self.subset_indices,
                image_width=w,
                image_height=h
            )

            face_crop = extract_face_crop(
                image_rgb=image_rgb,
                landmarks=landmarks,
                output_size=self.output_face_size,
                margin=self.margin
            )

            return {
                "success": True,
                "face_crop": face_crop,               # RGB image
                "landmarks": normalized_points,      # (N, 2)
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "face_crop": None,
                "landmarks": None,
                "error": str(e)
            }

    def close(self):
        if self.detector is not None:
            self.detector.close()