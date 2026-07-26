import cv2
import numpy as np
import mediapipe as mp
import scipy.io as sio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. HELPERS
# =========================================================
def rodrigues_to_matrix(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def matrix_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3)


def normalize_vector(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def equalize_gray(img_gray):
    img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(img_gray)


def get_image_points_from_tasks(face_landmarks, img_w, img_h, landmark_ids):
    pts = []
    for idx in landmark_ids:
        lm = face_landmarks[idx]
        x = lm.x * img_w
        y = lm.y * img_h
        pts.append([x, y])
    return np.array(pts, dtype=np.float64)


def estimate_head_pose(image_points_2d, model_points_3d, camera_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(
        model_points_3d,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None, None

    success, rvec, tvec = cv2.solvePnP(
        model_points_3d,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None

    return rvec.reshape(3), tvec.reshape(3)


def compute_reprojection_error(model_points_3d, image_points_2d, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(
        model_points_3d,
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        camera_matrix,
        dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    point_errs = np.linalg.norm(projected_points - image_points_2d, axis=1)
    mean_err = point_errs.mean()
    return projected_points, point_errs, mean_err


def reconstruct_eye_centers_from_real_model(rvec, tvec, face_model_3x6):
    """
    Exact MPIIGaze MATLAB logic:
      right_eye_center = midpoint(Fc[:,0], Fc[:,1])
      left_eye_center  = midpoint(Fc[:,2], Fc[:,3])
    """
    R = rodrigues_to_matrix(rvec)
    Fc = R @ face_model_3x6 + tvec.reshape(3, 1)

    right_eye_center = 0.5 * (Fc[:, 0] + Fc[:, 1])
    left_eye_center  = 0.5 * (Fc[:, 2] + Fc[:, 3])

    return left_eye_center, right_eye_center, Fc


def normalize_left_eye(
    img_bgr,
    left_eye_center,
    headpose_rvec,
    camera_matrix,
    dist_coeffs,
    eye_w=60,
    eye_h=36,
    distance_norm=600.0,
    focal_norm=960.0
):
    """
    MPIIGaze-style normalization for left eye.
    Returns:
        eye_img_eq         : (36, 60) uint8 grayscale
        headpose_norm_rvec : (3,) float64
    """
    camera_norm = np.array([
        [focal_norm, 0, eye_w / 2],
        [0, focal_norm, eye_h / 2],
        [0, 0, 1.0]
    ], dtype=np.float64)

    hR = rodrigues_to_matrix(headpose_rvec)

    forward = normalize_vector(left_eye_center)
    hRx = hR[:, 0]

    down = normalize_vector(np.cross(forward, hRx))
    right = normalize_vector(np.cross(down, forward))

    R = np.stack([right, down, forward], axis=0)

    z_scale = distance_norm / np.linalg.norm(left_eye_center)
    S = np.diag([1.0, 1.0, z_scale]).astype(np.float64)

    W = camera_norm @ S @ R @ np.linalg.inv(camera_matrix)

    img_undist = cv2.undistort(img_bgr, camera_matrix, dist_coeffs)
    eye_img = cv2.warpPerspective(img_undist, W, (eye_w, eye_h))

    eye_img_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img_eq = equalize_gray(eye_img_gray)

    headpose_norm_R = R @ hR
    headpose_norm_rvec = matrix_to_rodrigues(headpose_norm_R)

    return eye_img_eq, headpose_norm_rvec


def headpose_rvec_to_angles(rvec):
    R = rodrigues_to_matrix(rvec)
    Zv = R[:, 2]
    theta = np.arcsin(np.clip(Zv[1], -1.0, 1.0))
    phi = np.arctan2(Zv[0], Zv[2])
    return theta, phi


# =========================================================
# 2. GAZE PREPROCESSOR
# =========================================================
class GazePreprocessor:
    def __init__(
        self,
        model_path,
        face_model_mat_path,
        camera_matrix,
        dist_coeffs,
        eye_w=60,
        eye_h=36,
        distance_norm=600.0,
        focal_norm=960.0,
    ):
        self.model_path = model_path
        self.face_model_mat_path = face_model_mat_path
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

        self.eye_w = eye_w
        self.eye_h = eye_h
        self.distance_norm = distance_norm
        self.focal_norm = focal_norm

        # Best validated landmark order for your setup
        self.landmark_ids = [33, 133, 362, 263, 61, 291]

        # Load MPIIGaze 6-point face model
        mat_data = sio.loadmat(self.face_model_mat_path)
        if "model" not in mat_data:
            raise KeyError(f"'model' key not found in {self.face_model_mat_path}")

        self.face_model_3x6 = mat_data["model"].astype(np.float64)
        if self.face_model_3x6.shape != (3, 6):
            raise ValueError(
                f"Expected face model shape (3,6), got {self.face_model_3x6.shape}"
            )

        self.model_points_3d = self.face_model_3x6.T  # (6,3)

        # MediaPipe face landmarker
        base_options = python.BaseOptions(model_asset_path=self.model_path)
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
            frame_bgr: BGR webcam frame

        Returns dict with:
            success: bool
            eye_image: (36,60) uint8 or None
            head_pose: (3,) float32/float64 or None
            raw_rvec: (3,) or None
            raw_tvec: (3,) or None
            theta: float or None
            phi: float or None
            reproj_error: float or None
            error: str or None
        """
        try:
            image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self.detector.detect(mp_image)

            if len(result.face_landmarks) == 0:
                return {
                    "success": False,
                    "eye_image": None,
                    "head_pose": None,
                    "raw_rvec": None,
                    "raw_tvec": None,
                    "theta": None,
                    "phi": None,
                    "reproj_error": None,
                    "error": "no_face_detected"
                }

            face_landmarks = result.face_landmarks[0]

            image_points_2d = get_image_points_from_tasks(
                face_landmarks, w, h, self.landmark_ids
            )

            rvec, tvec = estimate_head_pose(
                image_points_2d=image_points_2d,
                model_points_3d=self.model_points_3d,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )

            if rvec is None:
                return {
                    "success": False,
                    "eye_image": None,
                    "head_pose": None,
                    "raw_rvec": None,
                    "raw_tvec": None,
                    "theta": None,
                    "phi": None,
                    "reproj_error": None,
                    "error": "pnp_failed"
                }

            _, _, mean_err = compute_reprojection_error(
                model_points_3d=self.model_points_3d,
                image_points_2d=image_points_2d,
                rvec=rvec,
                tvec=tvec,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )

            left_eye_center, right_eye_center, _ = reconstruct_eye_centers_from_real_model(
                rvec=rvec,
                tvec=tvec,
                face_model_3x6=self.face_model_3x6
            )

            norm_eye, norm_headpose = normalize_left_eye(
                img_bgr=frame_bgr,
                left_eye_center=left_eye_center,
                headpose_rvec=rvec,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                eye_w=self.eye_w,
                eye_h=self.eye_h,
                distance_norm=self.distance_norm,
                focal_norm=self.focal_norm
            )

            theta, phi = headpose_rvec_to_angles(norm_headpose)

            return {
                "success": True,
                "eye_image": norm_eye,                        # (36, 60), uint8
                "head_pose": norm_headpose.astype(np.float32),  # (3,)
                "raw_rvec": rvec.astype(np.float32),
                "raw_tvec": tvec.astype(np.float32),
                "theta": float(theta),
                "phi": float(phi),
                "reproj_error": float(mean_err),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "eye_image": None,
                "head_pose": None,
                "raw_rvec": None,
                "raw_tvec": None,
                "theta": None,
                "phi": None,
                "reproj_error": None,
                "error": str(e)
            }

    def close(self):
        if self.detector is not None:
            self.detector.close()