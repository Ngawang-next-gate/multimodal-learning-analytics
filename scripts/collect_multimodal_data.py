import os
import time
import cv2
import numpy as np
import pygame

from utils import (
    load_stimulus_schedule,
    create_session_structure,
    write_session_info,
    init_stimulus_log,
    init_sync_log,
    append_stimulus_log,
    append_sync_log,
    make_frame_id,
    build_gaze_paths,
    build_affect_paths,
    relative_time_sec,
    video_time_sec,
    get_video_path,
    count_trials,
)

from gaze_preprocessing import GazePreprocessor
from affect_preprocessing import AffectPreprocessor


# =========================================================
# 1. MANUAL SESSION SETTINGS
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PARTICIPANT_ID = "PXX"
SESSION_ID = "session_001"
SESSION_NOTES = "multimodal collection run with fullscreen video + audio"

TARGET_FPS = 5.0
PROCESS_INTERVAL = 1.0 / TARGET_FPS

# No webcam preview for participant
SHOW_DEBUG_WINDOWS = False

# Camera calibration for gaze branch
camera_matrix = np.array([
    [951.8260391,   0.0,        633.62181582],
    [0.0,         943.73803651, 348.74927997],
    [0.0,           0.0,          1.0]
], dtype=np.float64)

dist_coeffs = np.array([[0.06256892, -0.19577242, -0.00046166, 0.00223187, 0.32183045]], dtype=np.float64)

# Paths
SCHEDULE_CSV = os.path.join(PROJECT_ROOT, "configs", "stimulus_schedule.csv")
VIDEO_DATA_DIR = os.path.join(PROJECT_ROOT, "video_data")
AUDIO_DATA_DIR = os.path.join(PROJECT_ROOT, "audio_data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "face_landmarker.task")
FACE_MODEL_MAT_PATH = os.path.join(PROJECT_ROOT, "assets", "6_points_based_face_model.mat")


# =========================================================
# 2. HELPERS
# =========================================================
def get_audio_path(audio_dir: str, video_name: str) -> str:
    audio_name = os.path.splitext(video_name)[0] + ".wav"
    audio_path = os.path.join(audio_dir, audio_name)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    return audio_path


def setup_fullscreen_window(window_name="stimulus_fullscreen"):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return window_name


def fit_frame_to_screen(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# =========================================================
# 3. SETUP
# =========================================================
schedule_df = load_stimulus_schedule(SCHEDULE_CSV)
num_videos = count_trials(schedule_df)

paths = create_session_structure(
    project_root=PROJECT_ROOT,
    participant_id=PARTICIPANT_ID,
    session_id=SESSION_ID
)

write_session_info(
    session_info_path=paths["session_info_json"],
    participant_id=PARTICIPANT_ID,
    session_id=SESSION_ID,
    fps=TARGET_FPS,
    num_videos=num_videos,
    notes=SESSION_NOTES
)

init_stimulus_log(paths["stimulus_log_csv"])
init_sync_log(paths["sync_log_csv"])

print("Session structure created:")
print(paths["session_root"])
print()

# preprocessors
gaze_preprocessor = GazePreprocessor(
    model_path=MODEL_PATH,
    face_model_mat_path=FACE_MODEL_MAT_PATH,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    eye_w=60,
    eye_h=36,
    distance_norm=600.0,
    focal_norm=960.0,
)

affect_preprocessor = AffectPreprocessor(
    model_path=MODEL_PATH,
    output_face_size=(224, 224),
    margin=0.20,
)

# webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# audio
pygame.mixer.init()

# fullscreen stimulus window
window_name = setup_fullscreen_window("stimulus_fullscreen")

# screen size based on current display
screen_w = 1920
screen_h = 1080

print(f"Participant: {PARTICIPANT_ID}")
print(f"Session    : {SESSION_ID}")
print(f"Videos     : {num_videos}")
print(f"Target FPS : {TARGET_FPS}")
print("Fullscreen stimulus will start immediately with audio.")
print("Press 'q' to stop early.")
print()

experiment_start_time = time.perf_counter()
global_frame_index = 0
last_process_time = 0.0


# =========================================================
# 4. MAIN EXPERIMENT LOOP
# =========================================================
try:
    for _, trial_row in schedule_df.iterrows():
        trial_id = int(trial_row["trial_id"])
        video_name = str(trial_row["video_name"])
        stimulus_side = str(trial_row["stimulus_side"])
        notes = str(trial_row["notes"])

        video_path = get_video_path(VIDEO_DATA_DIR, video_name)
        audio_path = get_audio_path(AUDIO_DATA_DIR, video_name)

        print(f"Starting trial {trial_id}: {video_name} | side={stimulus_side} | notes={notes}")

        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            print(f"[WARNING] Could not open video: {video_path}")
            continue

        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        if video_fps is None or video_fps <= 0:
            video_fps = 25.0

        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None

        # preload first frame timing
        current_video_frame_idx = 0
        next_frame_due = 0.0
        current_stimulus_frame = None

        # start audio + timing together
        trial_start_abs = time.perf_counter()
        video_start_rel = relative_time_sec(trial_start_abs, experiment_start_time)

        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

        # read the first frame immediately
        ret_video, first_frame = video_cap.read()
        if not ret_video:
            print(f"[WARNING] Could not read first frame of {video_name}")
            video_cap.release()
            pygame.mixer.music.stop()
            continue

        current_stimulus_frame = first_frame
        current_video_frame_idx = 1
        next_frame_due = current_video_frame_idx / video_fps

        while True:
            now_abs = time.perf_counter()
            elapsed_in_video = now_abs - trial_start_abs
            now_rel = relative_time_sec(now_abs, experiment_start_time)

            # end condition
            if total_frames is not None and current_video_frame_idx >= total_frames and not pygame.mixer.music.get_busy():
                break

            # advance video frames based on elapsed video time
            while elapsed_in_video >= next_frame_due:
                ret_video, frame = video_cap.read()
                if not ret_video:
                    break
                current_stimulus_frame = frame
                current_video_frame_idx += 1
                next_frame_due = current_video_frame_idx / video_fps

            # show fullscreen video frame
            if current_stimulus_frame is not None:
                fullscreen_frame = fit_frame_to_screen(current_stimulus_frame, screen_w, screen_h)
                cv2.imshow(window_name, fullscreen_frame)

            # collect webcam frame
            ret_cam, webcam_frame = cap.read()
            if ret_cam and (now_abs - last_process_time >= PROCESS_INTERVAL):
                last_process_time = now_abs
                global_frame_index += 1
                frame_id = make_frame_id(global_frame_index)

                current_video_time = elapsed_in_video

                # same webcam frame for both branches
                gaze_result = gaze_preprocessor.process_frame(webcam_frame)
                affect_result = affect_preprocessor.process_frame(webcam_frame)

                gaze_valid = bool(gaze_result["success"])
                affect_valid = bool(affect_result["success"])

                gaze_eye_path = ""
                gaze_pose_path = ""
                affect_face_path = ""
                affect_landmark_path = ""
                gaze_reproj_error = None

                # save gaze outputs
                if gaze_valid:
                    gaze_eye_path, gaze_pose_path = build_gaze_paths(paths, frame_id)
                    cv2.imwrite(gaze_eye_path, gaze_result["eye_image"])
                    np.save(gaze_pose_path, gaze_result["head_pose"])
                    gaze_reproj_error = gaze_result["reproj_error"]

                # save affect outputs
                if affect_valid:
                    affect_face_path, affect_landmark_path = build_affect_paths(paths, frame_id)
                    face_bgr = cv2.cvtColor(affect_result["face_crop"], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(affect_face_path, face_bgr)
                    np.save(affect_landmark_path, affect_result["landmarks"])

                # log synchronized row
                append_sync_log(
                    csv_path=paths["sync_log_csv"],
                    frame_id=frame_id,
                    timestamp_sec=now_rel,

                    trial_id=trial_id,
                    video_name=video_name,
                    stimulus_side=stimulus_side,
                    notes=notes,
                    video_time_sec=current_video_time,

                    gaze_eye_path=gaze_eye_path,
                    gaze_head_pose_path=gaze_pose_path,
                    gaze_reproj_error=gaze_reproj_error,
                    gaze_valid=gaze_valid,

                    affect_face_path=affect_face_path,
                    affect_landmark_path=affect_landmark_path,
                    affect_valid=affect_valid,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt

            # if video ended but audio still running, keep showing last frame until audio ends
            if total_frames is not None and current_video_frame_idx >= total_frames and not pygame.mixer.music.get_busy():
                break

            time.sleep(0.001)

        # stop audio explicitly
        pygame.mixer.music.stop()

        trial_end_abs = time.perf_counter()
        video_end_rel = relative_time_sec(trial_end_abs, experiment_start_time)

        append_stimulus_log(
            csv_path=paths["stimulus_log_csv"],
            trial_id=trial_id,
            video_name=video_name,
            stimulus_side=stimulus_side,
            notes=notes,
            video_start_sec=video_start_rel,
            video_end_sec=video_end_rel,
        )

        video_cap.release()
        print(f"Finished trial {trial_id}: {video_name}")

        time.sleep(0.2)

    print("\nExperiment complete.")

except KeyboardInterrupt:
    print("\nStopped early by user.")

finally:
    cap.release()
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    gaze_preprocessor.close()
    affect_preprocessor.close()
    cv2.destroyAllWindows()

    print("\nSession saved to:")
    print(paths["session_root"])
    print("Stimulus log:", paths["stimulus_log_csv"])
    print("Sync log    :", paths["sync_log_csv"])