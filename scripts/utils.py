import os
import json
import csv
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# =========================================================
# 1. SCHEDULE LOADING
# =========================================================
def load_stimulus_schedule(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the stimulus schedule CSV.
    Required columns:
        trial_id, video_name, stimulus_side, notes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Stimulus schedule not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["trial_id", "video_name", "stimulus_side", "notes"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in stimulus schedule: {missing}")

    # normalize strings
    df["video_name"] = df["video_name"].astype(str).str.strip()
    df["stimulus_side"] = df["stimulus_side"].astype(str).str.strip().str.lower()
    df["notes"] = df["notes"].astype(str).str.strip()

    valid_sides = {"left", "right", "both", "none"}
    invalid = df.loc[~df["stimulus_side"].isin(valid_sides), "stimulus_side"].unique().tolist()
    if invalid:
        raise ValueError(f"Invalid stimulus_side values found: {invalid}")

    return df


# =========================================================
# 2. SESSION / FOLDER CREATION
# =========================================================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def create_session_structure(
    project_root: str,
    participant_id: str,
    session_id: str
) -> Dict[str, str]:
    """
    Creates the full folder structure for one collection session.

    Returns a dict of important paths.
    """
    collected_root = os.path.join(project_root, "collected_sessions")
    participant_root = os.path.join(collected_root, participant_id)
    session_root = os.path.join(participant_root, session_id)

    paths = {
        "project_root": project_root,
        "collected_root": ensure_dir(collected_root),
        "participant_root": ensure_dir(participant_root),
        "session_root": ensure_dir(session_root),

        "gaze_root": ensure_dir(os.path.join(session_root, "gaze")),
        "gaze_eye_dir": ensure_dir(os.path.join(session_root, "gaze", "eye_images")),
        "gaze_pose_dir": ensure_dir(os.path.join(session_root, "gaze", "head_pose")),

        "affect_root": ensure_dir(os.path.join(session_root, "affect")),
        "affect_face_dir": ensure_dir(os.path.join(session_root, "affect", "face_crops")),
        "affect_landmark_dir": ensure_dir(os.path.join(session_root, "affect", "landmarks")),

        "logs_root": ensure_dir(os.path.join(session_root, "logs")),
        "stimulus_log_csv": os.path.join(session_root, "logs", "stimulus_log.csv"),
        "sync_log_csv": os.path.join(session_root, "logs", "synchronized_samples.csv"),

        "session_info_json": os.path.join(session_root, "session_info.json"),
    }

    return paths


# =========================================================
# 3. SESSION INFO
# =========================================================
def write_session_info(
    session_info_path: str,
    participant_id: str,
    session_id: str,
    fps: float,
    num_videos: int,
    notes: str = ""
) -> None:
    payload = {
        "participant_id": participant_id,
        "session_id": session_id,
        "fps": fps,
        "num_videos": num_videos,
        "notes": notes,
    }

    with open(session_info_path, "w") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# 4. CSV INITIALIZATION
# =========================================================
def init_stimulus_log(csv_path: str) -> None:
    """
    One row per played video / trial.
    """
    header = [
        "trial_id",
        "video_name",
        "stimulus_side",
        "notes",
        "video_start_sec",
        "video_end_sec",
        "duration_sec",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def init_sync_log(csv_path: str) -> None:
    """
    One row per processed webcam frame.
    Stores synchronized raw inputs only, not model predictions.
    """
    header = [
        "frame_id",
        "timestamp_sec",

        "trial_id",
        "video_name",
        "stimulus_side",
        "notes",
        "video_time_sec",

        "gaze_eye_path",
        "gaze_head_pose_path",
        "gaze_reproj_error",
        "gaze_valid",

        "affect_face_path",
        "affect_landmark_path",
        "affect_valid",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


# =========================================================
# 5. CSV APPEND HELPERS
# =========================================================
def append_stimulus_log(
    csv_path: str,
    trial_id,
    video_name: str,
    stimulus_side: str,
    notes: str,
    video_start_sec: float,
    video_end_sec: float,
) -> None:
    duration_sec = float(video_end_sec - video_start_sec)

    row = [
        trial_id,
        video_name,
        stimulus_side,
        notes,
        float(video_start_sec),
        float(video_end_sec),
        duration_sec,
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def append_sync_log(
    csv_path: str,
    frame_id: str,
    timestamp_sec: float,

    trial_id,
    video_name: str,
    stimulus_side: str,
    notes: str,
    video_time_sec: float,

    gaze_eye_path: str,
    gaze_head_pose_path: str,
    gaze_reproj_error,
    gaze_valid: bool,

    affect_face_path: str,
    affect_landmark_path: str,
    affect_valid: bool,
) -> None:
    row = [
        frame_id,
        float(timestamp_sec),

        trial_id,
        video_name,
        stimulus_side,
        notes,
        float(video_time_sec),

        gaze_eye_path,
        gaze_head_pose_path,
        "" if gaze_reproj_error is None else float(gaze_reproj_error),
        int(gaze_valid),

        affect_face_path,
        affect_landmark_path,
        int(affect_valid),
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================================================
# 6. FRAME IDS + FILENAMES
# =========================================================
def make_frame_id(frame_index: int) -> str:
    return f"frame_{frame_index:06d}"


def build_gaze_paths(paths: Dict[str, str], frame_id: str) -> Tuple[str, str]:
    eye_path = os.path.join(paths["gaze_eye_dir"], f"{frame_id}.png")
    pose_path = os.path.join(paths["gaze_pose_dir"], f"{frame_id}.npy")
    return eye_path, pose_path


def build_affect_paths(paths: Dict[str, str], frame_id: str) -> Tuple[str, str]:
    face_path = os.path.join(paths["affect_face_dir"], f"{frame_id}.jpg")
    landmark_path = os.path.join(paths["affect_landmark_dir"], f"{frame_id}.npy")
    return face_path, landmark_path


# =========================================================
# 7. TIME HELPERS
# =========================================================
def relative_time_sec(now_time: float, experiment_start_time: float) -> float:
    return float(now_time - experiment_start_time)


def video_time_sec(now_time: float, video_start_time: float) -> float:
    return float(now_time - video_start_time)


# =========================================================
# 8. OPTIONAL SMALL HELPERS
# =========================================================
def get_video_path(video_data_dir: str, video_name: str) -> str:
    path = os.path.join(video_data_dir, video_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def safe_bool(x) -> bool:
    return bool(x)


def count_trials(schedule_df: pd.DataFrame) -> int:
    return int(len(schedule_df))


# =========================================================
# 9. QUICK TEST
# =========================================================
if __name__ == "__main__":
    # Example local sanity test
    project_root = "."
    participant_id = "P01"
    session_id = "session_001"

    paths = create_session_structure(project_root, participant_id, session_id)
    init_stimulus_log(paths["stimulus_log_csv"])
    init_sync_log(paths["sync_log_csv"])
    write_session_info(
        paths["session_info_json"],
        participant_id=participant_id,
        session_id=session_id,
        fps=5.0,
        num_videos=10,
        notes="test session"
    )

    print("Created session structure successfully.")
    for k, v in paths.items():
        print(f"{k}: {v}")