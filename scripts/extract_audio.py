import os
# CHANGE 1: Dropped '.editor' for MoviePy 2.0+ compatibility
from moviepy import VideoFileClip

# =========================================================
# PATHS
# =========================================================
VIDEO_DIR = "/Users/macbook/Desktop/student_engagement_collection/video_data"
AUDIO_DIR = "/Users/macbook/Desktop/student_engagement_collection/audio_data"

os.makedirs(AUDIO_DIR, exist_ok=True)

# =========================================================
# PROCESS ALL VIDEOS
# =========================================================
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print(f"Found {len(video_files)} videos")

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)

    # Replace .mp4 -> .wav
    audio_filename = os.path.splitext(video_file)[0] + ".wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)

    print(f"\nProcessing: {video_file}")

    try:
        # CHANGE 2: Added 'with' statement to prevent RAM memory leaks
        with VideoFileClip(video_path) as video:
            if video.audio is None:
                print("⚠️ No audio track found, skipping...")
                continue

            video.audio.write_audiofile(
                audio_path,
                fps=44100,
                codec='pcm_s16le',   # high-quality WAV
                logger=None          # Suppresses the massive progress bars in terminal
            )

        print(f"Saved: {audio_filename}")

    except Exception as e:
        print(f" Error processing {video_file}: {e}")

print("\n Audio extraction complete!")