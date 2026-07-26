## Overview

This repository contains the complete multimodal data collection and preprocessing framework used for our study on learner engagement in unconstrained digital learning environments.

Unlike traditional engagement monitoring systems that rely on a single behavioral cue, the proposed framework synchronizes multiple behavioral modalities including

- Spatial gaze behavior
- Continuous affective states
- Physiological fatigue indicators

using only a standard webcam.

The collected multimodal streams are synchronized frame-by-frame to support subsequent behavioral analysis.

---

## Framework

The framework consists of two parallel processing modules.

### Gaze Module

The gaze module performs

- Face detection
- Six-point head pose estimation
- Left-eye extraction
- Eye normalization
- ROI image generation
- Camera calibration

using the six-point face model proposed by Sugano et al.

Outputs include

- normalized left-eye images
- head pose
- gaze preprocessing metadata

### Affect Module

The affect module performs

- Face detection
- MediaPipe Face Mesh
- Facial landmark extraction
- Landmark normalization
- Facial crop generation

The processed facial images and normalized landmarks are used for continuous valence-arousal prediction.

### Data Collection

The data collection framework

- Presents educational videos
- Synchronizes webcam capture
- Records timestamps
- Saves gaze and affect preprocessing outputs
- Logs stimulus presentation
- Generates session metadata

All modalities are synchronized using timestamps for later multimodal analysis.

## Repository Structure

```
project/

│
├── assets/
│   ├── 6_points_based_face_model.mat
│   └── face_landmarker.task
│
├── scripts/
│   ├── collect_multimodal_data.py
│   ├── gaze_preprocessing.py
│   ├── affect_preprocessing.py
│   ├── extract_audio.py
│   └── utils.py
│
├── README.md
└── requirements.
```

## Assets

Two assets are required before running the framework.

### 1. Six-point Face Model

```
assets/
└── 6_points_based_face_model.mat
```

Used for

- head pose estimation
- gaze normalization

---

### 2. MediaPipe Face Landmarker

```
assets/
└── face_landmarker.task
```

Used for

- facial landmark detection
- facial region extraction

## Dataset Availability

The datasets used in this work include

- MPIIGaze
- AFEW-VA

## Running the Framework

Start the multimodal data collection pipeline

```bash

python scripts/collect_multimodal_data.py

```

## Experimental Dataset

The real-world deployment dataset collected for this study is **not publicly released**.

This includes

- participant videos
- participant audio
- educational CNN tutorial videos
- synchronized multimodal recordings

The dataset contains identifiable participant information and educational materials that cannot be redistributed because of ethical, privacy, and copyright considerations.

Only anonymized experimental logs and the preprocessing framework are released.


## Reproducibility

This repository provides

- complete preprocessing pipeline
- experimental collection framework
- synchronization pipeline
- preprocessing assets

allowing researchers to reproduce the complete multimodal data acquisition workflow using their own educational content.

