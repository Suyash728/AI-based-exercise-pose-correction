# 🏋️ AI-Based Exercise Pose Correction
**AesCode Nexus Hackathon · Round 1 Submission** *Automated biomechanical analysis and injury prevention using temporal pose estimation and hybrid machine learning.*

---

## 📖 Problem Context
Incorrect exercise form is a leading contributor to musculoskeletal injuries in both fitness and rehabilitation settings. While visual assessment by professional trainers or physiotherapists is effective, it is not scalable or accessible to the average individual exercising at home. 

Current AI solutions often rely on static image analysis, which fails to capture the continuous, 3D nature of human motion. **This project bridges that gap by introducing a temporal, multi-frame video analysis pipeline** that measures the absolute kinematic range of motion to detect joint misalignment, compensatory movement, and incorrect biomechanics.

---

## 🚀 Key Innovations & Methodology

Unlike standard classifiers, this system employs a **Hybrid AI Architecture** to ensure robust performance across different environments:

1. **Temporal Kinematic Sampling:** Instead of analyzing a single static frame, the system samples 8 keyframes evenly across a video sequence. It tracks the maximum extension and minimum flexion of specific joints to grade the full rep, rather than an isolated moment.
2. **Soft-Voting Classification Ensemble:** The system averages the raw prediction probabilities across the temporal sequence, preventing single-frame motion blur from causing a misclassification.
3. **Hybrid AI Physics Override:** To counter "Domain Shift" (where the CNN misclassifies an exercise due to unfamiliar backgrounds), the system cross-references the CNN output with MediaPipe's physical coordinates. If the spatial math (e.g., deep hip flexion + overhead reach) contradicts the CNN, the physics engine intelligently overrides the prediction.
4. **Tolerance Epsilon (Human Grace Margin):** To account for 2D camera foreshortening and clothing obstruction, the rules engine applies a 15-degree mathematical tolerance to joint angle targets, preventing the system from failing clinically acceptable movements.

---

## 📊 Dataset Overview
**IMPORTANT:** This model was trained *exclusively* on the filtered subset of the provided dataset to prevent overfitting and satisfy competition constraints.

* **Dataset Name:** Filtered UCF-101 Subset (Provided by Organizers)
* **Classes Used (5):** `CleanAndJerk`, `JumpingJack`, `PushUps`, `WallPushups`, `HandstandPushups`
* **Total Image Count:** 512 images
* **Total Dataset Size:** 10.8 MB
* **Image Dimensions:** 224 x 224 px
* **Format:** Extracted JPEG frames from AVI videos

---

## 🧠 System Architecture

1. **Input Layer:** User uploads an MP4/AVI video or JPG/PNG image via the Streamlit UI.
2. **Feature Extraction (CNN):** A locally fine-tuned **MobileNetV2** (TensorFlow) classifies the exercise type.
3. **Pose Estimation:** **MediaPipe** extracts 33 3D skeletal keypoints per frame.
4. **Biomechanical Math:** Vector dot-product calculations measure precise joint angles (knees, hips, elbows, shoulders).
5. **Rules Engine:** The system evaluates the kinematic extremes (Min/Max angles) against hardcoded sports-science rules, returning a real-time `Correct` or `Incorrect` verdict with specific error feedback.

---

## 💻 Local Setup & Installation

This project is optimized for a local Windows environment with an NVIDIA GPU (RTX 3050).

### Prerequisites
* Python 3.9 (Strict requirement for TensorFlow 2.10 native Windows GPU support)
* CUDA Toolkit 11.2 & cuDNN 8.1
* NVIDIA GPU

### Installation
1. Clone the repository and navigate to the project folder.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate

###Install the specific dependencies:
pip install -r requirements.txt
(Note: The environment specifically pairs tensorflow==2.10.0 with protobuf==3.20.3 to support MediaPipe 0.10.5).

###Running the App
To bypass Windows Protocol Buffer descriptor conflicts, the app explicitly forces the Python implementation. 
Start the application by running:
streamlit run app.py

📁 Project Structure:
AI-based-exercise-pose-correction/
│
├── models/
│   ├── exercise_classifier.h5    # Trained MobileNetV2 weights
│   └── class_labels.json         # Exercise index mapping
│
├── app.py                        # Streamlit web interface & temporal engine
├── train_model.py                # Local GPU training script (Batch size 16)
├── pose_extractor.py             # MediaPipe initialization and skeleton mapping
├── angle_calculator.py           # Vector dot-product math for joint angles
├── form_rules.py                 # Rules engine with Epsilon Tolerance logic
└── requirements.txt              # Strict version-locked dependencies

👥 Team
Team Name: Team Error404

Institution: [YOUR COLLEGE/UNIVERSITY]

Team ID: [YOUR TEAM ID]