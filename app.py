import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import tempfile
import os

from pose_extractor import get_landmarks, draw_pose
from angle_calculator import extract_exercise_angles
from form_rules import analyze_video_form

# ── Load Model ───────────────────────────────────────────
@st.cache_resource  # load only once
def load_model():
    model = tf.keras.models.load_model('models/exercise_classifier.h5')
    with open('models/class_labels.json') as f:
        labels = json.load(f)
    return model, labels

model, labels = load_model()

# ── UI Layout ────────────────────────────────────────────
st.title("🏋️ AI Exercise Form Analyzer")
st.write("Upload an exercise video or image to get form feedback.")

upload_type = st.radio("Choose input type:", ["Image", "Video"])
uploaded_file = st.file_uploader(
    "Upload here",
    type=["jpg", "png", "mp4", "avi"]
)

if uploaded_file:
    
    if upload_type == "Image":
        # Read image
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Predict exercise type
        img_resized = cv2.resize(image, (224, 224)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        predictions = model.predict(img_input)
        class_idx = str(np.argmax(predictions))
        exercise_name = labels[class_idx]
        confidence = float(np.max(predictions)) * 100
        
        st.subheader(f"Detected Exercise: **{exercise_name}** ({confidence:.1f}% confidence)")
        
        # Get pose landmarks
        landmarks = get_landmarks(image)
        
        if landmarks:
            # Draw skeleton
            annotated = draw_pose(image.copy(), landmarks)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Pose Detection", use_column_width=True)
            
            # Calculate angles
            angles = extract_exercise_angles(landmarks)
            
            # Analyze form (Treating the single frame as both min and max)
            verdict, errors = analyze_video_form(exercise_name, min_angles=angles, max_angles=angles)
            
            # Show results
            st.subheader("Form Analysis")
            if "Correct" in verdict:
                st.success(verdict)
            else:
                st.error(verdict)
                for error in errors:
                    st.warning(error)
            
            # Show measured angles
            with st.expander("📐 Measured Joint Angles"):
                for name, angle in angles.items():
                    st.write(f"**{name.replace('_', ' ').title()}:** {angle}°")
        
        else:
            st.warning("No person detected in image. Try a clearer photo.")
    
    elif upload_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 1. Sample 8 frames evenly across the video
        num_samples = 8
        step = max(1, total_frames // num_samples)
        
        raw_predictions = []
        all_angles_timeline = []
        best_annotated_frame = None
        
        progress_text = st.empty()
        
        for i in range(num_samples):
            progress_text.text(f"Analyzing video frame {i+1}/{num_samples}...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret: continue
            
            # Predict Exercise for this frame
            img_resized = cv2.resize(frame, (224, 224)) / 255.0
            predictions = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)
            raw_predictions.append(predictions[0])
            
            # Extract Angles for this frame
            landmarks = get_landmarks(frame)
            if landmarks:
                angles = extract_exercise_angles(landmarks)
                all_angles_timeline.append(angles)
                best_annotated_frame = draw_pose(frame.copy(), landmarks) # Save last good frame for UI
                
        cap.release()
        os.unlink(tfile.name)
        progress_text.empty() # clear progress text
        
        # 2. Soft-Voting Ensemble Classification
        avg_predictions = np.mean(raw_predictions, axis=0)
        class_idx = str(np.argmax(avg_predictions))
        most_common_exercise = labels[class_idx]
        confidence = np.max(avg_predictions) * 100
        
        st.subheader(f"Detected Exercise: **{most_common_exercise}** ({confidence:.1f}% Match)")
        
        if best_annotated_frame is not None:
            st.image(cv2.cvtColor(best_annotated_frame, cv2.COLOR_BGR2RGB))
            
            # 3. Calculate Min and Max ranges across the whole movement
            max_angles = {}
            min_angles = {}
            for angle_dict in all_angles_timeline:
                for joint, angle in angle_dict.items():
                    max_angles[joint] = max(max_angles.get(joint, 0), angle)
                    min_angles[joint] = min(min_angles.get(joint, 180), angle)
            
            # ---------------------------------------------------------
            # EXPERT ML OVERRIDE: Prevent CNN "Domain Shift" hallucinations 
            # using kinematic physics rules.
            # ---------------------------------------------------------
            if most_common_exercise in ['WallPushups', 'PushUps']:
                deep_squat = min_angles.get('left_hip', 180) < 100 or min_angles.get('right_hip', 180) < 100
                hands_overhead = max_angles.get('left_shoulder', 0) > 150 or max_angles.get('right_shoulder', 0) > 150
                
                if deep_squat and hands_overhead:
                    most_common_exercise = 'CleanAndJerk'
                    st.info("🧠 **Hybrid AI Override:** Visual CNN got confused by the background, but Kinematic Physics detected a Clean & Jerk based on deep hip flexion and overhead reach.")
            # ---------------------------------------------------------

            # 4. Apply dynamic rules
            from form_rules import analyze_video_form
            verdict, errors = analyze_video_form(most_common_exercise, min_angles, max_angles)
            
            if "Correct" in verdict:
                st.success(verdict)
            else:
                st.error(verdict)
                for e in errors:
                    st.warning(e)
                    
            with st.expander("📊 Range of Motion Data"):
                st.write("Tracks the peak extension (Max) and deepest flexion (Min) during the rep.")
                for joint in max_angles.keys():
                    st.write(f"**{joint.replace('_', ' ').title()}**: Min {min_angles[joint]}° | Max {max_angles[joint]}°")
