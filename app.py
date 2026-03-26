import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import tempfile
import os

from pose_extractor import get_landmarks, draw_pose
from angle_calculator import extract_exercise_angles
from form_rules import analyze_form

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
            
            # Analyze form
            verdict, errors = analyze_form(exercise_name, angles)
            
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
        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Analyze middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Same pipeline as image
            img_resized = cv2.resize(frame, (224, 224)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0)
            predictions = model.predict(img_input)
            class_idx = str(np.argmax(predictions))
            exercise_name = labels[class_idx]
            
            landmarks = get_landmarks(frame)
            annotated = draw_pose(frame.copy(), landmarks)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            
            st.subheader(f"Exercise: **{exercise_name}**")
            
            if landmarks:
                angles = extract_exercise_angles(landmarks)
                verdict, errors = analyze_form(exercise_name, angles)
                
                if "Correct" in verdict:
                    st.success(verdict)
                else:
                    st.error(verdict)
                    for e in errors:
                        st.warning(e)
        
        os.unlink(tfile.name)  # cleanup temp  file