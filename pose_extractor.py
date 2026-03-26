import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_landmarks(image):
    """
    Takes an image (numpy array),
    returns 33 landmark positions as a dict
    """
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    )
    
    # MediaPipe needs RGB, OpenCV gives BGR — convert
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None  # No person detected
    
    # Extract all 33 landmarks as (x, y) pairs
    landmarks = {}
    h, w = image.shape[:2]
    
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        landmarks[idx] = (
            int(landmark.x * w),   # convert from 0-1 to pixels
            int(landmark.y * h)
        )
    
    return landmarks

def draw_pose(image, landmarks):
    """Draw skeleton on image for visualization"""
    pose = mp_pose.Pose()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    return image