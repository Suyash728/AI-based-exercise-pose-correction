import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize ONCE globally to save massive memory and time
pose_estimator = mp_pose.Pose(
    static_image_mode=False, # False makes it faster for video sequences
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    landmarks = {}
    h, w = image.shape[:2]
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        landmarks[idx] = (int(landmark.x * w), int(landmark.y * h))
    return landmarks

def draw_pose(image, landmarks):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image