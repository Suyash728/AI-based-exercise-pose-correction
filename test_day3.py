import cv2
from pose_extractor import get_landmarks, draw_pose
from angle_calculator import extract_exercise_angles

# Open your laptop webcam
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # 1. Get the skeleton points
    landmarks = get_landmarks(frame)
    
    if landmarks:
        # 2. Draw the skeleton on the frame
        frame = draw_pose(frame, landmarks)
        
        # 3. Calculate the angles
        angles = extract_exercise_angles(landmarks)
        
        # Print just one angle to the terminal to verify it works
        if 'right_elbow' in angles:
            cv2.putText(frame, f"R Elbow: {angles['right_elbow']}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow('Day 3 Pose Test', frame)
    
    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()