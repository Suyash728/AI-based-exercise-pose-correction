import numpy as np

def calculate_angle(point_a, point_b, point_c):
    """
    Calculates angle at point_b (the vertex/joint)
    
    Example: knee angle = calculate_angle(hip, knee, ankle)
    
    Returns angle in degrees (0-180)
    """
    a = np.array(point_a)
    b = np.array(point_b)  # this is the joint we're measuring
    c = np.array(point_c)
    
    # Vectors from vertex to each end
    ba = a - b
    bc = c - b
    
    # Dot product formula for angle
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    
    return round(angle, 1)

def extract_exercise_angles(landmarks):
    """
    Given landmarks dict, extract all useful angles
    
    MediaPipe landmark indices (memorize these):
    11=left shoulder, 12=right shoulder
    13=left elbow,    14=right elbow
    15=left wrist,    16=right wrist
    23=left hip,      24=right hip
    25=left knee,     26=right knee
    27=left ankle,    28=right ankle
    """
    if not landmarks:
        return {}
    
    L = landmarks  # shorthand
    
    angles = {}
    
    try:
        # Knee angles
        angles['left_knee']  = calculate_angle(L[23], L[25], L[27])
        angles['right_knee'] = calculate_angle(L[24], L[26], L[28])
        
        # Hip angles
        angles['left_hip']   = calculate_angle(L[11], L[23], L[25])
        angles['right_hip']  = calculate_angle(L[12], L[24], L[26])
        
        # Elbow angles
        angles['left_elbow']  = calculate_angle(L[11], L[13], L[15])
        angles['right_elbow'] = calculate_angle(L[12], L[14], L[16])
        
        # Shoulder angles
        angles['left_shoulder']  = calculate_angle(L[13], L[11], L[23])
        angles['right_shoulder'] = calculate_angle(L[14], L[12], L[24])
        
    except KeyError:
        pass  # some landmarks not visible, skip
    
    return angles