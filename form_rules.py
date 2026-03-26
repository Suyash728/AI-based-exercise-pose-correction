# Each exercise has a dict of angle rules
# Format: 'angle_name': (min, max, error_message)
# If angle is OUTSIDE (min, max) → flag as error

EXERCISE_RULES = {
    
    'Lunges': {
        'front_knee':  (80, 100, "Front knee should be ~90° — don't let it cave inward"),
        'back_knee':   (80, 100, "Back knee too straight or over-bent"),
        'torso_lean':  (75, 105, "Keep torso upright — don't lean too far forward"),
    },
    
    'WallPushups': {
        'left_elbow':  (80, 110, "Elbow angle off — keep arms at 90° at bottom"),
        'right_elbow': (80, 110, "Elbow angle off — keep arms at 90° at bottom"),
        'left_shoulder': (70, 110, "Shoulder position incorrect"),
        'right_shoulder': (70, 110, "Shoulder position incorrect"),
    },
    
    'PullUps': {
        'left_elbow':  (140, 180, "Arms not fully extended at bottom"),
        'right_elbow': (140, 180, "Arms not fully extended at bottom"),
        'left_shoulder': (150, 180, "Shoulder engagement off"),
        'right_shoulder': (150, 180, "Shoulder engagement off"),
    },
    
    'JumpingJack': {
        'left_shoulder':  (60, 180, "Arms not raising fully"),
        'right_shoulder': (60, 180, "Arms not raising fully"),
        'left_hip':       (20, 60,  "Legs not spreading wide enough"),
        'right_hip':      (20, 60,  "Legs not spreading wide enough"),
    },
    
    'HandstandPushups': {
        'left_elbow':  (80, 110, "Elbow angle off for handstand pushup"),
        'right_elbow': (80, 110, "Elbow angle off for handstand pushup"),
    },
    
    'Rowing': {
        'left_elbow':  (60, 120, "Pulling angle incorrect"),
        'right_elbow': (60, 120, "Pulling angle incorrect"),
        'left_hip':    (100, 140, "Hip hinge angle off"),
    },
    
    # Add more as you go...
}

def analyze_form(exercise_name, angles):
    """
    Takes exercise name + measured angles
    Returns: verdict (Correct/Incorrect) + list of errors
    """
    errors = []
    
    # Check if we have rules for this exercise
    if exercise_name not in EXERCISE_RULES:
        return "Unknown", ["No form rules defined for this exercise yet"]
    
    rules = EXERCISE_RULES[exercise_name]
    
    for angle_name, (min_val, max_val, error_msg) in rules.items():
        if angle_name in angles:
            measured = angles[angle_name]
            if not (min_val <= measured <= max_val):
                errors.append(f"⚠️ {error_msg} (measured: {measured}°, ideal: {min_val}-{max_val}°)")
    
    verdict = "✅ Correct Form" if not errors else "❌ Incorrect Form"
    return verdict, errors