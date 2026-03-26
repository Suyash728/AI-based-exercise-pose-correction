# Rule format: 'angle_name': ('check_type', target_value, error_message)
# 'max' = the joint must reach AT LEAST this angle (extension)
# 'min' = the joint must bend to AT MOST this angle (flexion)

EXERCISE_RULES = {
    'CleanAndJerk': {
        'left_elbow':  ('max', 150, "Arms must fully lock out overhead during the jerk"),
        'right_elbow': ('max', 150, "Arms must fully lock out overhead during the jerk"),
        'left_knee':   ('min', 110, "Did not squat deep enough during the clean catch"),
        'right_knee':  ('min', 110, "Did not squat deep enough during the clean catch"),
    },
    'PushUps': {
        'left_elbow':  ('min', 90, "Go lower! Elbows must bend to at least 90 degrees"),
        'right_elbow': ('min', 90, "Go lower! Elbows must bend to at least 90 degrees"),
    },
    'WallPushups': {
        'left_elbow':  ('min', 100, "Bend elbows more to bring chest closer to the wall"),
        'right_elbow': ('min', 100, "Bend elbows more to bring chest closer to the wall"),
    },
    'HandstandPushups': {
        'left_elbow':  ('min', 90, "Descend until elbows are at 90 degrees"),
        'right_elbow': ('min', 90, "Descend until elbows are at 90 degrees"),
        'left_shoulder': ('max', 160, "Push all the way up to full shoulder extension"),
    },
    'JumpingJack': {
        'left_shoulder': ('max', 150, "Arms not raising high enough overhead"),
        'left_hip':      ('max', 35, "Legs not spreading wide enough"),
    }
}

def analyze_video_form(exercise_name, min_angles, max_angles, tolerance=20):
    """
    Analyzes the form using the extracted min/max angles, 
    applying a mathematical tolerance to account for 2D camera distortion.
    """
    errors = []
    if exercise_name not in EXERCISE_RULES:
        return "Unknown", ["No temporal rules defined for this exercise."]
    
    rules = EXERCISE_RULES[exercise_name]
    
    for angle_name, (check_type, target, error_msg) in rules.items():
        if angle_name in max_angles and angle_name in min_angles:
            if check_type == 'max':
                # e.g., Target is 150. With 15 deg tolerance, we accept 135 and above.
                if max_angles[angle_name] < (target - tolerance):
                    errors.append(f"⚠️ {error_msg} (Best: {max_angles[angle_name]}°, Target: ~{target}°)")
            elif check_type == 'min':
                # e.g., Target is 90. With 15 deg tolerance, we accept 105 and below.
                if min_angles[angle_name] > (target + tolerance):
                    errors.append(f"⚠️ {error_msg} (Lowest: {min_angles[angle_name]}°, Target: ~{target}°)")
                    
    verdict = "✅ Correct Form" if not errors else "❌ Incorrect Form"
    return verdict, errors