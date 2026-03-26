import shutil
import os

# Define the exercises you have rules for in form_rules.py
TARGET_EXERCISES = ['PushUps', 'WallPushups', 'HandstandPushups', 'JumpingJack', 'CleanAndJerk']

ORIGINAL_FRAMES = 'frames'
NEW_FRAMES = 'frames_filtered'

for split in ['train', 'val']:
    for exercise in TARGET_EXERCISES:
        src = os.path.join(ORIGINAL_FRAMES, split, exercise)
        dst = os.path.join(NEW_FRAMES, split, exercise)
        
        if os.path.exists(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied {exercise} to {NEW_FRAMES}")

print("Filtering complete!")