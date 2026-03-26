import cv2
import os
from tqdm import tqdm  # progress bar

DATA_DIR   = 'data'        # your dataset folder
OUTPUT_DIR = 'frames'      # where extracted images go

def extract_middle_frame(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    success, frame = cap.read()
    cap.release()
    
    if success:
        cv2.imwrite(save_path, frame)
        return True
    return False

for split in ['train', 'val']:
    split_dir = os.path.join(DATA_DIR, split)
    
    if not os.path.exists(split_dir):
        print(f"Skipping {split} — folder not found")
        continue
    
    exercise_folders = os.listdir(split_dir)
    print(f"\nProcessing {split}: {len(exercise_folders)} exercises")
    
    for exercise in tqdm(exercise_folders):
        exercise_path = os.path.join(split_dir, exercise)
        if not os.path.isdir(exercise_path):
            continue
        out_path = os.path.join(OUTPUT_DIR, split, exercise)
        os.makedirs(out_path, exist_ok=True)
        
        for video_file in os.listdir(exercise_path):
            if video_file.endswith('.avi'):
                vid_path  = os.path.join(exercise_path, video_file)
                save_path = os.path.join(out_path, video_file.replace('.avi', '.jpg'))
                extract_middle_frame(vid_path, save_path)

print("\n✅ All frames extracted!")