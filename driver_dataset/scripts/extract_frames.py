import cv2
import os

def extract_frames_from_video(video_path, output_folder, step=30):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    success, frame = cap.read()

    while success:
        if count % step == 0:
            filename = os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame{frame_id}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1
        success, frame = cap.read()
        count += 1

    cap.release()
    print(f"Extracted {frame_id} frames from {video_path}.")

def batch_process(category):
    input_dir = f"../videos/{category}"
    output_dir = f"../frames/{category}"

    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            extract_frames_from_video(os.path.join(input_dir, file), output_dir)

if __name__ == "__main__":
    categories = ["drowsy", "yawning", "phone_call", "texting", "smoking", "looking_back", "normal"]
    for cat in categories:
        print(f"Processing category: {cat}")
        batch_process(cat)