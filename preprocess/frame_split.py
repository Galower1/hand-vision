import os
import cv2

VIDEOS_FOLDER = "./training-videos"

def split_video(dir_path, video_path):
    video_path = video_path
    output_dir = dir_path
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")

    frame_interval = int(round(video_fps / 25))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'")


def main():
    files = os.listdir(VIDEOS_FOLDER)
    for file in files:
        name = file.replace(".mp4", "")
        split_video(name, f"./training-videos/{file}")

if __name__ == "__main__":
    main()

