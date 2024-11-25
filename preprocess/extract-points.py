import cv2
import os
import mediapipe as mp
import time
import numpy as np
import json

FRAMES_FOLDER = "./video-frames"

# Inicializar Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def extract_hand_keypoints(frame):
    # Convertir la imagen a RGB (Mediapipe espera imágenes RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])  # Coordenadas (x, y, z)
    return keypoints


def pad_keypoints(keypoints, target_length):
    # Pad the keypoints array to ensure consistent shape
    padded_keypoints = np.zeros((target_length, 3))
    keypoints = np.array(keypoints)
    padded_keypoints[:min(len(keypoints), target_length), :] = keypoints[:target_length, :]
    return padded_keypoints.tolist()


def parse_points(frame):
    keypoints = extract_hand_keypoints(frame)
    
    if len(keypoints) > 42:
        return keypoints[:42]

    return pad_keypoints(keypoints, 42)


def find_frame_word(data, video_id):
    for sample in data:
        for instance in sample["instances"]:
            if instance["video_id"] == video_id:
                return sample["gloss"]



def main():
    folders = os.listdir(FRAMES_FOLDER)

    dataset = []

    with open("training.json") as training_config_file:
        data = json.load(training_config_file)

        for folder in folders:
            input_points = []

            folder_files = os.listdir(f"{FRAMES_FOLDER}/{folder}")

            for file in folder_files[:30]:
                path = f"{FRAMES_FOLDER}/{folder}/{file}"
                frame = cv2.imread(path)
                try:
                    input_points.append(parse_points(frame))
                except IndexError:
                    continue

            if len(input_points) < 30:
                for _ in range(30 - len(input_points)):
                    input_points.append(pad_keypoints([[0, 0, 0]], 42))

            print(len(input_points))

            dataset.append({
                "fps": 25,
                "word": find_frame_word(data, folder),
                "frames_compiled_points": input_points
            })

    with open("dataset.json", "w") as file:
        json.dump(dataset, file)

if __name__ == "__main__":
    main()
