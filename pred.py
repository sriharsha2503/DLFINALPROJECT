import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, font
import collections

class PostureClassifier(nn.Module):
    def __init__(self):
        super(PostureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(132, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def load_model(exercise_name, device):
    model = PostureClassifier().to(device)
    model.load_state_dict(torch.load(f'models/{exercise_name}_posture_model.pth', map_location=device))
    model.eval()
    return model

def preprocess_landmarks(landmarks):
    mp_pose = mp.solutions.pose

    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    origin_x = (left_shoulder.x + right_shoulder.x) / 2
    origin_y = (left_shoulder.y + right_shoulder.y) / 2
    origin_z = (left_shoulder.z + right_shoulder.z) / 2

    keypoints = []
    for lm in landmarks.landmark:
        keypoints.extend([
            lm.x - origin_x,
            lm.y - origin_y,
            lm.z - origin_z,
            lm.visibility
        ])
    return np.array(keypoints, dtype=np.float32)

def log_reps_to_file(exercise_name, reps):
    with open("reps_log.txt", "a") as f:
        f.write(f"{datetime.now()} - Exercise: {exercise_name}, Total Reps: {reps}\n")

def calculate_elbow_angle(landmarks, mp_pose):
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

    upper_arm = np.array([left_shoulder.x - left_elbow.x, left_shoulder.y - left_elbow.y])
    forearm = np.array([left_wrist.x - left_elbow.x, left_wrist.y - left_elbow.y])

    dot_product = np.dot(upper_arm, forearm)
    norm_product = np.linalg.norm(upper_arm) * np.linalg.norm(forearm) + 1e-6
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_shoulder_abduction_angle(landmarks, mp_pose):
    # Angle between torso and upper arm to detect abduction
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

    upper_arm = np.array([left_elbow.x - left_shoulder.x, left_elbow.y - left_shoulder.y])
    torso = np.array([left_hip.x - left_shoulder.x, left_hip.y - left_shoulder.y])

    dot_product = np.dot(upper_arm, torso)
    norm_product = np.linalg.norm(upper_arm) * np.linalg.norm(torso) + 1e-6
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def start_prediction(exercise_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(exercise_name, device)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    reps = 0
    arm_raised = False
    knee_raised = False

    angle_history = collections.deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        posture_text = "No Pose Detected"
        current_pose_correct = False

        if results.pose_landmarks:
            keypoints = preprocess_landmarks(results.pose_landmarks)
            input_tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.item()

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if exercise_name == "shoulder_abduction":
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]

                # Check visibility to avoid false positives
                if left_wrist.visibility < 0.5 or left_shoulder.visibility < 0.5:
                    current_pose_correct = False
                else:
                    # Wrist should be above shoulder (y smaller) and arm abducted sufficiently
                    wrist_above_shoulder = left_wrist.y < left_shoulder.y - 0.05  # small margin
                    abduction_angle = calculate_shoulder_abduction_angle(results.pose_landmarks, mp_pose)
                    # Abduction angle roughly between 40 and 160 degrees indicates arm raised sideways
                    abduction_correct = 40 <= abduction_angle <= 160

                    current_pose_correct = prediction > 0.5 and wrist_above_shoulder and abduction_correct

                posture_text = "Correct" if current_pose_correct else "Incorrect"

                if current_pose_correct:
                    arm_raised = True
                elif arm_raised and not current_pose_correct:
                    reps += 1
                    arm_raised = False

            elif exercise_name == "elbow_flexion":
                angle_deg = calculate_elbow_angle(results.pose_landmarks, mp_pose)
                angle_history.append(angle_deg)
                smooth_angle = np.mean(angle_history)

                min_correct_angle = 20
                max_correct_angle = 145
                flexion_threshold = 60
                extension_threshold = 130

                current_pose_correct = min_correct_angle <= smooth_angle <= max_correct_angle
                posture_text = f"{'Correct' if current_pose_correct else 'Incorrect'} | Elbow Angle: {int(smooth_angle)}°"

                if current_pose_correct and smooth_angle < flexion_threshold:
                    arm_raised = True
                elif arm_raised and smooth_angle > extension_threshold:
                    reps += 1
                    arm_raised = False

            elif exercise_name == "standing_knee_raise":
                left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
                raise_threshold_y = 0.5

                current_pose_correct = prediction > 0.5 and left_knee_y < raise_threshold_y
                posture_text = "Correct" if current_pose_correct else "Incorrect"

                if current_pose_correct:
                    knee_raised = True
                elif knee_raised and not current_pose_correct:
                    reps += 1
                    knee_raised = False

        color = (0, 255, 0) if current_pose_correct else (0, 0, 255)
        cv2.putText(frame, f"Posture: {posture_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        text = f"REPS: {reps}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = frame.shape[0] - 20
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow('Posture Check', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_reps_to_file(exercise_name, reps)
            break

    cap.release()
    cv2.destroyAllWindows()

def create_gui():
    root = tk.Tk()
    root.title("Exercise Posture Prediction")
    root.geometry("400x300")
    root.configure(bg="#f0f0f0")

    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    button_font = font.Font(family="Helvetica", size=12)

    tk.Label(root, text="Select Exercise", font=title_font, bg="#f0f0f0").pack(pady=20)

    exercises = ["Shoulder Abduction", "Elbow Flexion", "Standing Knee Raise"]

    def on_select(exercise):
        start_prediction(exercise.lower().replace(" ", "_"))

    for exercise in exercises:
        btn = tk.Button(root, text=exercise, font=button_font, width=20, height=2,
                        command=lambda ex=exercise: on_select(ex))
        btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
