import cv2
import mediapipe as mp
import csv
import os
import time
import threading

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def save_data_to_csv(data, exercise_name, class_name, timestamp):
    base_dir = os.path.join("C:\\Users\\sanma\\Desktop\\rehabproj\\dataset", exercise_name, class_name)
    os.makedirs(base_dir, exist_ok=True)
    
    filename = f"{exercise_name}_{class_name}_{timestamp}.csv"
    filepath = os.path.join(base_dir, filename)
    
    header = ['exercise', 'class']
    for idx in range(33):  
        header += [f'x{idx}', f'y{idx}', f'z{idx}', f'v{idx}']
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    
    print(f"Saved {len(data)} frames to {filepath}")

def collect_data(exercise_name, class_name):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    print("Camera initialized. Starting data collection...")
    
    recording = False
    data = []
    frame_count = 0
    
    print(f"Press 's' to start/stop recording for {exercise_name} - {class_name}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([
                    landmark.x, 
                    landmark.y, 
                    landmark.z, 
                    landmark.visibility
                ])
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            if recording:
                data.append([exercise_name, class_name] + keypoints)
                frame_count += 1

        status_color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f"Recording: {recording} [{frame_count} frames]", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  
            recording = not recording
            if not recording and data:
                timestamp = int(time.time())
                threading.Thread(target=save_data_to_csv, args=(data, exercise_name, class_name, timestamp)).start()
                data = []
                frame_count = 0
        elif key == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Select exercise to collect data for:")
    print("1. Shoulder Abduction")
    print("2. Elbow Flexion")
    print("3. Standing Knee Raise")
    choice = input("Enter choice (1/2/3): ").strip()

    exercise_map = {
        "1": "shoulder_abduction",
        "2": "elbow_flexion",
        "3": "standing_knee_raise"
    }

    if choice not in exercise_map:
        print("Invalid choice.")
        exit()

    class_label = input("Enter class (correct/incorrect): ").strip().lower()
    collect_data(exercise_map[choice], class_label)
