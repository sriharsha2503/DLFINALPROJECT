import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ExerciseDataset(Dataset):
    def __init__(self, exercise_name, split='train', test_size=0.2, random_state=42):
        self.exercise_name = exercise_name
        self.split = split
        self.scaler = StandardScaler()
        
        features, labels = self._load_data()
        
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, 
            test_size=test_size, 
            stratify=labels,
            random_state=random_state
        )
        
        if len(X_train) > 0:
            self.scaler.fit(X_train)
        
        if split == 'train':
            self.X = self.scaler.transform(X_train) if len(X_train) > 0 else X_train
            self.y = y_train
        else:
            self.X = self.scaler.transform(X_val) if len(X_val) > 0 else X_val
            self.y = y_val

    def _load_data(self):
        features = []
        labels = []
        
        for class_name in ['correct', 'incorrect']:
            class_path = os.path.join('dataset', self.exercise_name, class_name)
            if not os.path.exists(class_path):
                continue
                
            for file in os.listdir(class_path):
                if file.endswith('.csv'):
                    with open(os.path.join(class_path, file), 'r') as f:
                        reader = csv.reader(f)
                        next(reader)  # skip header
                        for row in reader:
                            keypoints = list(map(float, row[2:134]))
                            keypoints = normalize_landmarks(keypoints)
                            label = 1 if row[1] == 'correct' else 0
                            features.append(keypoints)
                            labels.append(label)
        
        return np.array(features), np.array(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return features, label

def normalize_landmarks(keypoints):
    left_shoulder_idx = 11 * 4
    right_shoulder_idx = 12 * 4

    left_shoulder = np.array(keypoints[left_shoulder_idx:left_shoulder_idx+3])
    right_shoulder = np.array(keypoints[right_shoulder_idx:right_shoulder_idx+3])

    origin = (left_shoulder + right_shoulder) / 2

    normalized = np.array(keypoints).copy()
    for i in range(33):
        normalized[i*4] -= origin[0]    
        normalized[i*4 + 1] -= origin[1]
        normalized[i*4 + 2] -= origin[2]
    return normalized

if __name__ == "__main__":
    exercises = ["shoulder_abduction", "elbow_flexion", "standing_knee_raise"]
    for ex in exercises:
        train_ds = ExerciseDataset(ex, split='train')
        val_ds = ExerciseDataset(ex, split='val')
        print(f"{ex}: Train samples = {len(train_ds)}, Val samples = {len(val_ds)}")
