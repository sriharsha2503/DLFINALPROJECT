import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import ExerciseDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def compute_metrics(preds, labels):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1 = f1_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np)
    return f1, precision

def train_model(exercise_name, epochs=50, batch_size=32, lr=0.001, early_stopping_patience=10):
    set_seed()
    train_ds = ExerciseDataset(exercise_name, split='train')
    val_ds = ExerciseDataset(exercise_name, split='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {exercise_name} on {device}")

    model = PostureClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_precisions, val_precisions = [], []

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).view(-1,1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds)
            all_labels.extend(labels)

        train_loss = running_loss / total
        train_acc = correct / total
        f1, precision = compute_metrics(torch.stack(all_preds), torch.stack(all_labels))
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1s.append(f1)
        train_precisions.append(precision)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_labels_all = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).view(-1,1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds)
                val_labels_all.extend(labels)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1, val_precision = compute_metrics(torch.stack(val_preds), torch.stack(val_labels_all))

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_precisions.append(val_precision)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {f1:.4f} Precision: {precision:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} Precision: {val_precision:.4f}")

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/{exercise_name}_posture_model.pth')
            print(f"Saved best model for {exercise_name} at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Plot training curves
    plt.figure(figsize=(16,10))

    plt.subplot(2,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title(f'{exercise_name} Loss')

    plt.subplot(2,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title(f'{exercise_name} Accuracy')

    plt.subplot(2,2,3)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.legend()
    plt.title(f'{exercise_name} F1 Score')

    plt.subplot(2,2,4)
    plt.plot(train_precisions, label='Train Precision')
    plt.plot(val_precisions, label='Val Precision')
    plt.legend()
    plt.title(f'{exercise_name} Precision')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train posture classification models.")
    parser.add_argument('--exercises', nargs='+', default=["shoulder_abduction", "elbow_flexion", "standing_knee_raise"],
                        help="List of exercises to train on.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    args = parser.parse_args()

    for ex in args.exercises:
        train_model(ex, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
