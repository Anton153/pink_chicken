import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom PyTorch Dataset
class DrivingDataset(Dataset):
    def __init__(self, video_filenames, processed_data_dir):
        self.video_filenames = video_filenames
        self.processed_data_dir = processed_data_dir
        self.data = []
        self._load_data()

    def _load_data(self):
        # Lazy load metadata for filenames; actual frames are accessed during __getitem__
        for filename in self.video_filenames:
            file_path = os.path.join(self.processed_data_dir, filename)
            with open(file_path, 'rb') as f:
                video_data = pickle.load(f)
                self.data.append({
                    'frames': video_data["frames"],
                    'commands': video_data["commands"],  # Continuous velocities
                })

    def __len__(self):
        # Total number of frames across all videos
        return sum(len(item['frames']) for item in self.data)

    def __getitem__(self, idx):
        # Map global index to video and frame
        cumulative_frames = 0
        for item in self.data:
            if idx < cumulative_frames + len(item['frames']):
                local_idx = idx - cumulative_frames
                frame = item['frames'][local_idx]
                command = item['commands'][local_idx]
                
                # Extract only linear and angular velocities
                linear_velocity = command[1]  # Assuming column 1
                angular_velocity = command[2]  # Assuming column 2
                return np.array(frame), np.array([linear_velocity, angular_velocity], dtype=np.float32)
            cumulative_frames += len(item['frames'])
        raise IndexError("Index out of range")


# Custom collate_fn for batching
def driving_collate_fn(batch):
    X_images, commands = [], []
    for frame, command in batch:
        X_images.append(frame)
        commands.append(command)  # Continuous values for linear and angular velocity

    # Convert to PyTorch tensors
    X_images = torch.tensor(np.array(X_images), dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
    commands = torch.tensor(np.array(commands), dtype=torch.float32)  # (N, 2)

    return X_images, commands  # Commands contains both linear and angular velocity

# Instantiate Dataset
processed_data_dir = "/home/fizzer/ENPH353_Competition/src/pink_chicken/pipeline_testing"
video_filenames = [f for f in os.listdir(processed_data_dir) if f.endswith('.pkl')]
dataset = DrivingDataset(video_filenames, processed_data_dir)

# DataLoader with Lazy Loading
batch_size = 32
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=driving_collate_fn,
    shuffle=True
)

def verify_dataset(dataset, num_samples=2):
    """
    Visualizes random samples from the dataset to verify correctness.

    Args:
        dataset (DrivingDataset): The dataset to verify.
        num_samples (int): Number of random samples to display.
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx in indices:
        frame, command = dataset[idx]
        
        # Convert frame to uint8 for display (assuming values are in [0, 255])
        if frame.max() > 1:
            frame = frame.astype(np.uint8)
        else:
            frame = (frame * 255).astype(np.uint8)
        
        # Display frame and command
        plt.imshow(frame)
        plt.title(f"Linear: {command[0]:.2f}, Angular: {command[1]:.2f}")
        plt.axis('off')
        plt.show()

# Define Model
class DualOutputDrivingCNN(nn.Module):
    def __init__(self):
        super(DualOutputDrivingCNN, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),

            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.linear_velocity_head = nn.Linear(128, 1)
        self.angular_velocity_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.model(x)
        linear_velocity = self.linear_velocity_head(features)
        angular_velocity = self.angular_velocity_head(features)
        return linear_velocity, angular_velocity

#verify dataset
verify_dataset(dataset, num_samples=0)

# Instantiate Model
model = DualOutputDrivingCNN()

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float('inf')  # Initialize best loss to a very large value
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0   .5, patience=22, verbose=True)
best_model_path = "pytorchTest_5.pth"

# Training Loop
epochs = 1000
early_stop_patience = 20
no_improve_epochs = 0

for epoch in range(epochs):
    model.train()
    total_epoch_loss = 0  # Track cumulative loss for the epoch

    for X_images, commands in dataloader:
        X_images = X_images.to(device)
        linear_targets = commands[:, 0].unsqueeze(1).to(device)  # Linear velocity
        angular_targets = commands[:, 1].unsqueeze(1).to(device)  # Angular velocity

        # Forward pass
        optimizer.zero_grad()
        linear_preds, angular_preds = model(X_images)

        # Calculate losses
        linear_loss = criterion(linear_preds, linear_targets)
        angular_loss = criterion(angular_preds, angular_targets)
        total_loss = linear_loss + angular_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Accumulate loss
        total_epoch_loss += total_loss.item()

    # Average training loss for the epoch
    avg_train_loss = total_epoch_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")

    # Save the model if it has the best training loss so far
    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with training loss: {best_loss:.4f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Early stopping check
    if no_improve_epochs >= early_stop_patience:
        print("Early stopping triggered.")
        break

    # Adjust learning rate based on training loss
    scheduler.step(avg_train_loss)


