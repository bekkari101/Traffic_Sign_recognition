import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set the CPU core limit to 3
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["NUMEXPR_NUM_THREADS"] = "3"
os.environ["OMP_THREAD_LIMIT"] = "3"

# Limit how many CPU threads PyTorch uses
torch.set_num_threads(3)

# Data paths
base_data_folder = r"C:\Users\bekka\Desktop\SD\code"
data_folder = r"C:\Users\bekka\Desktop\SD\data"
train_path = r"C:\Users\bekka\Desktop\SD\data\train"
valid_path = r"C:\Users\bekka\Desktop\SD\data\valid"

# Checkpoints and Graph paths
checkpoint_folder = os.path.join(base_data_folder, "checkpoints")
graph_folder = os.path.join(base_data_folder, "graphs")
graph_path = os.path.join(graph_folder, "training_progress.png")
json_path = os.path.join(graph_folder, "training_process.json")

# Ensure directories exist
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(graph_folder, exist_ok=True)

# Hyperparameters
input_channels = 3
learning_rate = 0.000001
batch_size = 256
epochs = 30
hidden_layers = [7680, 3840, 1920, 960, 480, 240]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * input_channels, std=[0.5] * input_channels)
])

# Dataset and Dataloader
class ProgressBarDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

def get_dataloader(dataset_path, transform, batch_size, shuffle=True, num_workers=2):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

train_loader = get_dataloader(train_path, transform, batch_size, shuffle=True, num_workers=2)
valid_loader = get_dataloader(valid_path, transform, batch_size, shuffle=False, num_workers=2)

# Define the model
class TrafficSignClassifier(nn.Module):
    def __init__(self, input_channels, hidden_layers, num_classes):
        super(TrafficSignClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dummy input to calculate flattened size of features
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 68, 68)
            flat_size = self.features(dummy_input).view(1, -1).size(1)

        layers = []
        for in_features, out_features in zip([flat_size] + hidden_layers[:-1], hidden_layers):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        layers.pop()
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, criterion, optimizer, and scheduler
num_classes = len(os.listdir(train_path))
model = TrafficSignClassifier(input_channels, hidden_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

def load_checkpoint(model, optimizer, checkpoint_folder, custom_lr=None):
    checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pth')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint = torch.load(os.path.join(checkpoint_folder, latest_checkpoint))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        # Set custom learning rate if provided
        if custom_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = custom_lr
            print(f"Custom learning rate set to {custom_lr}")

        return start_epoch
    else:
        return 0

# Usage
custom_learning_rate = 0.0000001  # Set your desired learning rate here
start_epoch = load_checkpoint(model, optimizer, checkpoint_folder, custom_lr=custom_learning_rate)


def save_epoch_to_json(epoch, train_loss, valid_loss, train_accuracy, valid_accuracy, learning_rate):
    epoch_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'train_accuracy': train_accuracy,
        'valid_accuracy': valid_accuracy,
        'learning_rate': learning_rate
    }
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(epoch_data)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        valid_loss, valid_correct, valid_total = 0.0, 0, 0

        with tqdm(total=len(train_loader) + len(valid_loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
            pbar.set_postfix({"Train Loss": 0, "Valid Loss": 0, "Train Acc": 0, "Valid Acc": 0})

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

                train_accuracy = 100.0 * train_correct / train_total

                pbar.set_postfix({
                    "Train Loss": train_loss / (train_total or 1),
                    "Train Acc": train_accuracy
                })
                pbar.update(1)

            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    valid_total += labels.size(0)
                    valid_correct += predicted.eq(labels).sum().item()

                valid_accuracy = 100.0 * valid_correct / valid_total

            pbar.set_postfix({
                "Train Loss": train_loss / (train_total or 1),
                "Train Acc": train_accuracy,
                "Valid Loss": valid_loss / (valid_total or 1),
                "Valid Acc": valid_accuracy
            })
        
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        scheduler.step(valid_loss / len(valid_loader))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'valid_loss': valid_losses[-1],
            'train_accuracy': train_accuracy,
            'valid_accuracy': valid_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, os.path.join(checkpoint_folder, f"checkpoint_epoch_{epoch+1}.pth"))

        save_epoch_to_json(epoch+1, train_losses[-1], valid_losses[-1], train_accuracy, valid_accuracy, optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Valid Loss: {valid_losses[-1]:.4f} | Valid Acc: {valid_accuracy:.2f}%")

    # Plot and save the training progress graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig(graph_path)
    plt.show()

    print("Training complete!")
