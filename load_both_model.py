import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F  # For softmax
import torchvision.transforms as transforms
import os
import json

# Global variables
model_yolo = None
model_pytorch = None
image_path = None
class_names = None

# Path to the JSON file for saving model paths
config_path = "model_paths.json"

# Load class names from the train folder
def load_class_names(train_path):
    class_names = sorted(os.listdir(train_path))
    return class_names

# Function to load the YOLOv8 model
def load_yolo_model():
    global model_yolo
    model_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
    if model_path:
        try:
            model_yolo = YOLO(model_path)
            messagebox.showinfo("Success", "YOLO model loaded successfully!")
            save_model_path("yolo_model_path", model_path)  # Save the path to JSON
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

# Function to load the PyTorch model
def load_pytorch_model():
    global model_pytorch
    model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
    if model_path:
        try:
            # Define the PyTorch model architecture
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

            # Load the model
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model_pytorch = TrafficSignClassifier(input_channels=3, hidden_layers=[7680, 3840, 1920, 960, 480, 240], num_classes=len(class_names))
            model_pytorch.load_state_dict(checkpoint['model_state_dict'])
            model_pytorch.eval()
            messagebox.showinfo("Success", "PyTorch model loaded successfully!")
            save_model_path("pytorch_model_path", model_path)  # Save the path to JSON
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PyTorch model: {e}")

# Function to save model paths to JSON
def save_model_path(key, path):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config[key] = path
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

# Function to load model paths from JSON
def load_model_paths():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("yolo_model_path"), config.get("pytorch_model_path")
    return None, None

# Function to select an image
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        display_image(image_path)

# Function to display the selected image
def display_image(image_path):
    image = Image.open(image_path)
    image = image.resize((600, 400), Image.Resampling.LANCZOS)  # Resize for display
    image_tk = ImageTk.PhotoImage(image)
    image_label.config(image=image_tk)
    image_label.image = image_tk  # Keep a reference to avoid garbage collection

# Function to perform object detection and classification
def detect_objects():
    if not model_yolo:
        messagebox.showerror("Error", "Please load the YOLO model first!")
        return
    if not model_pytorch:
        messagebox.showerror("Error", "Please load the PyTorch model first!")
        return
    if not image_path:
        messagebox.showerror("Error", "Please select an image first!")
        return

    # Load the image
    image = cv2.imread(image_path)

    # Perform object detection using YOLOv8
    results = model_yolo(image)

    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0]  # Get confidence score
            class_id = box.cls[0]  # Get class ID
            yolo_class_label = model_yolo.names[int(class_id)]  # Get YOLO class label

            # Skip if confidence is less than 0.5
            if confidence < 0.1:
                continue

            # Crop the detected object and resize to 68x68
            detected_object = image[y1:y2, x1:x2]
            detected_object = cv2.resize(detected_object, (68, 68))

            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ])
            detected_object = transform(detected_object).unsqueeze(0)

            # Classify the detected object using the PyTorch model
            with torch.no_grad():
                output = model_pytorch(detected_object)
                probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
                confidence_pytorch, predicted = torch.max(probabilities, 1)
                pytorch_class_label = class_names[predicted.item()]  # Get PyTorch class label

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{yolo_class_label} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Print detection results to the console
            print(f"YOLO Detection: {yolo_class_label} | Confidence: {confidence:.2f} | Bounding Box: ({x1}, {y1}, {x2}, {y2})")
            print(f"PyTorch Classification: {pytorch_class_label} | Confidence: {confidence_pytorch.item():.2f}")
            print("-" * 50)

    # Convert the image from BGR (OpenCV) to RGB (for displaying in Tkinter)
    detected_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_image = Image.fromarray(detected_image)
    detected_image = detected_image.resize((600, 400), Image.Resampling.LANCZOS)
    detected_image_tk = ImageTk.PhotoImage(detected_image)

    # Display the image with detections
    image_label.config(image=detected_image_tk)
    image_label.image = detected_image_tk  # Keep a reference to avoid garbage collection

# Create the Tkinter window
root = tk.Tk()
root.title("YOLOv8 and PyTorch Object Detection")
root.geometry("800x600")

# Load class names
train_path = r"C:\Users\bekka\Desktop\SD\data\train"
class_names = load_class_names(train_path)

# Load saved model paths
yolo_model_path, pytorch_model_path = load_model_paths()

# Load YOLO model if path exists
if yolo_model_path and os.path.exists(yolo_model_path):
    try:
        model_yolo = YOLO(yolo_model_path)
        print("YOLO model loaded automatically from saved path.")
    except Exception as e:
        print(f"Failed to load YOLO model from saved path: {e}")

# Load PyTorch model if path exists
if pytorch_model_path and os.path.exists(pytorch_model_path):
    try:
        # Define the PyTorch model architecture
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

        # Load the model
        checkpoint = torch.load(pytorch_model_path, map_location=torch.device('cpu'))
        model_pytorch = TrafficSignClassifier(input_channels=3, hidden_layers=[7680, 3840, 1920, 960, 480, 240], num_classes=len(class_names))
        model_pytorch.load_state_dict(checkpoint['model_state_dict'])
        model_pytorch.eval()
        print("PyTorch model loaded automatically from saved path.")
    except Exception as e:
        print(f"Failed to load PyTorch model from saved path: {e}")

# Create buttons
load_yolo_button = tk.Button(root, text="Load YOLO Model (.pt)", command=load_yolo_model)
load_yolo_button.pack(pady=10)

load_pytorch_button = tk.Button(root, text="Load PyTorch Model (.pth)", command=load_pytorch_model)
load_pytorch_button.pack(pady=10)

select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack(pady=10)

detect_button = tk.Button(root, text="Detect Objects", command=detect_objects)
detect_button.pack(pady=10)

# Create a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()