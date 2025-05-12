import os
import cv2
import torch
import yaml
from facenet_pytorch import MTCNN, InceptionResnetV1

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

CONFIG_PATH = "train_config.yml"

default_config = {
    'model': {
        'name': 'face_recognition_model',
        'save_path': 'models/face_recognition_model.pkl',
    },
    'data': {
        'dataset_path': 'registered_faces',
        'image_size': [160, 160],
        'train_split': 0.8
    },
    'training': {
        'algorithm': 'facenet',
        'num_components': 100,
        'threshold': 50.0
    },
    'logging': {
        'log_dir': 'logs/',
        'verbose': True
    }
}

def create_default_config():
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(default_config, f)
    print(f"Default config created at: {CONFIG_PATH}")

def train_model():
    # Create config if missing
    if not os.path.exists(CONFIG_PATH):
        create_default_config()

    print("Training started...")

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['data']['dataset_path']
    if not os.path.exists(data_dir):
        print("No registered faces found.")
        return

    face_count = 0
    for name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, name)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_tensor, prob = mtcnn(img_rgb, return_prob=True)
            if face_tensor is not None:
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                resnet(face_tensor.to(device))  # Forward pass
                face_count += 1
                print(f"Trained: {name}/{img_name}")
            else:
                print(f"Face not detected: {name}/{img_name}")

    print(f"Training complete. Total trained faces: {face_count}")

if __name__ == "__main__":
    train_model()
