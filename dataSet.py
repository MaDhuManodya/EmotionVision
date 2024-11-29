import os
import cv2
import shutil
from tqdm import tqdm  # For progress bar
from mtcnn import MTCNN  # Import MTCNN for face detection

# Path to your dataset and new folder to save cropped faces
base_dir = "data"  # Replace with your dataset path
output_dir = "faceImages"  # Directory where cropped faces will be saved

# Desired dimensions for resizing (224x224 is a standard size for emotion recognition tasks)
IMG_HEIGHT, IMG_WIDTH = 224, 224  # Resize dimensions

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize MTCNN face detector
detector = MTCNN()

def contains_face(image_path: str) -> bool:
    """Check if an image contains a face using MTCNN."""
    image = cv2.imread(image_path)
    if image is None:
        return False  # Skip unreadable images

    # Convert the image to RGB (MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(image_rgb)

    # If one or more faces are detected, return True
    return len(faces) > 0

def crop_and_save_faces(image_path: str, output_dir: str, label: str, count: int):
    """Crop the faces from an image, resize them, and save them."""
    image = cv2.imread(image_path)
    if image is None:
        return  # Skip unreadable images

    # Convert the image to RGB (MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(image_rgb)
    
    for i, face in enumerate(faces):
        # Get the bounding box of the face
        x, y, w, h = face['box']
        
        # Crop the face from the image
        cropped_face = image[y:y+h, x:x+w]
        
        # Resize the cropped face to the desired dimensions
        resized_face = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))
        
        # Create a unique name for each cropped face
        face_filename = f"{label}_{count}_{i}.jpg"  # Include label, image index, and face index
        
        # Save the cropped and resized face
        cv2.imwrite(os.path.join(output_dir, face_filename), resized_face)
        print(f"Saved cropped and resized face: {os.path.join(output_dir, face_filename)}")

# Check all images and crop faces from those containing them
count = 0  # To keep track of image count in the output folder
for label in os.listdir(base_dir):
    label_dir = os.path.join(base_dir, label)
    for image_name in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
        image_path = os.path.join(label_dir, image_name)

        if contains_face(image_path):
            count += 1
            crop_and_save_faces(image_path, output_dir, label, count)

print(f"Face-detected and resized images are saved in: {output_dir}")
