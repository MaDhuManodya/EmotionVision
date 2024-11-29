import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Initialize MTCNN face detector
detector = MTCNN()

# Load your pre-trained model
model_path = 'facial_emotion_model.h5'  # Replace with the correct path to your saved model
model = load_model(model_path)  # Load the model

# Function to detect and crop faces, and also draw bounding boxes
def detect_and_crop_faces(img_path):
    # Load image using OpenCV
    img = cv2.imread(img_path)
    
    # Convert the image to RGB (MTCNN requires RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detector.detect_faces(img_rgb)
    
    if len(faces) > 0:
        return img, faces  # Return the original image with bounding boxes and the detected faces
    else:
        print("No faces detected.")
        return img, None

# Load and preprocess the new image
img_path = 'image-asset.jpeg'  # Replace with the path to the new image

# Detect faces and get bounding boxes
original_image, faces = detect_and_crop_faces(img_path)

if faces is not None:
    # Loop through all detected faces
    for face in faces:
        # Extract the bounding box coordinates
        x, y, width, height = face['box']
        
        # Draw bounding box around the face
        cv2.rectangle(original_image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green color for bounding box

        # Extract the face region from the image
        face_region = original_image[y:y + height, x:x + width]
        
        # Resize the face to 224x224 (input size for the model)
        face_resized = cv2.resize(face_region, (224, 224))
        
        # Convert the resized face to an array
        img_array = image.img_to_array(face_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize image
        
        # Predict the emotion for the current face
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions) * 100  # Get confidence level
        
        # Class labels corresponding to the emotions
        class_labels = ['Angry', 'Happy', 'Sad']
        
        # Get the predicted emotion and display it with confidence
        predicted_emotion = class_labels[predicted_class[0]]
        print(f"Predicted emotion: {predicted_emotion} with confidence: {confidence:.2f}%")

        # Add the prediction text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_image, f"{predicted_emotion}: {confidence:.2f}%", (x, y - 10), font, 0.9, (255, 255, 0), 2)

    # Resize the image to fit the window size
    window_width = 1000
    window_height = 1000
    resized_image = cv2.resize(original_image, (window_width, window_height))

    # Show the image with bounding boxes and predictions for all faces
    cv2.imshow("Face Detection and Emotion Prediction", resized_image)

    # Resize the window (optional, you can adjust the size based on your preferences)
    cv2.resizeWindow("Face Detection and Emotion Prediction", window_width, window_height)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No faces detected in the image.")
