import torch
import cv2
import pyttsx3
import numpy as np

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# Image path (update to the actual path of your image)
image_path = r"C:\Users\sunay\Downloads\image.jpg"

# Read the image using OpenCV
image_cv2 = cv2.imread(image_path)
if image_cv2 is None:
    print("Failed to load the image. Please check the file path.")
    exit()

# Perform inference
results = model(image_cv2)

# Extract detected object names
detected_objects = [result['name'] for result in results.pandas().xyxy[0].to_dict('records')]

# Announce detected objects if any
if detected_objects:
    text_output = "I have detected: " + ", ".join(detected_objects)
    print(text_output)
    engine.say(text_output)
    engine.runAndWait()
else:
    print("No objects detected.")
    engine.say("No objects detected.")
    engine.runAndWait()

# Render results on the image
annotated_image = results.render()[0]

# Display the image
# cv2.imshow('YOLOv5 Object Detection', annotated_image)
# cv2.waitKey(0)  # Wait indefinitely until a key is pressed
# cv2.destroyAllWindows()
engine.stop()
