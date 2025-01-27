import torch
import cv2
import pyttsx3
engine = pyttsx3.init()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
video = cv2.VideoCapture(0)
detected_objects = set()
while True:
    ret, frame = video.read()
    if not ret:
        break
    results = model(frame)
    current_objects = [result['name'] for result in results.pandas().xyxy[0].to_dict('records')]
    detected_objects.update(current_objects)
    if len(detected_objects) > 0:
        text_output = "I have detected: " + ", ".join(detected_objects)
        print(text_output)
        engine.say(text_output)
        engine.runAndWait()
        detected_objects.clear() 
    frame = results.render()[0]
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
engine.stop()

# import torch
# import cv2
# import pyttsx3
# # Load the YOLOv5 model with trust
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)  # Load the small model (yolov5s)

# # Initialize webcam
# engine = pyttsx3.init()
# video = cv2.VideoCapture(0)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break

#     # Perform inference
#     results = model(frame)
#     detected_objects = [result['name'] for result in results.pandas().xyxy[0].to_dict('records')]

#     # Create text output and speech input
#     text_output = "I have detected: " + ", ".join(detected_objects)
#     engine.say(text_output)
#     engine.runAndWait()
#     # Render results on the frame
#     frame = results.render()[0]

#     # Show the output
#     cv2.imshow('YOLOv5 Object Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()
# engine.stop()