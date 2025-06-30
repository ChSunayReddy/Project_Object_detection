import torch
import cv2

# Load YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load video
cap = cv2.VideoCapture("home.mp4")  # Use 0 for webcam

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define VideoWriter to save output as MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Render predictions
    annotated_frame = results.render()[0]

    # Save frame to output video
    out.write(annotated_frame)

    # Display
    cv2.imshow("Object Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
