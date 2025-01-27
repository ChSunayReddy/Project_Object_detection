import cv2
import time
import math

# Input video file
video_file = "path_to_video.mp4"  # Replace with your video file path

# Real-world distance calibration (in meters)
real_world_width = 2.0  # Example: width of the scene in meters
frame_width_in_pixels = 640  # Adjust based on your video resolution
pixel_to_meter_ratio = real_world_width / frame_width_in_pixels

# Open the video file
cap = cv2.VideoCapture(video_file)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Define a dictionary to track object positions
object_positions = {}
target_object_id = None  # ID of the specific object to track
next_object_id = 0       # Incremental ID for new objects

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Cannot read the first frame.")
    exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break  # Exit loop if no more frames are available

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track time for speed calculation
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current frame time in seconds

    new_object_positions = {}

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2  # Center point of the bounding box

        # Match the detected object to an existing one or assign a new ID
        matched_id = None
        for obj_id, (prev_cx, prev_cy, _) in object_positions.items():
            distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            if distance < 50:  # Threshold to associate contours
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_object_id
            next_object_id += 1

        new_object_positions[matched_id] = (cx, cy, timestamp)

        # Draw the bounding box
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame2, f"ID: {matched_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # If this is the target object, calculate and display its speed
        if matched_id == target_object_id:
            prev_cx, prev_cy, prev_time = object_positions[matched_id]
            pixel_distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            time_elapsed = timestamp - prev_time

            # Convert pixel distance to meters
            real_distance = pixel_distance * pixel_to_meter_ratio
            speed = real_distance / time_elapsed if time_elapsed > 0 else 0

            # Display speed
            cv2.putText(frame2, f"Speed: {speed:.2f} m/s", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Update object positions for the next frame
    object_positions = new_object_positions

    # Display the resulting frame
    cv2.imshow('Motion Detection with Specific Object Speed', frame2)

    # Set target object ID on mouse click
    def select_target(event, x, y, flags, param):
        global target_object_id
        if event == cv2.EVENT_LBUTTONDOWN:
            for obj_id, (cx, cy, _) in object_positions.items():
                if abs(cx - x) < 20 and abs(cy - y) < 20:  # Click near the object's center
                    target_object_id = obj_id
                    print(f"Target object selected: ID {target_object_id}")
                    break

    cv2.setMouseCallback('Motion Detection with Specific Object Speed', select_target)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
