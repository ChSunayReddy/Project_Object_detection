import cv2
import time
import math

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Real-world distance calibration (in meters)
real_world_width = 2.0  # Example: width of the scene in meters
frame_width_in_pixels = 640  # Example: width of the video frame in pixels
pixel_to_meter_ratio = real_world_width / frame_width_in_pixels

# Read the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# Define a dictionary to track object positions
object_positions = {}

while True:
    ret, frame2 = cap.read()
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Track time for speed calculation
    timestamp = time.time()

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2  # Center point of the bounding box

        # Draw the bounding box
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if the object is already being tracked
        if i in object_positions:
            prev_cx, prev_cy, prev_time = object_positions[i]
            pixel_distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            time_elapsed = timestamp - prev_time

            # Convert pixel distance to meters
            real_distance = pixel_distance * pixel_to_meter_ratio
            speed = real_distance / time_elapsed if time_elapsed > 0 else 0

            # Display speed on the frame
            cv2.putText(frame2, f"Speed: {speed:.2f} m/s", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Update position and time
            object_positions[i] = (cx, cy, timestamp)
        else:
            # Initialize tracking for a new object
            object_positions[i] = (cx, cy, timestamp)

    # Update the previous frame
    gray1 = gray2.copy()

    # Display the resulting frame
    cv2.imshow('Motion Detection with Speed in m/s', frame2)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# import cv2
# import time
# import math

# # Initialize the video capture object
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# # Read the first frame
# ret, frame1 = cap.read()
# gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

# # Define a dictionary to track object positions
# object_positions = {}

# while True:
#     ret, frame2 = cap.read()
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

#     diff = cv2.absdiff(gray1, gray2)
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#     thresh = cv2.dilate(thresh, None, iterations=2)

#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Track time for speed calculation
#     timestamp = time.time()

#     for i, contour in enumerate(contours):
#         if cv2.contourArea(contour) < 500:
#             continue

#         (x, y, w, h) = cv2.boundingRect(contour)
#         cx, cy = x + w // 2, y + h // 2  # Center point of the bounding box

#         # Draw the bounding box
#         cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Check if the object is already being tracked
#         if i in object_positions:
#             prev_cx, prev_cy, prev_time = object_positions[i]
#             distance = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)  # Pixel displacement
#             time_elapsed = timestamp - prev_time

#             # Speed calculation: distance/time
#             speed = distance / time_elapsed if time_elapsed > 0 else 0

#             # Display speed on the frame
#             cv2.putText(frame2, f"Speed: {speed:.2f} px/s", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#             # Update position and time
#             object_positions[i] = (cx, cy, timestamp)
#         else:
#             # Initialize tracking for a new object
#             object_positions[i] = (cx, cy, timestamp)

#     # Update the previous frame
#     gray1 = gray2.copy()

#     # Display the resulting frame
#     cv2.imshow('Motion Detection with Speed Estimation', frame2)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
