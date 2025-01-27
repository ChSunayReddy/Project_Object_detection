import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Read the first frame
ret, frame1 = cap.read()
# Convert the frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise and detail
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    # Read the next frame
    ret, frame2 = cap.read()
    # Convert the frame to grayscale
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

    # Compute the absolute difference between the current frame and the first frame
    diff = cv2.absdiff(gray1, gray2)
    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    # Dilate the thresholded image to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours to reduce false positives
        if cv2.contourArea(contour) < 500:
            continue
        # Get the bounding box coordinates of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw a rectangle around the detected motion
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame2)

    # Update the previous frame
    gray1 = gray2.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
