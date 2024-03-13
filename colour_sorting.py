import cv2
import numpy as np

def nothing(x):
    pass

def find_and_draw_contours(color_range, hsv_frame, original_frame, draw_color, area_threshold):
    lower_color = np.array(color_range[0:3])
    upper_color = np.array(color_range[3:6])
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:
            #cv2.drawContours(original_frame, [contour], -1, draw_color, 2)
            cv2.fillPoly(original_frame, [contour], draw_color)

# Create a window for the controls
cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)

# Create trackbars for the area threshold
cv2.createTrackbar('Area Threshold', 'Controls', 500, 10000, nothing)  # Default value set to 500

# Define color ranges and colors for drawing
color_ranges = {
    'red1': (0, 50, 0, 14, 255, 255),
    'yellow': (15, 45, 0, 44, 255, 255),
    'green': (45, 0, 0, 74, 255, 255),
    'cyan': (75, 0, 0, 104, 255, 255),
    'blue': (105, 0, 0, 134, 255, 255),
    'magenta': (135, 0, 0, 169, 255, 255),
    'red2': (170, 50, 0, 179, 255, 255)
}

draw_colors = {
    'red1': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'cyan': (255, 255, 0),
    'blue': (255, 0, 0),
    'magenta': (255, 0, 255),
    'red2': (0, 0, 255)
}

# Start video capture
cap = cv2.VideoCapture(0)

contoured_frame = np.zeros_like(cap.read()[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check for space bar press
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        contoured_frame = frame.copy()
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the current position of the area threshold trackbar
        area_threshold = cv2.getTrackbarPos('Area Threshold', 'Controls')

        for color, range_values in color_ranges.items():
            find_and_draw_contours(range_values, hsv_frame, contoured_frame, draw_colors[color], area_threshold)

    # Display the original and contoured frames side by side
    combined_frame = np.hstack((frame, contoured_frame))

    cv2.imshow('Live Feed + Contoured', combined_frame)

    if key == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
