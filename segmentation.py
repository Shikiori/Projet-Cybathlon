import cv2
import numpy as np

def nothing(x):
    pass

# Window
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 600, 200)

# Trackbars for color change
# Lower HSV
cv2.createTrackbar("LowerH", "Controls", 0, 179, nothing)
cv2.createTrackbar("LowerS", "Controls", 0, 255, nothing)
cv2.createTrackbar("LowerV", "Controls", 0, 255, nothing)

# Upper HSV
cv2.createTrackbar("UpperH", "Controls", 0, 179, nothing)
cv2.createTrackbar("UpperS", "Controls", 0, 255, nothing)
cv2.createTrackbar("UpperV", "Controls", 0, 255, nothing)

# Placeholder initialisation for the segmented frame
segmented_placeholder = None

# Video capture start
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Initialisation of placeholder if not already done
    if segmented_placeholder is None:
        segmented_placeholder = np.zeros_like(frame)
        cv2.putText(
            segmented_placeholder,
            "No segmented frame yet",
            (50, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    combined = np.hstack((frame, segmented_placeholder))

    # Wait for space bar to segment
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        # Get current positions of trackbars
        lower_h = cv2.getTrackbarPos("Lower H", "Controls")
        lower_s = cv2.getTrackbarPos("Lower S", "Controls")
        lower_v = cv2.getTrackbarPos("Lower V", "Controls")

        upper_h = cv2.getTrackbarPos("Upper H", "Controls")
        upper_s = cv2.getTrackbarPos("Upper S", "Controls")
        upper_v = cv2.getTrackbarPos("Upper V", "Controls")

        # Frame conversion to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Range definition of selected color in HSV
        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
        # Threshold the HSV image to get only selected colors
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Bitwise-AND mask & original image
        segmented = cv2.bitwise_and(frame, frame, mask=mask)

        # Placeholder update w/ new segmented frame
        segmented_placeholder = segmented

        # Placeholder combined w/ original frame for display
        combined = np.hstack((frame, segmented_placeholder))

    cv2.imshow("Original + Segmented", combined)

    # Exit when 'q' pressed
    if key == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
