import cv2
import numpy as np

def get_largest_contour(frame):
    """ Detect largest rectangle in frame """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def calculate_average_hsv(frame, contour):
    """ Returns avg HSV value within given contour """
    # Create mask for the contour
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    # Calculate average HSV value within the contour
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_frame[mask[:,:,1] > 0]
    average_hsv = np.mean(hsv_values, axis=0)
    return average_hsv

def match_color(average_hsv, references):
    """ Match color seen on webcam to stored values """
    # Calculate the Euclidean distance between the observed HSV and each stored reference HSV
    distances = [np.linalg.norm(average_hsv - ref_hsv) for ref_hsv in references]
    # Find the index of the smallest distance
    match_index = np.argmin(distances)
    return match_index

# Array for storing HSV values of reference colors
references = [0 for _ in range(3)] 

# Capture video from webcam
print("Début de la capture des références. \nAppuyer sur Entrée pour capturer la première nuance.\n")

cap = cv2.VideoCapture(0)

for j in range(3):
    print("Nuance n°" + str(j+1))
    while True:
        # Capture & display frame
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)

        # Wait for Enter key press
        key = cv2.waitKey(1)
        if key == 13:  # Enter key code is 13
            # Get largest contour and calculate average HSV value
            largest_contour = get_largest_contour(frame)
            if largest_contour is not None:
                average_hsv = calculate_average_hsv(frame, largest_contour)
                print("avg HSV=", average_hsv)
                # Store average HSV value
                references[j] = average_hsv
                print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la nuance suivante.")
                break
        elif key == ord('q'):  # Press 'q' to quit capturing
            break

    # Check if 'q' is pressed to quit capturing
    if key == ord('q'):
        exit()

# Array for storing matched reference color indices for each frame
matched_indices = []

print("\nRéférences capturées. \nAppuyez sur Entrée pour capturer une couleur.\n")

# Wait for Enter key press to start capturing frames
for i in range(3):
    print("Couleur n°" + str(i+1))
    while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)

        # Check for Enter key press to start capturing frames
        key = cv2.waitKey(1)
        if key == 13:  # Enter key code is 13
            # Get largest contour and calculate average HSV value
            largest_contour = get_largest_contour(frame)
            if largest_contour is not None:
                average_hsv = calculate_average_hsv(frame, largest_contour)
                # Match the average HSV value to stored reference values
                match_index = match_color(average_hsv, references)
                matched_indices.append(match_index)
                print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la couleur suivante.\n")
            break
        elif key == ord('q'):  # Press 'q' to quit capturing
            exit()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print matched indices
print(references)
print(matched_indices)
