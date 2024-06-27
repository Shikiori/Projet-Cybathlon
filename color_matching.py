import cv2
import numpy as np

# Detect largest rectangle in frame
def get_largest_contour(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find contourst
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Returns avg hsv value within given contour
def calculate_average_hsv(frame, contour):
    # Create mask for the contour
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    # Calculate average HSV value within the contour
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = hsv_frame[mask[:,:,1] > 0]
    average_hsv = np.mean(hsv_values, axis=0)
    return average_hsv

# Match color seen on webcam to stored values
def match_color(average_hsv, stored_hsv_values):
    # Calculate the difference between average HSV value and stored values
    differences = np.linalg.norm(stored_hsv_values - average_hsv, axis=1)
    # Find the index with the smallest difference
    match_index = np.argmin(differences)
    return match_index

def sort_order(arr):
    # Enumerate the array to keep track of original indices
    indexed_arr = [(val, idx) for idx, val in enumerate(arr)]
    
    # Sort the array
    sorted_arr = sorted(indexed_arr)
    
    # Get the indices of the sorted elements
    sorted_indices = [pair[1] + 1 for pair in sorted_arr]
    
    return sorted_indices

# Array for storing reference colors (hsv values)
stored_hsv_values = [[0 for _ in range(3)] for _ in range(5)]   # 5 couleurs, 3 nuances par couleur

# Capture video from webcam
print("Début de la capture des références. \n Appuyer sur Entrée pour capturer la première nuance.\n")

cap = cv2.VideoCapture(0)

# Store reference color values
for i in range(5):
    print("Couleur n°" + str(i+1) + "\n")
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
                average_hsv = calculate_average_hsv(frame, largest_contour)
                # Store average HSV value
                stored_hsv_values[i][j] = average_hsv
                print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la nuance suivante.")
                break
            elif key == ord('q'):  # Press 'q' to quit capturing
                break
            else:
                continue

        # Check if 'q' is pressed to quit capturing
        if key == ord('q'):
            exit()
    else:  # This else block will only execute if the inner loop completes without breaking
        continue
    break  # Break out of the outer loop if any key other than Enter is pressed


# Array for storing matched reference color indices for each frame
matched_indices = []

print("\nRéférences capturées. \nAppuyez sur Entrée pour capturer une couleur. \n")

# Wait for Enter key press to start capturing frames
for i in range(6):
    print("Couleur n°" + str(i+1))
    while True:
        ret, frame = cap.read()
        cv2.imshow("Frame", frame)
        
        # Check for Enter key press to start capturing frames
        key = cv2.waitKey(1)
        if key == 13:  # Enter key code is 13
            # Get largest contour and calculate average HSV value
            largest_contour = get_largest_contour(frame)
            average_hsv = calculate_average_hsv(frame, largest_contour)
            
            # Match the average HSV value to stored reference values
            match_index = match_color(average_hsv, stored_hsv_values)
            matched_indices.append(match_index)
            print("Capture OK. Appuyer à nouveau sur Entrée pour capturer la couleur suivante.\n")
        elif key == ord('q'):  # Press 'q' to quit capturing
            exit()
        else:
            continue
        break
    
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print matched indices
print(matched_indices)
#print("Ordre des couleurs observées à l'écran :")
#print(sort_order(matched_indices))