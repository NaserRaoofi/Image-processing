import cv2
import numpy as np

# Load the image
image_path = "E:\\Image processing\\Image-processing\\Top Image\\Messi.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("❌ Error: Could not load the image. Check the file path.")
    exit()

# Clone the image for selection
clone = image.copy()
roi = None  # Store selected region

# Variables for mouse callback
drawing = False
x_start, y_start, x_end, y_end = -1, -1, -1, -1

# Mouse callback function
def select_region(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing, roi, clone

    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse click down
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse movement
        if drawing:
            temp_image = clone.copy()
            cv2.rectangle(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Region", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # Mouse release
        x_end, y_end = x, y
        drawing = False

        # Ensure correct region selection (handles dragging in any direction)
        x_start, x_end = min(x_start, x_end), max(x_start, x_end)
        y_start, y_end = min(y_start, y_end), max(y_start, y_end)

        print(f"✅ Selected Region: x_start={x_start}, y_start={y_start}, x_end={x_end}, y_end={y_end}")

        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Adjust brightness in the selected region (Modify L channel)
        brightness_factor = 1.15  # Increase brightness (Use 0.85 to decrease)
        l_channel[y_start:y_end, x_start:x_end] = np.clip(
            l_channel[y_start:y_end, x_start:x_end] * brightness_factor, 0, 255
        ).astype(np.uint8)

        # Merge LAB channels back together
        adjusted_lab = cv2.merge([l_channel, a_channel, b_channel])
        adjusted_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

        # Show the updated image with brightness adjusted in the selected area
        cv2.imshow("Brightness Adjusted Image", adjusted_image)

# Create window and set mouse callback
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", select_region)

# Show the image and wait for selection
cv2.imshow("Select Region", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
