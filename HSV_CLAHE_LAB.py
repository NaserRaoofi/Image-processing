#HSV & CLAHE & LAB (Used in this Code)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "E:\\Image processing\\Image-processing\\Top Image\\Messi.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("❌ Error: Could not load the image. Check the file path.")
    exit()

### **✅ 1️⃣ Original Image**
original_image = image.copy()

### **✅ 2️⃣ Histogram Equalization using HSV & CLAHE**
# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Apply CLAHE to the Value (V) Channel
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
v = clahe.apply(v)

# Merge Back and Convert to BGR
hsv_equalized = cv2.merge([h, s, v])
histogram_equalized_image = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

# Normalize Each RGB Channel
b, g, r = cv2.split(histogram_equalized_image)
b = cv2.normalize(b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
g = cv2.normalize(g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
r = cv2.normalize(r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

histogram_equalized_image = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

# Apply a Small Gaussian Blur
histogram_equalized_image = cv2.GaussianBlur(histogram_equalized_image, (1, 1), 0)

### **✅ 3️⃣ Brightness Adjustment using LAB Color Space**
# Convert to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Adjust brightness by modifying L channel
brightness_factor = 1.15  # Use 0.85 for decreasing brightness
l_channel = np.clip(l_channel * brightness_factor, 0, 255).astype(np.uint8)

# Merge LAB channels back together
lab_adjusted_image = cv2.merge([l_channel, a_channel, b_channel])
lab_adjusted_image = cv2.cvtColor(lab_adjusted_image, cv2.COLOR_LAB2BGR)

### **✅ 4️⃣ Compute Histograms for Comparison**
color_channels = ('b', 'g', 'r')  # OpenCV uses BGR order

# Original Image Histogram
original_hist = [cv2.calcHist([original_image], [i], None, [256], [0, 256]) for i in range(3)]
# Histogram Equalized Image Histogram
enhanced_hist = [cv2.calcHist([histogram_equalized_image], [i], None, [256], [0, 256]) for i in range(3)]
# LAB Adjusted Image Histogram
lab_hist = [cv2.calcHist([lab_adjusted_image], [i], None, [256], [0, 256]) for i in range(3)]

### **✅ 5️⃣ Display All Three Images Side-by-Side**
cv2.imshow("Original Image", original_image)
cv2.imshow("Histogram Equalized Image (HSV & CLAHE)", histogram_equalized_image)
cv2.imshow("LAB Adjusted Image", lab_adjusted_image)

### **✅ 6️⃣ Plot Histograms for All Three Images**
plt.figure(figsize=(12, 8))

for i, color in enumerate(color_channels):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Original {color.upper()} Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.plot(original_hist[i], color=color)
    plt.xlim([0, 256])

    plt.subplot(3, 3, i + 4)
    plt.title(f"Histogram Equalized {color.upper()} Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.plot(enhanced_hist[i], color=color)
    plt.xlim([0, 256])

    plt.subplot(3, 3, i + 7)
    plt.title(f"LAB Adjusted {color.upper()} Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.plot(lab_hist[i], color=color)
    plt.xlim([0, 256])

plt.tight_layout()
plt.show()

### **✅ 7️⃣ Wait for a key press and close windows**
cv2.waitKey(0)
cv2.destroyAllWindows()
