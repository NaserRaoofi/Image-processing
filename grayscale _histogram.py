import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = "E:\\Image processing\\Image-processing\\Top Image\\Histogram.jpg"  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded
if image is None:
    print("❌ Error: Could not load the image. Check the file path.")
    exit()

# ✅ 1️⃣ Standard Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# ✅ 2️⃣ Adaptive Histogram Equalization (AHE) - Not recommended due to excessive noise
ahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8, 8))  # Higher clipLimit can cause over-enhancement
ahe_image = ahe.apply(image)

# ✅ 3️⃣ Contrast Limited Adaptive Histogram Equalization (CLAHE) - Best for natural results
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# ✅ 4️⃣ Gamma Correction (for brightness & contrast control)
gamma = 1.2  # Increase contrast (higher value = brighter, lower = darker)
gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)

# ✅ 5️⃣ Logarithmic Transformation (Enhances dark regions)
log_transformed = np.array(255 * (np.log1p(image) / np.log(256)), dtype=np.uint8)

# ✅ 6️⃣ Compute Histograms for All Methods
original_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])
gamma_hist = cv2.calcHist([gamma_corrected], [0], None, [256], [0, 256])
log_hist = cv2.calcHist([log_transformed], [0], None, [256], [0, 256])

# ✅ 7️⃣ Display Original and Enhanced Images
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image (Global)", equalized_image)
cv2.imshow("CLAHE (Best Local Enhancement)", clahe_image)
cv2.imshow("Gamma Corrected Image", gamma_corrected)
cv2.imshow("Logarithmic Transform", log_transformed)

# ✅ 8️⃣ Plot Histograms for Comparison
plt.figure(figsize=(15, 8))

# 🔹 Original Histogram
plt.subplot(3, 2, 1)
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.plot(original_hist, color="black")
plt.xlim([0, 256])

# 🔹 Histogram Equalization
plt.subplot(3, 2, 2)
plt.title("Equalized Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.plot(equalized_hist, color="blue")
plt.xlim([0, 256])

# 🔹 CLAHE Histogram
plt.subplot(3, 2, 3)
plt.title("CLAHE Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.plot(clahe_hist, color="red")
plt.xlim([0, 256])

# 🔹 Gamma Correction Histogram
plt.subplot(3, 2, 4)
plt.title("Gamma Correction Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.plot(gamma_hist, color="green")
plt.xlim([0, 256])

# 🔹 Log Transform Histogram
plt.subplot(3, 2, 5)
plt.title("Log Transform Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.plot(log_hist, color="purple")
plt.xlim([0, 256])

# Show the plots
plt.tight_layout()
plt.show()

# ✅ 9️⃣ Wait for a key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
