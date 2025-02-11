import cv2
import numpy as np

# Load Image
img = cv2.imread("rbc.jpg")  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Otsu’s Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Display Images
cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Threshold", thresh)
cv2.imshow("Contours", contour_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
