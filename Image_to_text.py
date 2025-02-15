import cv2
import pytesseract

# Specify the Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = "C:\\Users\\gurum\\OneDrive\\Pictures\\Screenshots\\Screenshot 2025-02-15 132040.png"

image = cv2.imread(image_path)

# Debugging print
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract text
text = pytesseract.image_to_string(gray)

# Print the extracted text
print(text)