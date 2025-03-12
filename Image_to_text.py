import cv2
import pytesseract
import os
from dotenv import load_dotenv
import fitz
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def Extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page  = doc[page_num]
        text = page.get_text("text")

        if text.strip():
            full_text.append(text)
        else:
            image_list = page.get_images(page)
            extracted_text=""

            for img_index,img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(io.BytesIO(image_bytes))
                extracted_text+=pytesseract.image_to_string(img_pil) + "/n"
                full_text.append(extracted_text)
    return("\n".join(full_text))

pdf_file_path = r"C:\Users\gurum\Documents\Judgement 010.pdf"
print(Extract_text_from_pdf(pdf_file_path))