import pdfplumber
import pytesseract
from PIL import Image

def pdf_to_text_and_images(pdf_path):
    text_content = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text()
            for image in page.images:
                image = page.to_image().crop(image)
                images.append(image)

    return text_content, images

def ocr_image(image):
    return pytesseract.image_to_string(image)
