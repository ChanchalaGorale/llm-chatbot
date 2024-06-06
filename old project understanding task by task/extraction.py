def extract_key_values(text, images):
    key_values = {}
    key_questions = {
        "Invoice Number": "What is the invoice number?",
        "Invoice Date": "What is the invoice date?",
        "Total Amount": "What is the total amount?"
    }

    for key, question in key_questions.items():
        key_values[key] = extract_information(text, question)

    for image in images:
        image_text = ocr_image(image)
        for key, question in key_questions.items():
            if key not in key_values or not key_values[key]:
                key_values[key] = extract_information(image_text, question)

    return key_values
