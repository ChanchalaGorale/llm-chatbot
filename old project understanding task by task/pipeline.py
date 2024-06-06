def process_document(file_path):
    text, images = pdf_to_text_and_images(file_path)
    key_values = extract_key_values(text, images)
    classification = classify_document(text)
    # Optionally translate content if needed
    translated_text = translate_text(text, src_lang="en", tgt_lang="fr")
    return {
        "classification": classification,
        "key_values": key_values,
        "translated_text": translated_text
    }

@app.post("/process/")
async def process_file(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = process_document(file_path)
    return result
