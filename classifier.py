from transformers import pipeline

classifier = pipeline("zero-shot-classification")

def classify_document(text):
    labels = ["invoice", "receipt", "other"]
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0]
