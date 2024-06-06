from transformers import pipeline

nlp_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def extract_information(text, question):
    response = nlp_pipeline(question=question, context=text)
    return response['answer']
