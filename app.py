import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pdfplumber
import pytesseract
from dotenv import load_dotenv
import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import re
from langchain_core.documents import Document
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from streamlit_star_rating import st_star_rating

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

nlp = spacy.load("en_core_web_sm")

languages =["English",
"Chinese",
"Hindi",
"Spanish",
"French",
"Arabic",
"Bengali",
"Russian",
"Portuguese",
"Urdu",
"Indonesian",
"German",
"Japanese",
"Turkish",
"Cantonese",
"Vietnamese"]

loc_languages ={"English":"en",
"Chinese":"zh",
"Hindi":"hi",
"Spanish":"es",
"French":"fr",
"Arabic":"ar",
"Bengali":"bn",
"Russian":"ru",
"Portuguese":"pt",
"Urdu":"ur",
"Indonesian":"id",
"German":"de",
"Japanese":"ja",
"Turkish":"tr",
"Cantonese":"yue",
"Vietnamese":"vi"}

ner = pipeline("ner")

relationship_extractor = pipeline("text-classification", model="dslim/bert-base-NER")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

with st.sidebar:
    st.title("Doc GPT")
    st.markdown('''
    ### How to use Doc GPT?

    - Upload any PDF document
    - Get summary of the document
    - Ask question to retriev specific information from the uploaded document
    ''')
    add_vertical_space(5)
    st.write("Developed by Chanchala Gorale . 2024")


def pdf_to_text_and_images(pdf_path):
    text_content = ""
    images = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            text_content += page.extract_text()
           
            for img in page.images:
                img_obj = page.to_image().original.crop((img['x0'], img['top'], img['x1'], img['bottom']))
                text_content += pytesseract.image_to_string(img_obj)
                

    return text_content


def extract_key_value_pairs(text):
    # A simple regex-based approach for key-value pair extraction
    key_value_pattern = re.compile(r'(\b\w+):\s*(.+)')
    key_value_pairs = dict(re.findall(key_value_pattern, text))
    return key_value_pairs

def clean_text(text):
    # Remove special characters, punctuation, and formatting
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Normalize to lowercase
    text = text.lower()

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# Sentence Segmentation
def segment_sentences(text):
    return sent_tokenize(text)

# Tokenization
def tokenize_text(text):
    return word_tokenize(text)

def find_entities(text):
    doc = nlp(text)

    entities={}
 
    for ent in doc.ents:
        entities[ent.label_] = ent.text

    return entities

def translate_to_english(text, target_lang):
    lang = detect(text)
    if lang == 'en':
        return text
    model_name = f'Helsinki-NLP/opus-mt-{lang}-{loc_languages[target_lang]}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

def preprocess_text(text, max_length=1024):
    sentences = text.split('. ')
    processed_text = ""
    for sentence in sentences:
        if len(processed_text.split()) + len(sentence.split()) > max_length:
            break
        processed_text += sentence + '. '
    return processed_text.strip()


def main():
    load_dotenv()
    
    st.header("📃 Doc GPT")

    st.write("Hello!")
    st.write("Upload your PDF file & start ask question...")
 
    # upload a PDF file
    pdf = st.file_uploader("Only PDF files accepted.", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:

        #extract text
        unclean_text = pdf_to_text_and_images(pdf)
        
        text=unclean_text

        if text:
            text = clean_text(text)

        #summarize text
        summarizer = pipeline("summarization" ,model="facebook/bart-large-cnn")
        summary = summarizer(preprocess_text(text), max_length=150, min_length=30, do_sample=False)

        st.title("Summary:")
        st.write(summary[0]['summary_text'])

        # tokenize
        #sentences = segment_sentences(text)
        #tokens = [tokenize_text(sentence) for sentence in sentences]
        

        #find entities text
        entities=find_entities(text)
        if entities:
            st.write(entities)

        #key value extraction
        key_value_pairs = extract_key_value_pairs(text)
        if key_value_pairs:
            st.write(key_value_pairs)

        #ask if use need to traslate the information
        target_language = st.selectbox("Choose language to traslate document", languages)

        if st.button("Translate"):
            translation = translate_to_english(text,  target_language)
            st.write(translation)


        #query 
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            )
            chunks = text_splitter.split_text(text=unclean_text)
            embeddings = OpenAIEmbeddings()
            VectorStore =FAISS.from_texts(chunks, embedding=embeddings)

            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            if response:
                st.write(response)
            else:
                st.write("Answer not Found! We suggest you upload relevant file to get the answer to this question.")

        #collect feedback 

        stars = st_star_rating("Please rate you experience", maxValue=5, defaultValue=0, key="rating",  )
        st.write(stars)

        ######## Thats all!

  

if __name__ == "__main__":
    main()