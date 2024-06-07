import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import sklearn
import  PyPDF2 
import pdfplumber
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import pipeline
from pdf2image import convert_from_path
import re

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
            for image in page.images:
                image = page.to_image().crop(image)
                text_content += pytesseract.image_to_string(image)

    return text_content


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)

    text = ""
    for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def extract_images_and_text_from_pdf(file):
    images = convert_from_path(file)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def extract_key_value_pairs(text):
    # A simple regex-based approach for key-value pair extraction
    key_value_pattern = re.compile(r'(\b\w+):\s*(.+)')
    key_value_pairs = dict(re.findall(key_value_pattern, text))
    return key_value_pairs

def main():
    st.header("📃 Doc GPT")

    st.write("Hello!")
    st.write("Upload your PDF file & start ask question...")
 
    # upload a PDF file
    pdf = st.file_uploader("Only PDF files accepted.", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        
        #extract text
        text1 = pdf_to_text_and_images(pdf)
        ocr_text ="" # extract_images_and_text_from_pdf(os.path.join("tempDir", pdf.name))
        text = text1 + "\n" + ocr_text

        st.write("Text extracted!")
        st.write(text)

        key_value_pairs = extract_key_value_pairs(text)

        st.write("key_value_pairs extracted!")
        st.write(key_value_pairs)

        #summarieze text
        # summarizer = pipeline("summarization")
        # summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

        # st.title("Summary:")
        # st.write(summary[0]['summary_text'])
        # st.write(key_value_pairs)
 
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        #     )
        
        # chunks = text_splitter.split_text(text=text)
 
        # # # embeddings
        # store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        # # st.write(chunks)
 
        # if os.path.exists(f"{store_name}.pkl"):
        #     try:
        #         with open(f"{store_name}.pkl", "rb") as f:
        #             VectorStore = pickle.load(f)

        #     except (EOFError, FileNotFoundError) as e:
        #         embeddings = OpenAIEmbeddings()
        #         VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #         with open(f"{store_name}.pkl", "wb") as f:
        #             pickle.dump(VectorStore, f)
        #     # st.write('Embeddings Loaded from the Disk')s
        # else:
        #     embeddings = OpenAIEmbeddings()
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)
 
  
 
        # # Accept user questions/query
        # query = st.text_input("Ask questions about your PDF file:")
        # # st.write(query)
 
        # if query:
        #     docs = VectorStore.similarity_search(query=query, k=3)
 
        #     llm = OpenAI()
        #     chain = load_qa_chain(llm=llm, chain_type="stuff")
        #     with get_openai_callback() as cb:
        #         response = chain.run(input_documents=docs, question=query)
        #         print(cb)
        #     st.write(response)

if __name__ == "__main__":
    main()