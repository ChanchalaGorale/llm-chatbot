import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import sklearn
import  PyPDF2 
import pdfplumber
import pytesseract
from PIL import Image
import cv2
from dotenv import load_dotenv
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import pipeline

import re

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

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
        text = pdf_to_text_and_images(pdf)
        key_value_pairs = extract_key_value_pairs(text)


        #summarieze text
        summarizer = pipeline("summarization")
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

        st.title("Summary:")
        st.write(summary[0]['summary_text'])
        st.write(key_value_pairs)
 
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        )
        
        chunks = text_splitter.split_text(text=text)
 
        st.write(chunks)

        # # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        # st.write(chunks)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # if os.path.exists(f"{store_name}.pkl"):
        #     try:
        #         with open(f"{store_name}.pkl", "rb") as f:
        #             VectorStore = pickle.load(f)

        #     except (EOFError, FileNotFoundError) as e:
        #         embeddings = OpenAIEmbeddings()
        #         VectorStore = FAISS.from_texts(chunks, embedding=embeddings).docstore._dict

                
        #         with open(f"{store_name}.pkl", "wb") as f:
        #             pickle.dump(VectorStore, f)

        # else:
        #     embeddings = OpenAIEmbeddings()
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings).docstore._dict
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)
 
  
 
        # # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == "__main__":
    main()