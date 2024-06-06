import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import sklearn
from  PyPDF2 import PdfReader
import pdfplumber
import pytesseract
from PIL import Image
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embedding.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
import os


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


def main():
    st.header("📃 Doc GPT")
    st.write("Hello!")
    st.write(" Upload your PDF document & start asking questions...")

    pdf =  st.file_uploader("Accepted file formats: .pdf", type="pdf")

    if pdf is not None:
        pdf_read = PdfReader(pdf)

        text = ""

        for page in pdf_read.pages:
            text += page.extract_text()

        # images = []

        # with pdfplumber.open(pdf) as pdf:
        #     for page in pdf.pages:
        #         text += page.extract_text()
        #         print(text)
        #         for image in page.images:
        #             image = page.to_image().crop(image)
        #             images.append(image)
        #             text += pytesseract.image_to_string(image)

        text_spliiter  = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = text_spliiter.split_text(text= text)

        #embeddings = OpenAIEmbeddings()

        #VectorStore= FAISS.from_texts(text=chunks, embeddings=embeddings)

        query = st.text_input("Start asking questions...")


if __name__ == "__main__":
    main()