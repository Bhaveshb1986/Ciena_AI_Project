# Importing librabies
import os
import streamlit as st
import re
from pypdf import PdfReader


from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain


def get_api_key() -> str:
    """
    Retrieves the OpenAI API key from environment variables.
    Exits the program if the key is not found.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.write("OPENAI_API_KEY not found. Please check your .env file.")
        exit(1)
    return api_key


def get_document_from_user() -> str:
    """
    Loads and extracts text content from the PDF file.
    Returns the extracted text as a string.
    """

    with st.sidebar:
        st.title("Your Documents: ")
        file = st.file_uploader("Kindly Upload your PDF file: ", type="pdf")

    if file is None:
        return ""

    try:
        pdf_reader = PdfReader(file)

        information = ""
        for page in pdf_reader.pages:
            information += page.extract_text()
        return information
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        st.stop()
        exit(1)


def generate_chunks(pdf_text: str) -> list:
    # Break the large Text into small chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], 
        chunk_size=1000, 
        chunk_overlap=150, 
        length_function=len
    )

    chunks = text_splitter.split_text(pdf_text)

    return chunks

def extract_shelf_type(user_input: str) -> str | None:

    match = re.findall(r"\b(R2|R4|R6|R8)\b", user_input.upper())
    
    
    if len(match) == 0:
        st.error(
            "No Valid Shelf Types detected. Please enter only one from (R2, R4, R6, R8)"
        )
        st.stop()
    elif len(match) > 1:
        st.error(
            "Multiple Shelf Types detected. Please enter only one from (R2, R4, R6, R8)"
        )
        st.stop()

    return match[0]


def get_user_question() -> str:
    user_input = st.text_input("Kindly enter the valid Shelf Type (R2/R4/R6/R8): ")

    if not user_input:
        st.stop()
        return ""

    shelf_type = extract_shelf_type(user_input)

    st.success(f"Generating MOP for shelf type: {shelf_type}")

    if shelf_type:
        return f"Generate MOP for the given Shelf Type: {shelf_type}. Make the steps precise and to the point."
    else:
        return ""


def main():

    st.header(" Method of Procedure (MOP) to Replace the Access Panel(AP)")

    try:
        # ------------------------------------------------
        # Step 1: Load PDF document uploaded by the user
        # (I/O boundary – may fail if file is invalid)
        # ------------------------------------------------
        pdf_content = get_document_from_user()
        if pdf_content == "":
            st.stop()

        # ------------------------------------------------
        # Step 2: Split PDF text into overlapping chunks
        # (Pre-processing for embeddings & retrieval)
        # ------------------------------------------------
        chunks = generate_chunks(pdf_content)

        # ------------------------------------------------
        # Step 3: Read OpenAI API key from environment
        # ------------------------------------------------
        API_KEY = get_api_key()


        # ------------------------------------------------
        # Step 4: Initialize LLM (GPT)
        # External dependency – errors will bubble up
        # ------------------------------------------------
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0, openai_api_key=API_KEY)


        # ------------------------------------------------
        # Step 5: Create embedding model
        # Converts text chunks into vector representations
        # ------------------------------------------------
        embedding = OpenAIEmbeddings(openai_api_key=API_KEY)


        # ------------------------------------------------
        # Step 6: Build FAISS vector store
        # Enables semantic similarity search (RAG)
        # ------------------------------------------------
        vector_store = FAISS.from_texts(chunks, embedding)


        # ------------------------------------------------
        # Step 7: Read user prompt (shelf type input)
        # ------------------------------------------------
        user_prompt = get_user_question()

        if user_prompt.strip() is not None:

            # ------------------------------------------------
            # Step 8: Retrieve relevant document chunks
            # Using semantic similarity search
            # ------------------------------------------------
            match_text = vector_store.similarity_search(user_prompt)


            # ------------------------------------------------
            # Step 9: Run QA chain using retrieved context
            # 'stuff' → all retrieved chunks passed together
            # ------------------------------------------------
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=match_text, question=user_prompt)

            # ------------------------------------------------
            # Step 10: Display generated MOP output
            # ------------------------------------------------
            st.write(response)

        else:
            # st.warning(" Please provide input: Shelf Type is required.")
            pass

    except Exception as e:
        # ------------------------------------------------
        # Centralized error handling
        # Catches unexpected failures from any stage
        # ------------------------------------------------
        st.write(f"An error occurred: {e}")
        st.stop()



if __name__ == "__main__":
    main()
