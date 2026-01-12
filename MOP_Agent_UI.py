# Importing librabies
from dotenv import load_dotenv
import os
import streamlit as st
import re

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
        st.stop()
    return api_key


def load_document() -> str:
    """
    Loads and extracts text content from the PDF file.
    Returns the extracted text as a string.
    """
    BASE_DIR = Path(__file__).parent
    PDF_PATH = BASE_DIR / "Ciena_Technical_Publication.pdf"

    try:
        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        text = ""
        for doc in documents:
            text += doc.page_content

        return text

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
    #chunks=""
    chunks = text_splitter.split_text(pdf_text)

    return chunks


def extract_shelf_type(user_input: str) -> str | None:

    match = re.findall(r"\b(R2|R4|R6|R8)\b", user_input.upper())
    
    # st.write(f"Extracted Shelf Type: {match} \n")
    # st.write(len(match))
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

    #r"\b(R2|R4|R6|R8)\b"
    #r"(?:^|\s)(R2|R4|R6|R8)(?:\s|$)""


def get_user_input() -> str:
    user_input = st.text_input("Kindly enter the valid Shelf Type (R2/R4/R6/R8):")

    if not user_input:
        st.stop()
        return ""

    shelf_type = extract_shelf_type(user_input)

    st.success(f"Generating MOP for shelf type: {shelf_type}")
    return shelf_type


def main():

    VALID_AP_SLOTS = {40}

    st.header(" Method of Procedure (MOP) to Replace the Access Panel(AP)")
    
    # --------------------------------------------------
    # 1. Load PDF Document
    # --------------------------------------------------
    pdf_content = load_document()
    

    # --------------------------------------------------
    # 2. Creating Chunks from PDF content 
    # --------------------------------------------------

    chunks = generate_chunks(pdf_content)
    if not chunks:
        st.error("No content found after chunking.")
        st.stop()


    # --------------------------------------------------
    # 3. Configuration (Getting API keys)
    # --------------------------------------------------
    API_KEY = get_api_key()
    

    # --------------------------------------------------
    # 4. Initialize LLM 
    # --------------------------------------------------
    try:
        llm = ChatOpenAI(
            model="gpt-5-mini", 
            temperature=0, 
            openai_api_key=API_KEY
            )
        
        # Sanity check
        llm.invoke("health check")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        exit(1)

    # --------------------------------------------------
    # 5. Embeddings + Vector Store 
    # --------------------------------------------------
    try:
        embedding = OpenAIEmbeddings(openai_api_key=API_KEY)

        # Creating vector store
        vector_store = FAISS.from_texts(chunks, embedding)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        st.stop

    # --------------------------------------------------
    # 6. User Input - Taking Shelf Type as input
    # --------------------------------------------------
    user_input = get_user_input()
    

    # --------------------------------------------------
    # 7. Retrieval + Generation 
    # --------------------------------------------------

    prompt = (
        f"Generate MOP to replace Access panel slot:{VALID_AP_SLOTS}"
        f"for the given Shelf Type: {user_input}."
        f"Make the steps precise and to the point."
        f'If information is missing, say: "Not specified in document".'
    )
    try:
        match_text = vector_store.similarity_search(user_input)
        
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(
            input_documents=match_text, 
            question=prompt
            )

        st.write(response)
    except Exception as e:
        st.error(f"Failed during retrieval or generation: {e}")
        st.stop()


if __name__ == "__main__":
    main()
