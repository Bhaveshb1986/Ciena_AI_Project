# Importing librabies
import os
from turtle import width
import streamlit as st
from pypdf import PdfReader
import re

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains.question_answering import load_qa_chain
from mermaid import Mermaid

from openai import OpenAI
import base64


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

    pdf_reader = PdfReader(file)

    information = ""
    for page in pdf_reader.pages:
        information += page.extract_text()
    return information


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

    st.success(f"Generating MOP Diagram for Selected Shelf Type: {shelf_type}")

    if shelf_type:
        prompt = (
            f"Generate MOP to replace Access panel slot: 40"
            f"for the given Shelf Type: {user_input}."
            f"Make the steps precise and to the point."
            f'If information is missing, say: "Not specified in document".'
        )
        
        return prompt
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
        llm = ChatOpenAI(
            model="gpt-5-mini", 
            temperature=0, 
            openai_api_key=API_KEY
            )
        

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
        user_text = get_user_question()

        if user_text.strip() is not None:
            
            # ------------------------------------------------
            # Step 8: Retrieve relevant document chunks
            # Using semantic similarity search
            # ------------------------------------------------
            match = vector_store.similarity_search(user_text)


            # ------------------------------------------------
            # Step 9: Run QA chain using retrieved context
            # 'stuff' → all retrieved chunks passed together
            # ------------------------------------------------
            chain = load_qa_chain(llm, chain_type="stuff")   
            response = chain.run(input_documents=match, question=user_text)
          


            # ------------------------------------------------
            # Step 10: Prepare a prompt for diagram generation
            # ------------------------------------------------
            # - Convert the textual MOP response into a visual form (can be 1 of these)
            # - flowchart diagram
            # - Summary diagram
            # - Sequence diagram
            # ------------------------------------------------
            visual_prompt = f"""
            Create a clean technical flowchart diagram based on the following procedure:

            {response}

            Use telecom equipment maintenance style, clear boxes, arrows, light blue background.
            """


            # ------------------------------------------------
            # Step 11: Initialize OpenAI Image Generation Client
            # ------------------------------------------------
            # - This client is used only for image generation
            # - API key is reused from the application environment
            # ------------------------------------------------
            client = OpenAI(api_key=API_KEY)



            # ------------------------------------------------
            # Step 12: Generate diagram image using OpenAI image model
            # ------------------------------------------------
            # - model="gpt-image-1" is used for diagram-style images
            # - size="auto" allows the model to choose best dimensions
            # ------------------------------------------------
            img = client.images.generate(
                model="gpt-image-1",
                prompt=visual_prompt,
                size="auto"    # Supported values are: '1024x1024', '1024x1536', '1536x1024', and 'auto'."
            )

            # ------------------------------------------------
            # Step 13: Decode Base64 image data to binary format
            # ------------------------------------------------
            image_bytes = base64.b64decode(img.data[0].b64_json)


            # ------------------------------------------------
            # Step 14: Display the AI-generated diagram in Streamlit UI
            # ------------------------------------------------
            #st.subheader("AI Generated Diagram")
            
            st.image(image_bytes, width=1200)


        else:
            # st.warning(" Please provide input: Shelf Type is required.")
            pass

    except Exception as e:
        # ------------------------------------------------
        # Centralized error handling
        # Catches unexpected failures from any stage
        # ------------------------------------------------
        st.error(f"An error occurred: {e}") 
        st.stop

if __name__ == "__main__":
    main()
