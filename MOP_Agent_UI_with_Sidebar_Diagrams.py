# Importing librabies
import os
from turtle import width
import streamlit as st
from pypdf import PdfReader


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
        separators="\n", chunk_size=1000, chunk_overlap=150, length_function=len
    )

    chunks = text_splitter.split_text(pdf_text)

    return chunks


def get_user_question() -> str:
    user_question = st.text_input("Kindly provide your Prompt containing valid shelf type (R2/R4/R6/R8): ")

    if user_question:
        return f"Generate MOP for the given Shelf Type: {user_question}. Make the steps precise and to the point."
    else:
        return ""


def main():

    st.header(" Method of Procedure (MOP) to Replace the Access Panel(AP)")
    pdf_content = get_document_from_user()
    if pdf_content == "":
        return

    chunks = generate_chunks(pdf_content)

    API_KEY = get_api_key()
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, openai_api_key=API_KEY)
    #st.write(llm)

    # Generating embedding
    embedding = OpenAIEmbeddings(openai_api_key=API_KEY)

    # Creating vector store
    vector_store = FAISS.from_texts(chunks, embedding)

    user_question = get_user_question()

    if user_question.strip() is not None:
        match = vector_store.similarity_search(user_question)

        chain = load_qa_chain(llm, chain_type="stuff")   #chain_type="stuff": Concatenate all retrieved documents into one big prompt.
        response = chain.run(input_documents=match, question=user_question)

        #st.subheader("MOP Steps")
        #st.write(response)

        visual_prompt = f"""
        Create a clean technical Summary diagram based on the following procedure:

        {response}

        Use telecom equipment maintenance style, clear boxes, arrows, light blue background.
        """
        #flowchart diagram
        #Summary diagram
        #Sequence diagram

        client = OpenAI(api_key=API_KEY)

        img = client.images.generate(
            model="gpt-image-1",
            prompt=visual_prompt,
            size="auto"    # Supported values are: '1024x1024', '1024x1536', '1536x1024', and 'auto'."
        )

        image_bytes = base64.b64decode(img.data[0].b64_json)

        st.subheader("AI Generated Diagram")
        
        #width = st.slider("Diagram width", 600, 1400, 1200) #Implementing Slider for width adjustment
        #st.image(image_bytes, width=width)
        st.image(image_bytes, width=1200)


    else:
        # st.warning(" Please provide input: Shelf Type is required.")
        pass


if __name__ == "__main__":
    main()
