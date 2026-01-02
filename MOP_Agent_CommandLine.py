import os

from pathlib import Path
import re
from urllib import response
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


def get_api_key() -> str:
    """
    Retrieves the OpenAI API key from environment variables.
    Exits the program if the key is not found.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found. Please check your .env file.")
        exit(1)
    return api_key


def read_pdf_content() -> str:
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
        print(f"Failed to load PDF: {e}")
        return None


def get_shelf_type() -> str:
    """
    Takes shelf type input from user with validation.
    Keeps prompting until a valid value is entered.
    """
    while True:
        shelf_type = input("Enter the Shelf Type('e' for exit): ").strip()

        if not shelf_type:
            print(" Shelf type cannot be empty. Please enter a valid value.")
        elif shelf_type.lower() == "e":
            print("Exiting the program.")
            exit(0)

        match = re.findall(r"\b(R2|R4|R6|R8)\b", shelf_type.upper())
        # rint(f"Detected Shelf Types: {match} \n")
        if len(match) == 0:
            print(
                "No Valid Shelf Types detected. Please enter only one from (R2, R4, R6, R8)"
            )
            continue
        elif len(match) > 1:
            print(
                "Multiple Shelf Types detected. Please enter only one from (R2, R4, R6, R8)"
            )
            continue

        return match[0]


def create_prompt_template() -> PromptTemplate:
    """
    Creates and returns the PromptTemplate object.
    """
    VALID_AP_SLOTS = {40}

    summary_template = """
    Generate MOP to replace Access panel slot:{VALID_AP_SLOTS} for the given Shelf Type: {shelf_type}. Make the steps precise and to the point. 
    If information is missing, say: \"Not specified in document\"
    
    Document to Document:  {pdf_content}    
   
    """

    return PromptTemplate(
        input_variables=["pdf_content", "shelf_type", "VALID_AP_SLOTS"],
        template=summary_template,
    )


def generate_mop(pdf_content: str, shelf_type: str, llm) -> str:
    """
    Invokes the LLM chain and returns the generated response.
    """
    prompt = create_prompt_template()
    chain = prompt | llm

    response = chain.invoke(
        {"pdf_content": pdf_content, "shelf_type": shelf_type, "VALID_AP_SLOTS": "{40}"}
    )

    return response.content


def main():

    API_KEY = get_api_key()

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, openai_api_key=API_KEY)

    shelf_type = get_shelf_type()

    pdf_content = read_pdf_content()
    if pdf_content == "":
        print("Exiting due to failure in loading PDF content.")
        return

    mop_output = generate_mop(pdf_content, shelf_type, llm)
    print("\n\n Generated MOP:\n")
    print(mop_output)


if __name__ == "__main__":
    main()
