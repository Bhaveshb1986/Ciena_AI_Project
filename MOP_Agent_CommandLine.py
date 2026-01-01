import os

from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# from apps.MOP_Agent import BASE_DIR, PDF_PATH


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
    # print(f"BASE_DIR: {BASE_DIR}")
    PDF_PATH = BASE_DIR / "Ciena_Technical_Publication.pdf"

    file = PyPDFLoader(str(PDF_PATH))

    if file is not None:
        pdf_reader = file.load()

        information = ""
        for page in pdf_reader:
            information += page.page_content
    else:
        print("Failed to load the PDF file.")
        return ""
    return information


def get_shelf_type() -> str:
    """
    Takes shelf type input from user with validation.
    Keeps prompting until a valid value is entered.
    """
    while True:
        shelf_type = input("Enter the Shelf Type: ").strip()

        if not shelf_type:
            print(" Shelf type cannot be empty. Please enter a valid value.")
        else:
            return shelf_type


def create_prompt_template() -> PromptTemplate:
    """
    Creates and returns the PromptTemplate object.
    """

    summary_template = """
    Referring the Document {information}, generate Method of Procedure (MOP) for replacing the Access Panel (AP).
    
    INPUTS: {shelf_type}.
    
    SAFETY RULES:
        - Include all applicable CAUTION and DANGER warnings.
        - Do NOT omit safety steps even if repetitive.

    COMPLIANCE RULE:
        - If the document does not explicitly describe a step, do NOT include it.
        - If unsure, state that the document does not provide guidance for that step.
       
    Make the steps precise and to the point.
    """

    return PromptTemplate(
        input_variables=["information", "shelf_type"], template=summary_template
    )


def generate_mop(information: str, shelf_type: str, llm) -> str:
    """
    Invokes the LLM chain and returns the generated response.
    """
    prompt = create_prompt_template()
    chain = prompt | llm

    response = chain.invoke({"information": information, "shelf_type": shelf_type})

    return response.content


def main():
    """
    Application entry point.
    """
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
