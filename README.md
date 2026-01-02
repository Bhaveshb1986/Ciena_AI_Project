Flow diagram 

--MOP_Agent_UI_with_Sidebar.py
PDF Upload (User)
        ↓
Text Extraction
        ↓
Chunking (Recursive Splitter)
        ↓
Embeddings (OpenAI)
        ↓
FAISS Vector Store
        ↓
Similarity Search
        ↓
LLM (QA Chain)
        ↓
MOP Output (Streamlit UI)



--MOP_Agent_UI.py
Technical PDF
      ↓
Text Extraction
      ↓
Chunking (RecursiveCharacterTextSplitter)
      ↓
Embeddings (OpenAI)
      ↓
FAISS Vector Store
      ↓
Shelf Type Validation (Regex)
      ↓
Similarity Search
      ↓
LLM QA Chain
      ↓
MOP Output (Streamlit UI)



-- MOP_Agent_CommandLine.py
Load API Key
     ↓
Read Technical PDF
     ↓
Prompt User for Shelf Type (CLI)
     ↓
Validate Shelf Type (Regex)
     ↓
Create Prompt Template
     ↓
Invoke LLM
     ↓
Print Generated MOP


Prompt:
Generate MOP to replace Access panel slot:{VALID_AP_SLOTS} for the given Shelf Type: {user_input}. 
Make the steps precise and to the point. If information is missing, say: "Not specified in document"

--Supported Inputs
Shelf Types (Validated)
R2
R4
R6
R8

Access Panel Slot
Slot 40 only (hardcoded to avoid hallucinations)



Prerequisites

Python 3.9+
OpenAI API key
Internet access (for LLM invocation)


Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here











Install the Packages:

pip install streamlit pypdf2 langchain faiss-cpu
pip install langchain langchain-text-splitters  langchain-openai langchain-community


Run the streamlit file -- to open chatbox:
streamlit run .\mains.py << please put the filename >>

