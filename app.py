import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes  # To convert PDF pages to images
import pytesseract  # For OCR
from PIL import Image  # To handle images
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in environment variables!")
genai.configure(api_key=google_api_key)

# Optional: Set Tesseract path if not in PATH (uncomment and adjust if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_pdf_text(pdf_docs):
    """
    Extract text from PDFs, using OCR for image-only pages if needed.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            pdf_reader = PdfReader(pdf)

            # Try extracting text directly from each page
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # If text is extractable, use it
                    text += page_text
                else:  # If no text, convert to images and use OCR
                    try:
                        # Convert PDF to images (requires Poppler)
                        images = convert_from_bytes(
                            pdf_bytes
                        )  # Add poppler_path=r"C:\path\to\poppler\bin" if PATH fails
                        for img in images:
                            text += pytesseract.image_to_string(img) or ""
                    except Exception as ocr_error:
                        st.warning(
                            f"OCR failed for {pdf.name}: {str(ocr_error)}. Skipping image processing."
                        )
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return text


def get_text_chunks(text):
    """
    Split text into manageable chunks for vector storage.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(chunks):
    """
    Create and save a FAISS vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """
    Set up the conversational chain with a custom prompt.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, respond with "answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", client=genai, temperature=0.3
    )
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)


def clear_chat_history():
    """
    Reset chat history to initial message.
    """
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question!"}
    ]


def user_input(user_question):
    """
    Process user question and return an answer from the vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response
    except Exception as e:
        st.error(f"Error querying vector store: {str(e)}")
        return {"output_text": "Error retrieving response."}


def main():
    """
    Main Streamlit app function.
    """
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖", layout="wide")

    # Sidebar for PDF upload and processing
    st.sidebar.title("📂 Upload PDFs")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF files (text or image-based)",
        accept_multiple_files=True,
        type="pdf",
    )
    if st.sidebar.button("Process PDFs"):
        if not pdf_docs:
            st.sidebar.error("Please upload at least one PDF!")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.sidebar.success("Processing Complete!")
                else:
                    st.sidebar.warning("No text extracted from PDFs.")

    st.sidebar.button("🗑️ Clear Chat History", on_click=clear_chat_history)

    # Main chat interface
    st.title("📄 Chat with Your PDFs")
    st.write("Upload PDFs and start asking questions!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question!"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    user_query = st.chat_input("Ask a question...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(user_query)
                full_response = response.get(
                    "output_text", "Error retrieving response."
                )
                st.write(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
