import os
import streamlit as st
from pptx import Presentation
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import time
from langchain.llms.base import LLM
from langchain_groq import ChatGroq
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai.client import MistralClient

# Set your Hugging Face token as an environment variable (for Streamlit secrets)
# st.secrets["HF_TOKEN"]

folder_path = "Data"
chroma_path = "chroma"

@st.cache_resource
def load_and_process_data():
    """Loads data, splits it, and saves it to ChromaDB (runs once)."""
    if os.path.exists(chroma_path):
        print(f"ChromaDB already exists at {chroma_path}. Skipping data generation.")
        return Chroma(persist_directory=chroma_path, embedding_function=MistralAIEmbeddings())
    else:
        print("Generating and saving data to ChromaDB...")
        ppt_documents = load_powerpoint_from_folder(folder_path)
        pdf_documents = load_pdf_from_folder(folder_path)
        all_documents = ppt_documents + pdf_documents
        chunks = split_text(all_documents)
        embeddings = MistralAIEmbeddings()
        db = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)
        print(f"Saved {len(chunks)} chunks to {chroma_path}.")
        return db

def load_powerpoint_from_folder(folder_path):
    # ... (your existing load_powerpoint_from_folder function) ...
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".ppt", ".pptx")):
            file_path = os.path.join(folder_path,filename)
            try:
                presentation = Presentation(file_path)
                text = ""
                for slide in presentation.slides:
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for paragraph in shape.text_frame.paragraphs:
                                for run in paragraph.runs:
                                    text += run.text
                                text += "\n"
                    text += "\n\n"
                metadata = {"source": file_path, "file_type": "PowerPoint"}
                documents.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
    return documents

def load_pdf_from_folder(folder_path):
    # ... (your existing load_pdf_from_folder function) ...
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                pdf_documents = loader.load()
                for doc in pdf_documents:
                    doc.metadata["source"] = file_path
                    doc.metadata["file_type"] = "PDF"
                    documents.append(doc)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
    return documents

def split_text(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    st.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def query_data(db, query):
    """Queries the ChromaDB and returns the answer."""
    embeddings = MistralAIEmbeddings()
    retriever = db.as_retriever(search_kwargs={'k': 4})
    relevant_documents = retriever.invoke(query)
    context = "\n\n".join([doc.page_content + f" (Source: {doc.metadata.get('file_type', 'unknown')})" for doc in relevant_documents])

    prompt = f"""You will try to answer the following question based on the provided documents.
Prioritize information found directly within the content of PowerPoint presentations or PDF documents.
If the answer is clearly and sufficiently present in the provided document content, use that information to answer.
Cite the source of your information by mentioning "(from PowerPoint)" or "(from PDF)".

If, after reviewing the document content, you cannot find a direct and complete answer, or if the information is insufficient, then you can use your general knowledge to provide a more comprehensive answer. In this case, do not cite a specific document.

Question: {query}

Document Content:
{context}

Answer: """

    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")
        answer = llm.invoke(prompt)
        return answer.content

    except ImportError:
        st.error("Langchain Groq library not found. Please install it using 'pip install langchain-groq'.")
        return None
    except ValueError as e:
        st.error(f"Error: {e}")
        return None
    except Exception as e:
        st.error(f"Error during Groq API call: {e}")
        return None

def main():
    st.title("Document Query Application")

    # Ensure the 'Data' folder exists
    if not os.path.exists(folder_path):
        st.warning(f"The 'Data' folder does not exist in the current directory. Please create it and place your PPT/PPTX and PDF files there.")
        return

    # Load and process data (or load existing ChromaDB)
    db = load_and_process_data()

    if db:
        query = st.text_input("Enter your question:")
        if query:
            with st.spinner("Processing your query..."):
                answer = query_data(db, query)
            if answer:
                st.subheader("Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()