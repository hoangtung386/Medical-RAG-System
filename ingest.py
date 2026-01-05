import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "Medical_documents")
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

def create_vector_db():
    print(f"Loading documents from {DATA_PATH}...")
    
    # Load PDF documents
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # Create embeddings
    print("Creating embeddings with Multilingual model...")
    # Using a multilingual model to support Vietnamese queries against English docs
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                       model_kwargs={'device': 'cpu'}) # Use CPU for embeddings to save VRAM for the LLM

    # Create vector store
    print(f"Creating ChromaDB at {DB_PATH}...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    print("Vector database created successfully.")

if __name__ == "__main__":
    create_vector_db()
