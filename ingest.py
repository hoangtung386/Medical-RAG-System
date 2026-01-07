import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "Medical_documents")
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

# ✨ NEW: BGE-M3 embedding model (much better than old MiniLM)
EMBEDDING_MODEL = "BAAI/bge-m3"
# Alternative: "Alibaba-NLP/gte-multilingual-base" (lighter, still good)
# Alternative: "intfloat/multilingual-e5-large-instruct" (instruction-tuned)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_vector_db():
    print(f"🔍 Loading documents from {DATA_PATH}...")
    
    # Load PDF documents
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")

    # ✨ OPTIMIZED: Better chunking strategy for medical documents
    # BGE-M3 supports up to 8192 tokens, so we can use larger chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased from 2500 (better for semantic coherence)
        chunk_overlap=300,  # Increased overlap to preserve medical context
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence ends
            "? ",    # Question ends
            "! ",    # Exclamation ends
            "; ",    # Semicolon
            ", ",    # Comma
            " ",     # Space
            ""       # Character
        ],
        length_function=len,
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} chunks.")
    
    # Add metadata enrichment (helpful for filtering)
    for i, text in enumerate(texts):
        text.metadata["chunk_id"] = i
        # Preserve original metadata from PDF loader
        
    print(f"🧠 Creating embeddings with {EMBEDDING_MODEL}...")
    print(f"   Device: {DEVICE}")
    print(f"   This may take 5-15 minutes depending on document size...")
    
    # ✨ NEW: BGE-M3 embeddings with GPU support
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            'device': DEVICE,  # Use GPU if available for faster embedding
        },
        encode_kwargs={
            'normalize_embeddings': True,  # Important for cosine similarity
            'batch_size': 32 if DEVICE == 'cuda' else 8,  # Batch processing
        }
    )
    
    # Test embedding (optional but good for debugging)
    try:
        test_embed = embeddings.embed_query("test medical query")
        print(f"✅ Embedding dimension: {len(test_embed)}")
        print(f"   (BGE-M3 produces 1024-dim vectors)")
    except Exception as e:
        print(f"⚠️ Embedding test failed: {e}")
        return

    # Create vector store
    print(f"💾 Creating ChromaDB at {DB_PATH}...")
    print(f"   Processing {len(texts)} chunks...")
    
    # Process in batches to avoid memory issues
    BATCH_SIZE = 100
    
    if os.path.exists(DB_PATH):
        print(f"⚠️ Existing database found at {DB_PATH}")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(DB_PATH)
            print("🗑️ Old database deleted.")
        else:
            print("❌ Aborting. Please backup or delete the old database manually.")
            return
    
    try:
        # Create database with first batch
        db = Chroma.from_documents(
            texts[:BATCH_SIZE],
            embeddings,
            persist_directory=DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"✅ Created initial database with {BATCH_SIZE} chunks")
        
        # Add remaining batches
        for i in range(BATCH_SIZE, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            db.add_documents(batch)
            print(f"✅ Added batch {i//BATCH_SIZE + 1}: chunks {i+1}-{min(i+BATCH_SIZE, len(texts))}")
        
        print(f"🎉 Vector database created successfully!")
        print(f"📊 Total chunks indexed: {len(texts)}")
        print(f"💾 Database location: {DB_PATH}")
        
        # Verify database
        print("\n🔍 Verifying database...")
        test_results = db.similarity_search("diabetes", k=3)
        print(f"✅ Test query returned {len(test_results)} results")
        
        if test_results:
            print("\nSample result:")
            print(f"   Source: {test_results[0].metadata.get('source', 'Unknown')}")
            print(f"   Page: {test_results[0].metadata.get('page', 'Unknown')}")
            print(f"   Preview: {test_results[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✅ INGEST COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Run: python app_upgraded.py")
    print("2. Access the web interface")
    print("3. Test with medical queries")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("🏥 Medical RAG Data Ingestion (UPGRADED)")
    print("="*60)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Path: {DB_PATH}")
    print("="*60)
    print()
    
    create_vector_db()
    