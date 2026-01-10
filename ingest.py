"""
Enhanced Data Ingestion with OCR & Table Support
"""

import os
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
from ocr_processor import EnhancedDocumentProcessor
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = os.path.join(os.getcwd(), "Medical_documents")
DB_PATH = os.path.join(os.getcwd(), "chroma_db")

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_vector_db():
    print("="*60)
    print("Enhanced Medical RAG Ingestion with OCR")
    print("="*60)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Path: {DB_PATH}")
    print("="*60)
    print()
    
    # === STEP 1: Initialize OCR Processor ===
    print("🔧 Initializing OCR processor...")
    use_gpu = torch.cuda.is_available()
    
    # Use 'ch' (Chinese model) for multi-language support (EN + VI)
    # Alternative: 'en' for English only, 'vi' for Vietnamese only
    processor = EnhancedDocumentProcessor(use_gpu=use_gpu, lang='ch')
    
    # === STEP 2: Load and Process Documents ===
    print(f"\nScanning {DATA_PATH} for PDF files...")
    pdf_files = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_PATH}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Process each PDF with OCR
    all_documents = []
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        logger.info(f"\nProcessing: {os.path.basename(pdf_path)}")
        
        try:
            # Extract with OCR support
            pages_data = processor.extract_from_pdf(pdf_path)
            
            # Convert to LangChain Documents
            for page_data in pages_data:
                if page_data['text'].strip():  # Only add non-empty pages
                    doc = Document(
                        page_content=page_data['text'],
                        metadata={
                            'source': pdf_path,
                            'page': page_data['page'],
                            'has_tables': len(page_data['tables']) > 0,
                            'has_images': len(page_data['images_text']) > 0,
                            'method': page_data['method']
                        }
                    )
                    all_documents.append(doc)
            
            logger.info(f"  Extracted {len(pages_data)} pages")
            
        except Exception as e:
            logger.error(f"  Failed to process {pdf_path}: {e}")
            continue
    
    if not all_documents:
        print("No content extracted from documents")
        return
    
    print(f"\nTotal documents extracted: {len(all_documents)}")
    
    # === STEP 3: Smart Text Chunking ===
    print("\nChunking documents...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=[
            "\n=== BẢNG",  # Preserve table boundaries
            "\n=== TABLE",
            "\n=== TEXT CONTENT ===",
            "\n=== OCR EXTRACTED TEXT ===",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],
        length_function=len,
    )
    
    texts = text_splitter.split_documents(all_documents)
    
    # Enrich metadata
    for i, text in enumerate(texts):
        text.metadata["chunk_id"] = i
        # Detect content type
        content = text.page_content.lower()
        if "=== bảng" in content or "=== table" in content:
            text.metadata["content_type"] = "table"
        elif "=== ocr" in content:
            text.metadata["content_type"] = "ocr_text"
        else:
            text.metadata["content_type"] = "standard_text"
    
    print(f"Created {len(texts)} chunks")
    print(f"   Tables detected: {sum(1 for t in texts if t.metadata.get('content_type') == 'table')}")
    print(f"   OCR chunks: {sum(1 for t in texts if t.metadata.get('content_type') == 'ocr_text')}")
    
    # === STEP 4: Create Embeddings ===
    print(f"\nCreating embeddings with {EMBEDDING_MODEL}...")
    print(f"   Device: {DEVICE}")
    print(f"   This may take 10-20 minutes for large datasets...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32 if DEVICE == 'cuda' else 8,
        }
    )
    
    # Test embedding
    try:
        test_embed = embeddings.embed_query("test medical query")
        print(f"Embedding dimension: {len(test_embed)}")
    except Exception as e:
        print(f"Embedding test failed: {e}")
        return
    
    # === STEP 5: Create Vector Database ===
    print(f"\nCreating ChromaDB at {DB_PATH}...")
    
    if os.path.exists(DB_PATH):
        print(f"Existing database found")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(DB_PATH)
            print("Old database deleted")
        else:
            print("Aborting")
            return
    
    # Process in batches
    BATCH_SIZE = 100
    
    try:
        # Initial batch
        db = Chroma.from_documents(
            texts[:BATCH_SIZE],
            embeddings,
            persist_directory=DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Created database with initial {BATCH_SIZE} chunks")
        
        # Remaining batches
        for i in tqdm(range(BATCH_SIZE, len(texts), BATCH_SIZE), desc="Adding batches"):
            batch = texts[i:i+BATCH_SIZE]
            db.add_documents(batch)
        
        print(f"\nVector database created successfully!")
        print(f"Total chunks indexed: {len(texts)}")
        
        # Verify
        print("\nVerifying database...")
        test_results = db.similarity_search("bảng glucose", k=3)
        print(f"Test query returned {len(test_results)} results")
        
        if test_results:
            sample = test_results[0]
            print(f"\nSample result:")
            print(f"   Source: {sample.metadata.get('source', 'Unknown')}")
            print(f"   Page: {sample.metadata.get('page', 'Unknown')}")
            print(f"   Type: {sample.metadata.get('content_type', 'Unknown')}")
            print(f"   Preview: {sample.page_content[:200]}...")
        
    except Exception as e:
        logger.error(f"Database creation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("ENHANCED INGEST COMPLETE!")
    print("="*60)
    print("Features enabled:")
    print("  Standard text extraction")
    print("  OCR for images and scanned pages")
    print("  Table detection and parsing")
    print("  Multi-language support (EN + VI)")
    print("\nNext steps:")
    print("  1. Run: python app.py")
    print("  2. Test with queries about tables and images")
    print("="*60)

if __name__ == "__main__":
    create_vector_db()