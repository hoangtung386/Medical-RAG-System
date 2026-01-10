# Medical RAG System (Single-Model Architecture)

This project is a high-performance **Retrieval Augmented Generation (RAG)** application optimized for the medical domain. It utilizes a **Single-Model Architecture** to deliver precise, context-aware medical answers directly in Vietnamese, eliminating the need for intermediate translation layers and significantly reducing latency.

## 🚀 New Architecture: "Direct Vietnamese Processing"

By leveraging state-of-the-art Large Language Models (LLMs) with strong native support for Vietnamese, the system bypasses the traditional "translation bridge" approach (Vi -> En -> Vi). This results in a cleaner pipeline and more natural language generation.

**Simplified 3-Stage Workflow:**

1.  **Retrieval**: Advanced semantic search using **BGE-M3** to locate relevant medical documents.
2.  **Reasoning**: **Gpt-oss 20b bnb 4bit** analyzes the retrieved context and performs medical reasoning directly in Vietnamese.
3.  **Response**: Generation of evidence-based answers with strict source citation.

## 📄 Enhanced Document Processing (New!)

The system now features a robust ingestion pipeline powered by **PaddleOCR** and **PyMuPDF**, capable of handling complex medical documents:

*   **Hybrid OCR Engine**: Automatically detects and extracts text from scanned PDFs and embedded images using **PaddleOCR** (optimized for mixed Vietnamese/English content).
*   **Table Intelligence**: Special handling for medical tables (lab results, dosage charts) to preserve structural integrity during chunking.
*   **Smart Chunking**: Context-aware splitting that respects table boundaries and document sections, ensuring retrieval accuracy.

## 🧠 Core Models

1.  **Medical Logic & Reasoning:**
    *   [**unsloth/gpt-oss-20b-bnb-4bit**](https://huggingface.co/unsloth/gpt-oss-20b-bnb-4bit) (**Current**): A powerful 20B parameter model optimized for 4-bit quantization, offering superior reasoning capabilities while fitting within 16GB VRAM.

2.  **Embedding:** [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
    *   Retained for its State-of-the-Art multimedia and multilingual retrieval performance.

## 🖥️ System Interface

Below are screenshots of the running system:

**1. Login Screen**
Secure access via predefined credentials (`admin` / `123456`).
![Login Interface](/Images/Login_interface.png)



**2. Workspace (Chat Interface)**
The primary interface for medical professionals to query the knowledge base.
![Workspace Interface](/Images/Working_interface.png)

## 📦 Installation & Usage

### 1. Requirements
*   **Python**: 3.10+
*   **Hardware**: NVIDIA GPU with CUDA support (Minimum **16GB VRAM** to load the 20B model).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Ingestion
Place your medical PDF documents into the `Medical_documents/` directory and run:
```bash
python ingest.py
```

### 4. Launch Application
```bash
python app.py
```
*   Access the Web UI at: `http://localhost:7860`

## 📂 Project Structure
*   `app.py`: Main application logic (Single-Model RAG Pipeline).
*   `ingest.py`: Document processing and vector ingestion script.
*   `Medical_documents/`: Directory for input PDF files.
*   `chroma_db/`: Vector database storage (ChromaDB).
*   `Images/`: Interface screenshots.

---
**Medical Disclaimer**: This AI system is for informational and reference purposes only. It is not intended to replace professional medical diagnosis, advice, or treatment. Always consult with a qualified healthcare provider.
