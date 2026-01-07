import os
import gradio as gr
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from threading import Thread
import numpy as np
import logging

# LOGGER SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

# 🎯 SINGLE MODEL APPROACH (Choose ONE)
# Option 1: Gemma 3 27B (RECOMMENDED)
MODEL_ID = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"

# Option 2: Qwen 2.5 32B (Best Vietnamese support)
# MODEL_ID = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

# Option 3: Llama 3.3 70B (Needs more optimization)
# MODEL_ID = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

# RETRIEVAL MODELS
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL = "BAAI/bge-m3"

DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TWEAKABLE PARAMETERS
RELEVANCE_THRESHOLD = 0.3
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2  # Lower for medical accuracy
MAX_CONTEXT_LENGTH = 4096  # Adjust based on model's context window

# SECURITY
DEFAULT_AUTH = ("admin", "123456")

print(f"Device: {DEVICE}")
print(f"Selected Model: {MODEL_ID}")

# --- INITIALIZATION ---

# 1. Load Retriever & Reranker
print("Loading Retrieval System...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE}
)

if os.path.exists(DB_PATH):
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
else:
    logger.warning("Vector DB not found. Run ingest.py!")
    retriever = None

reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)

# 2. Load Main RAG Model (4-bit Quantized)
print(f"Loading RAG Model ({MODEL_ID})...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"  # Enable Flash Attention if available
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("✅ Model loaded successfully.")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

# --- HELPER FUNCTIONS ---

def format_system_prompt(context):
    """
    Creates a medical system prompt in Vietnamese.
    """
    return f"""Bạn là trợ lý y tế AI chuyên nghiệp. Nhiệm vụ của bạn:

1. Trả lời câu hỏi dựa CHÍNH XÁC trên ngữ cảnh được cung cấp
2. Nếu thông tin không có trong ngữ cảnh, hãy nói rõ "Thông tin không có trong tài liệu"
3. KHÔNG bịa đặt thông tin y khoa
4. Trích dẫn nguồn bằng [Nguồn X]
5. Trả lời ngắn gọn, chuyên nghiệp

NGỮ CẢNH:
{context}

LƯU Ý: Đây chỉ là thông tin tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên khoa."""

def truncate_context(context, max_tokens=2048):
    """Truncate context to fit within token limit."""
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        context = tokenizer.decode(tokens)
        logger.warning(f"Context truncated to {max_tokens} tokens")
    return context

def chat(message, history, progress=gr.Progress()):
    """
    Simplified 3-Stage Pipeline:
    1. Retrieve
    2. Reason (Single Model - Vietnamese capable)
    3. Return
    """
    if not retriever:
        yield "❌ Hệ thống chưa có dữ liệu. Vui lòng chạy ingest.py."
        return

    try:
        # STEP 1: Retrieval
        progress(0.2, desc="📚 Đang tìm kiếm tài liệu y khoa...")
        docs = retriever.invoke(message)
        
        if not docs:
            yield "❌ Không tìm thấy tài liệu phù hợp."
            return

        # Rerank
        doc_texts = [d.page_content for d in docs]
        scores = reranker.predict([[message, t] for t in doc_texts])
        top_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > RELEVANCE_THRESHOLD][:TOP_K_RERANK]
        top_docs = [docs[i] for i in top_indices]
        
        if not top_docs:
            yield "❌ Không tìm thấy tài liệu đủ độ tin cậy (confidence threshold not met)."
            return

        # Build Context
        context_parts = []
        sources_list = []
        for i, doc in enumerate(top_docs):
            src = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '?')
            context_parts.append(f"[Nguồn {i+1}] {doc.page_content}\n(File: {src}, Trang: {page})")
            sources_list.append(f"- **[Nguồn {i+1}]** {src} (Trang {page})")
            
        context_str = "\n\n".join(context_parts)
        
        # Truncate context if too long
        context_str = truncate_context(context_str, max_tokens=MAX_CONTEXT_LENGTH - 1024)
        
        # STEP 2: Medical Reasoning
        progress(0.5, desc="🧠 Đang phân tích với AI model...")
        
        system_prompt = format_system_prompt(context_str)
        
        # Construct prompt based on model type
        if "gemma" in MODEL_ID.lower():
            # Gemma format
            full_prompt = f"{system_prompt}\n\nCâu hỏi: {message}\n\nTrả lời:"
        elif "qwen" in MODEL_ID.lower():
            # Qwen format
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Generic format
            full_prompt = f"{system_prompt}\n\nCâu hỏi: {message}\n\nTrả lời:"
        
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH).to(DEVICE)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.15,
        )
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        # Stream response
        response = ""
        for token in streamer:
            response += token
            # Real-time streaming to UI
            yield response + "\n\n⏳ Đang tạo câu trả lời..."
        
        # STEP 3: Final Output with Sources
        final_output = (
            f"{response.strip()}\n\n"
            f"---\n### 📚 Nguồn tham khảo:\n" + "\n".join(sources_list)
        )
        
        yield final_output

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        yield f"❌ Lỗi hệ thống: {str(e)}"

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft(), title="Medical RAG System", fill_height=True) as demo:
    gr.Markdown(
        f"# 🏥 Hệ thống Trợ lý Y khoa AI\n"
        f"**Model:** {MODEL_ID.split('/')[-1]}\n"
        f"**Kiến trúc:** Single-Model RAG (No Translation Bridge)\n"
        f"**GPU:** Tesla P100 16GB"
    )
    
    with gr.Accordion("ℹ️ Lưu ý quan trọng", open=False):
        gr.Markdown(
            "- ✅ **Không còn translation bridge** - câu trả lời chính xác hơn\n"
            "- ⚡ **Tốc độ nhanh hơn** - chỉ 1 model duy nhất\n"
            "- 🇻🇳 **Native Vietnamese** - hiểu tiếng Việt tự nhiên\n"
            "- ⚕️ **Chỉ mang tính tham khảo** - luôn tham khảo bác sĩ"
        )

    gr.ChatInterface(
        fn=chat,
        description="Đặt câu hỏi y khoa bằng tiếng Việt. Hệ thống sẽ tìm kiếm và phân tích tài liệu để trả lời.",
        examples=[
            "Triệu chứng của bệnh tiểu đường type 2 là gì?",
            "Tác dụng phụ của thuốc aspirin?",
            "Làm sao để phòng ngừa bệnh tim mạch?",
            "Chế độ ăn cho người huyết áp cao?"
        ],
        retry_btn="🔄 Thử lại",
        undo_btn="↩️ Hoàn tác",
        clear_btn="🗑️ Xóa hết"
    )
    
    gr.Markdown(
        "\n---\n"
        "**⚠️ Cảnh báo y tế:** Thông tin từ hệ thống chỉ mang tính tham khảo. "
        "Không thay thế cho chẩn đoán và điều trị của bác sĩ chuyên khoa."
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True, 
        server_name="0.0.0.0", 
        auth=DEFAULT_AUTH,
        show_error=True
    )
    