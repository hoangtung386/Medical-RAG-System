import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from threading import Thread
import numpy as np
import logging

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UPGRADED CONFIGURATION ---
# ✨ NEW: DeepSeek-R1 with reasoning capabilities
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Reasoning model
# Alternative: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" or "Qwen/Qwen2.5-7B-Instruct"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Keep this - it's excellent

# ✨ NEW: BGE-M3 - SOTA multilingual embedding
EMBEDDING_MODEL = "BAAI/bge-m3"  # Much better than old MiniLM
# Alternative: "Alibaba-NLP/gte-multilingual-base" (lighter)
# Alternative: "intfloat/multilingual-e5-large-instruct" (instruction-tuned)

DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tweakable Parameters
RELEVANCE_THRESHOLD = 0.3
MAX_HISTORY_LEN = 10
MAX_INPUT_LEN = 2000
MIN_INPUT_LEN = 5
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 8
TEMPERATURE = 0.7  # Slightly higher for reasoning models
MAX_NEW_TOKENS = 768  # More tokens for chain-of-thought reasoning

# Security
DEFAULT_AUTH = ("admin", "123456")  # ⚠️ CHANGE THIS!

MEDICAL_DISCLAIMER = """
### ⚠️ CẢNH BÁO Y TẾ QUAN TRỌNG / IMPORTANT MEDICAL DISCLAIMER
1. **Mục đích tham khảo**: Công cụ này chỉ cung cấp thông tin y tế tổng quát để tham khảo, được trích xuất từ tài liệu có sẵn.
2. **Không thay thế bác sĩ**: Thông tin **KHÔNG** có giá trị chẩn đoán, điều trị hay tư vấn y khoa chính thức.
3. **Miễn trừ trách nhiệm**: Người dùng tự chịu trách nhiệm khi sử dụng thông tin. Luôn tham khảo ý kiến bác sĩ hoặc chuyên gia y tế cho các vấn đề sức khỏe cụ thể.

**🆕 Powered by**: DeepSeek-R1 (Reasoning Model) + BGE-M3 (Multilingual Embedding)
"""

print(f"Device: {DEVICE}")

# --- INITIALIZATION ---

# Load Retriever with NEW embedding model
print(f"Loading Vector Database with {EMBEDDING_MODEL}...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE if DEVICE == 'cuda' else 'cpu'}  # Use GPU if available
)

if not os.path.exists(DB_PATH):
    logger.warning(f"Vector DB not found at {DB_PATH}. Please run ingest_upgraded.py first.")
    db = None
else:
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

if db:
    retriever = db.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
else:
    retriever = None

# Load Reranker
print(f"Loading Reranker {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE)

# Load NEW reasoning model
print(f"Loading Reasoning Model {MODEL_ID}...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True  # Required for some models
    )
    
    if DEVICE == "cpu":
        logger.warning("Running on CPU! This will be slow.")
        model.to("cpu")
        
except Exception as e:
    logger.error(f"Error loading {MODEL_ID}: {e}")
    print("Please ensure you have model access and proper hardware.")
    raise e

# --- HELPER FUNCTIONS ---

def validate_input(message):
    """Checks input length and validity."""
    if not message or len(message.strip()) < MIN_INPUT_LEN:
        return "Câu hỏi quá ngắn. Vui lòng nhập chi tiết hơn."
    if len(message) > MAX_INPUT_LEN:
        return f"Câu hỏi quá dài (>{MAX_INPUT_LEN} ký tự). Vui lòng rút gọn."
    return None

def format_prompt_for_reasoning(message, history, context):
    """
    ✨ NEW: Enhanced prompt for reasoning models
    DeepSeek-R1 and similar models benefit from explicit reasoning instructions
    """
    system_prompt = (
        "You are a medical information assistant with advanced reasoning capabilities. Follow these rules:\n\n"
        "**REASONING APPROACH:**\n"
        "1. Think step-by-step through the medical question\n"
        "2. Identify key medical concepts and their relationships\n"
        "3. Cross-reference information from multiple sources when available\n"
        "4. Consider contraindications and safety concerns\n\n"
        
        "**RESPONSE FORMAT:**\n"
        "1. **Language**: Always respond in Vietnamese, even if context is in English\n"
        "2. **Citations**: MUST cite specific sources [Source X] for every medical claim\n"
        "3. **Accuracy**: If sources conflict, present all viewpoints with citations\n"
        "4. **Limitations**: If information is insufficient, clearly state 'Thông tin chưa đầy đủ'\n"
        "5. **Safety**: Never provide specific diagnoses or treatment recommendations\n"
        "6. **Structure**: Use clear sections if explaining complex topics\n\n"
        
        "**CRITICAL SAFETY RULES:**\n"
        "- Always defer to healthcare professionals for diagnosis/treatment\n"
        "- Highlight potential risks and contraindications\n"
        "- Encourage consulting doctors for serious symptoms\n\n"
        
        "Context provided includes numbered sources (e.g., [Source 1], [Source 2])."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history (Limited)
    for human, ai in history[-MAX_HISTORY_LEN:]:
        messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
    
    # Add current message with context
    content_with_context = (
        f"**Medical Context from Documents:**\n{context}\n\n"
        f"**Patient Question:**\n{message}\n\n"
        f"Please provide a thorough, well-reasoned response in Vietnamese."
    )
    messages.append({"role": "user", "content": content_with_context})
    
    return messages

def chat(message, history, progress=gr.Progress()):
    """
    Main chat logic with reasoning model
    """
    error_msg = validate_input(message)
    if error_msg:
        yield error_msg
        return

    if not retriever:
        yield "Lỗi: Cơ sở dữ liệu chưa sẵn sàng. Vui lòng chạy ingest_upgraded.py."
        return

    try:
        progress(0.1, desc="🔍 Đang tìm kiếm tài liệu với BGE-M3...")
        
        # 1. Retrieve with better embeddings
        docs = retriever.invoke(message)
        if not docs:
            yield "Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu."
            return

        progress(0.4, desc="🎯 Đang rerank với Cross-Encoder...")
        
        # 2. Rerank
        doc_texts = [doc.page_content for doc in docs]
        top_docs = []
        
        if doc_texts:
            pairs = [[message, doc_text] for doc_text in doc_texts]
            scores = reranker.predict(pairs)
            
            sorted_indices = np.argsort(scores)[::-1]
            
            top_k_indices = []
            for i in sorted_indices:
                if scores[i] > RELEVANCE_THRESHOLD:
                    top_k_indices.append(i)
                if len(top_k_indices) >= TOP_K_RERANK:
                    break
            
            top_docs = [docs[i] for i in top_k_indices]
        
        if not top_docs:
            yield "Xin lỗi, không tìm thấy thông tin đủ độ tin cậy (>30%) trong tài liệu."
            return

        progress(0.6, desc="🧠 Đang reasoning với DeepSeek-R1...")

        # 3. Context Construction
        context_pieces = []
        sources_list = []
        
        for i, doc in enumerate(top_docs):
            source_path = doc.metadata.get('source', 'Unknown File')
            filename = os.path.basename(source_path)
            
            raw_page = doc.metadata.get('page', -1)
            if isinstance(raw_page, int) and raw_page >= 0:
                page_display = raw_page + 1
            else:
                page_display = "Unknown"
            
            context_pieces.append(
                f"[Source {i+1}]: {doc.page_content}\n"
                f"(Reference: {filename}, Page {page_display})"
            )
            sources_list.append(f"- [Source {i+1}]: {filename} (Page {page_display})")
            
        context = "\n\n".join(context_pieces)
        
        # 4. Generate with Reasoning Model
        messages = format_prompt_for_reasoning(message, history, context)
        
        # Handle different tokenizer formats
        if hasattr(tokenizer, 'apply_chat_template'):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            # Fallback for models without chat template
            prompt_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in messages])
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,  # Add nucleus sampling for better quality
        )
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        partial_response = ""
        for new_token in streamer:
            partial_response += new_token
            yield partial_response

        # 5. Append Sources
        if sources_list and "Tài liệu tham khảo" not in partial_response:
            final_response = (
                partial_response + 
                "\n\n---\n**📚 Tài liệu tham khảo:**\n" + 
                "\n".join(sources_list)
            )
            yield final_response
        else:
            yield partial_response

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        yield f"Đã xảy ra lỗi hệ thống: {str(e)}"

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft(), title="Medical RAG Assistant", fill_height=True) as demo:
    gr.Markdown(
        f"# 🏥 Medical RAG Assistant (Upgraded)\n"
        f"**🧠 Reasoning Model:** {MODEL_ID.split('/')[-1]}\n"
        f"**🔍 Embedding:** {EMBEDDING_MODEL.split('/')[-1]}\n"
        f"**⚡ Pipeline:** Retrieve({TOP_K_RETRIEVAL}) → Rerank({TOP_K_RERANK}) → Reason"
    )
    
    with gr.Accordion("⚠️ ĐỌC KỸ: CẢNH BÁO Y TẾ", open=False):
        gr.Markdown(MEDICAL_DISCLAIMER)
    
    gr.ChatInterface(
        fn=chat,
        description="Hệ thống tra cứu thông tin y tế với khả năng reasoning nâng cao.",
        examples=[
            "Triệu chứng của bệnh tiểu đường type 2 là gì? Giải thích cơ chế sinh lý.",
            "So sánh phương pháp điều trị bệnh tim mạch: thuốc vs can thiệp phẫu thuật?",
            "Aspirin có tác dụng gì? Có tác dụng phụ nào cần lưu ý?",
            "Biến chứng của phẫu thuật thay khớp háng là gì? Tỷ lệ bao nhiêu phần trăm?"
        ],
        fill_height=True,
    )
    
    gr.Markdown(
        "---\n"
        "**💡 Tips:** Hệ thống sử dụng reasoning model - có thể mất thời gian suy nghĩ (~5-15s) "
        "nhưng câu trả lời sẽ có logic và trích dẫn rõ ràng hơn."
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        auth=DEFAULT_AUTH,
        root_path=None
    )
    