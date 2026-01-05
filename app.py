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

# LOGGER SETUP
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UPGRADED CONFIGURATION
# NEW: Ministral-3-8B-Reasoning - MUCH better Vietnamese support!
MODEL_ID = "mistralai/Ministral-3-8B-Reasoning-2512"  
# This model has excellent multilingual capabilities and follows system prompts strictly

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL = "BAAI/bge-m3"  # Keep the upgraded embedding

DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tweakable Parameters
RELEVANCE_THRESHOLD = 0.3
MAX_HISTORY_LEN = 10
MAX_INPUT_LEN = 2000
MIN_INPUT_LEN = 5
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 8
TEMPERATURE = 0.7  # Mistral recommends 0.7 for reasoning models
MAX_NEW_TOKENS = 768

# Security
DEFAULT_AUTH = ("admin", "123456")  # ⚠️ CHANGE THIS!

MEDICAL_DISCLAIMER = """
### CẢNH BÁO Y TẾ QUAN TRỌNG
1. **Mục đích tham khảo**: Công cụ này chỉ cung cấp thông tin y tế tổng quát để tham khảo.
2. **Không thay thế bác sĩ**: Thông tin **KHÔNG** có giá trị chẩn đoán, điều trị hay tư vấn y khoa.
3. **Miễn trừ trách nhiệm**: Người dùng tự chịu trách nhiệm khi sử dụng thông tin. Luôn tham khảo ý kiến bác sĩ.

Powered by: Ministral-3-8B-Reasoning (Multilingual + Reasoning) + BGE-M3 (SOTA Embedding)
"""

print(f"Device: {DEVICE}")

# INITIALIZATION

# Load Retriever with BGE-M3
print(f"Loading Vector Database with {EMBEDDING_MODEL}...")
embedding_function = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': DEVICE if DEVICE == 'cuda' else 'cpu'}
)

if not os.path.exists(DB_PATH):
    logger.warning(f"Vector DB not found at {DB_PATH}. Please run ingest.py first.")
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

# Load Ministral-3-8B-Reasoning Model
print(f"Loading Reasoning Model {MODEL_ID}...")

try:
    # Ministral supports 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True  # Extra optimization
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Important: Set pad_token if not set (Ministral sometimes needs this)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    if DEVICE == "cpu":
        logger.warning("Running on CPU! This will be slow.")
        model.to("cpu")
    
    print("Model loaded successfully!")
        
except Exception as e:
    logger.error(f"Error loading {MODEL_ID}: {e}")
    raise e

# HELPER FUNCTIONS

def validate_input(message):
    """Checks input length and validity."""
    if not message or len(message.strip()) < MIN_INPUT_LEN:
        return "Câu hỏi quá ngắn. Vui lòng nhập chi tiết hơn."
    if len(message) > MAX_INPUT_LEN:
        return f"Câu hỏi quá dài (>{MAX_INPUT_LEN} ký tự). Vui lòng rút gọn."
    return None

def format_prompt_for_ministral(message, history, context):
    """
    OPTIMIZED PROMPT for Ministral-3-8B-Reasoning
    
    Key changes from DeepSeek-R1 version:
    1. Stricter language control (MUST respond in Vietnamese)
    2. Simpler instructions (Ministral is smaller, less verbose)
    3. Emphasize system prompt adherence (Ministral's strength)
    """
    
    # Ministral recommends concise system prompts
    system_prompt = (
        "You are a medical information assistant. You MUST follow these rules:\n\n"
        
        "**CRITICAL - LANGUAGE RULE:**\n"
        "- Your ENTIRE response MUST be in Vietnamese only\n"
        "- Never mix English, French, or other languages in your answer\n"
        "- Translate all medical terms to Vietnamese\n"
        "- If you don't know the Vietnamese term, describe it in Vietnamese\n\n"
        
        "**RESPONSE STRUCTURE:**\n"
        "1. Answer the question directly in Vietnamese\n"
        "2. Cite sources using [Source X] format for every claim\n"
        "3. If sources conflict, present all viewpoints\n"
        "4. If information is insufficient, say 'Thông tin chưa đầy đủ'\n\n"
        
        "**SAFETY:**\n"
        "- Never provide diagnosis or treatment recommendations\n"
        "- Always encourage consulting healthcare professionals\n"
        "- Mention risks and contraindications when relevant\n\n"
        
        "Context includes numbered sources: [Source 1], [Source 2], etc."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history (limited)
    for human, ai in history[-MAX_HISTORY_LEN:]:
        messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
    
    # Add current message with context
    # Important: Remind the model again about Vietnamese
    content_with_context = (
        f"**Tài liệu y khoa (Medical Context):**\n{context}\n\n"
        f"**Câu hỏi (Question):**\n{message}\n\n"
        f"**QUAN TRỌNG:** Trả lời HOÀN TOÀN bằng tiếng Việt. Không lẫn lộn ngôn ngữ khác."
    )
    messages.append({"role": "user", "content": content_with_context})
    
    return messages

def chat(message, history, progress=gr.Progress()):
    """
    Main chat logic with Ministral-3-8B-Reasoning
    """
    error_msg = validate_input(message)
    if error_msg:
        yield error_msg
        return

    if not retriever:
        yield "Lỗi: Cơ sở dữ liệu chưa sẵn sàng. Vui lòng chạy ingest.py trước."
        return

    try:
        progress(0.1, desc="Đang tìm kiếm tài liệu...")
        
        # 1. Retrieve
        docs = retriever.invoke(message)
        if not docs:
            yield "Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu."
            return

        progress(0.4, desc="Đang đánh giá độ liên quan...")
        
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
            yield "Xin lỗi, không tìm thấy thông tin đủ độ tin cậy (>30%) để trả lời."
            return

        progress(0.6, desc="Đang suy luận với Ministral...")

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
                f"(Tài liệu: {filename}, Trang {page_display})"
            )
            sources_list.append(f"- [Source {i+1}]: {filename} (Trang {page_display})")
            
        context = "\n\n".join(context_pieces)
        
        # 4. Generate with Ministral
        messages = format_prompt_for_ministral(message, history, context)
        
        # Tokenize with Ministral's chat template
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception as e:
            # Fallback if chat template fails
            logger.warning(f"Chat template failed: {e}. Using manual formatting.")
            prompt_text = "\n\n".join([
                f"{'System' if m['role']=='system' else m['role'].capitalize()}: {m['content']}" 
                for m in messages
            ]) + "\n\nAssistant:"
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
            top_p=0.9,
            repetition_penalty=1.1,  # Prevent repetition
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
        f"# Medical RAG Assistant\n"
        f"**Model:** Ministral-3-8B-Reasoning (Multilingual + Reasoning)\n"
        f"**Embedding:** BGE-M3 (1024-dim, 8K context)\n"
        f"**Pipeline:** Retrieve({TOP_K_RETRIEVAL}) → Rerank({TOP_K_RERANK}) → Reason → Respond in Vietnamese"
    )
    
    with gr.Accordion("⚠️ ĐỌC KỸ: CẢNH BÁO Y TẾ", open=False):
        gr.Markdown(MEDICAL_DISCLAIMER)
    
    gr.ChatInterface(
        fn=chat,
        description="Hệ thống tra cứu y khoa với khả năng suy luận và trả lời HOÀN TOÀN bằng tiếng Việt.",
        examples=[
            "Triệu chứng của bệnh tiểu đường type 2 là gì?",
            "So sánh metformin và insulin cho điều trị tiểu đường?",
            "Tác dụng phụ của aspirin là gì?",
            "Biến chứng của phẫu thuật thay khớp háng?",
            "Cách phòng ngừa bệnh tim mạch ở người trên 50 tuổi?"
        ],
        fill_height=True,
    )
    
    with gr.Accordion("💡 Tips sử dụng", open=False):
        gr.Markdown("""
### Cách hỏi hiệu quả:
- **Tốt:** "Triệu chứng của bệnh tiểu đường type 2 là gì? Giải thích nguyên nhân."
- **Kém:** "tiểu đường" (quá ngắn, không rõ ràng)

### Hệ thống này:
- Trả lời hoàn toàn bằng tiếng Việt (đã fix lỗi lẫn lộn ngôn ngữ)
- Cung cấp trích dẫn rõ ràng từ tài liệu
- Có khả năng suy luận logic cho câu hỏi phức tạp
- KHÔNG thay thế bác sĩ - chỉ để tham khảo thông tin

### Thời gian xử lý:
- Câu hỏi đơn giản: ~5-8 giây
- Câu hỏi phức tạp: ~10-15 giây (model đang "suy nghĩ")
        """)

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        auth=DEFAULT_AUTH,
        debug=True,
        show_error=True
    )
