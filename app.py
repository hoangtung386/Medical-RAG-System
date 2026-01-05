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

# --- CONFIGURATION (CONSTANTS) ---
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tweakable Parameters
RELEVANCE_THRESHOLD = 0.3
MAX_HISTORY_LEN = 10  # Limit chat history
MAX_INPUT_LEN = 2000
MIN_INPUT_LEN = 5
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 8
TEMPERATURE = 0.6 # Slightly lower for Llama 3 to reduce hallucinations
MAX_NEW_TOKENS = 512

# Security
DEFAULT_AUTH = ("admin", "123456") # Change this!

MEDICAL_DISCLAIMER = """
### ⚠️ CẢNH BÁO Y TẾ QUAN TRỌNG / IMPORTANT MEDICAL DISCLAIMER
1. **Mục đích tham khảo**: Công cụ này chỉ cung cấp thông tin y tế tổng quát để tham khảo, được trích xuất từ tài liệu có sẵn.
2. **Không thay thế bác sĩ**: Thông tin **KHÔNG** có giá trị chẩn đoán, điều trị hay tư vấn y khoa chính thức.
3. **Miễn trừ trách nhiệm**: Người dùng tự chịu trách nhiệm khi sử dụng thông tin. Luôn tham khảo ý kiến bác sĩ hoặc chuyên gia y tế cho các vấn đề sức khỏe cụ thể.
"""

print(f"Device: {DEVICE}")

# --- INITIALIZATION ---

# Load Retriever
print("Loading Vector Database...")
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
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
# Use GPU for Reranker if available for speed
reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE) 

# Load Model
print(f"Loading Model {MODEL_ID}...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config if DEVICE == "cuda" else None,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    if DEVICE == "cpu":
        logger.warning("Running on CPU! This will be extremely slow for a 20B model.")
        model.to("cpu")
except Exception as e:
    logger.error(f"Error loading {MODEL_ID}: {e}")
    print("Please ensure you have the model access and hardware requirements.")
    raise e

# --- HELPER FUNCTIONS ---

def validate_input(message):
    """Checks input length and validity."""
    if not message or len(message.strip()) < MIN_INPUT_LEN:
        return "Câu hỏi quá ngắn. Vui lòng nhập chi tiết hơn."
    if len(message) > MAX_INPUT_LEN:
        return f"Câu hỏi quá dài (>{MAX_INPUT_LEN} ký tự). Vui lòng rút gọn."
    return None

def format_prompt(message, history, context):
    """
    Constructs the prompt for the LLM properly formatting history and context.
    """
    system_prompt = (
        "You are a medical information assistant. Follow these rules strictly:\n"
        "1. **Language**: Always respond in Vietnamese, even if context is in English.\n"
        "2. **Citations**: MUST cite specific sources [Source X] for every claim.\n"
        "3. **Accuracy**: If sources conflict, mention all viewpoints.\n"
        "4. **Limitations**: If information is insufficient, clearly state 'Thông tin chưa đầy đủ để trả lời chính xác'.\n"
        "5. **Scope**: Only answer medical/health questions based on provided context.\n"
        "6. **Safety**: Never provide specific diagnoses or treatment recommendations - always defer to healthcare professionals.\n\n"
        "Context provided includes strictly numbered sources (e.g., [Source 1], [Source 2])."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history (Limited)
    for human, ai in history[-MAX_HISTORY_LEN:]:
        messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
    
    # Add current message with context
    content_with_context = f"Context:\n{context}\n\nQuestion: {message}"
    messages.append({"role": "user", "content": content_with_context})
    
    return messages

def chat(message, history, progress=gr.Progress()):
    """
    Main chat logic: Validation -> Retrieval -> Reranking -> Generation -> Streaming.
    """
    error_msg = validate_input(message)
    if error_msg:
        yield error_msg
        return

    if not retriever:
        yield "Lỗi: Cơ sở dữ liệu chưa sẵn sàng. Vui lòng chạy ingest.py."
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
            
            # Sort & Filter
            sorted_indices = np.argsort(scores)[::-1]
            
            top_k_indices = []
            for i in sorted_indices:
                if scores[i] > RELEVANCE_THRESHOLD:
                    top_k_indices.append(i)
                if len(top_k_indices) >= TOP_K_RERANK:
                    break
            
            top_docs = [docs[i] for i in top_k_indices]
        
        if not top_docs:
            yield "Xin lỗi, tôi không tìm thấy thông tin đủ độ tin cậy trong tài liệu để trả lời câu hỏi của bạn."
            return

        progress(0.6, desc="Đang tổng hợp câu trả lời...")

        # 3. Context Construction with Metadata Fix
        context_pieces = []
        sources_list = []
        
        for i, doc in enumerate(top_docs):
            # Extract metadata safely (Audit Fix #1)
            source_path = doc.metadata.get('source', 'Unknown File')
            filename = os.path.basename(source_path)
            
            # Safe page number logic
            raw_page = doc.metadata.get('page', -1)
            if isinstance(raw_page, int) and raw_page >= 0:
                page_display = raw_page + 1
            else:
                page_display = "Unknown"
            
            # Create context string
            context_pieces.append(f"[Source {i+1}]: {doc.page_content}\n(Reference: {filename}, Page {page_display})")
            sources_list.append(f"- [Source {i+1}]: {filename} (Page {page_display})")
            
        context = "\n\n".join(context_pieces) if context_pieces else ""
        
        # 4. Generate
        messages = format_prompt(message, history, context)
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        partial_response = ""
        for new_token in streamer:
            partial_response += new_token
            yield partial_response

        # 5. Append Sources
        if sources_list:
            yield partial_response + "\n\n**Tài liệu tham khảo:**\n" + "\n".join(sources_list)

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        yield f"Đã xảy ra lỗi hệ thống: {str(e)}"

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft(), title="Medical RAG Assistant") as demo:
    gr.Markdown(MEDICAL_DISCLAIMER)
    gr.Markdown(f"# Medical RAG Assistant\nModel: {MODEL_ID} | Docs: {TOP_K_RETRIEVAL}->{TOP_K_RERANK}")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, show_label=False)
            msg = gr.Textbox(label="Nhập câu hỏi y tế của bạn...", placeholder="Ví dụ: Triệu chứng của bệnh tiểu đường là gì?")
            with gr.Row():
                submit_btn = gr.Button("Gửi câu hỏi", variant="primary")
                clear_btn = gr.Button("Xóa")
        with gr.Column(scale=1):
            gr.Markdown("### Gợi ý câu hỏi")
            gr.Examples(
                examples=[
                    "Triệu chứng của bệnh tiểu đường type 2 là gì?",
                    "Cách phòng ngừa bệnh tim mạch?",
                    "Tác dụng phụ của aspirin?",
                    "Biến chứng của phẫu thuật thay khớp háng?"
                ],
                inputs=msg
            )

    # Event Handlers
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = chat(user_message, history[:-1])
        history[-1][1] = ""
        for chunk in bot_message:
            history[-1][1] = chunk
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Audit Security Fix: Added Auth and server_name default
    demo.queue().launch(
        share=True, 
        server_name="0.0.0.0", 
        auth=DEFAULT_AUTH,
        root_path=None 
    )
