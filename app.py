import os
import gradio as gr
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
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

# 1. TRANSLATION MODELS (The Bridge)
VI2EN_MODEL_ID = "vinai/vinai-translate-vi2en"
EN2VI_MODEL_ID = "vinai/vinai-translate-en2vi"

# 2. MEDICAL REASONING MODEL (The Brain)
# Using MedGemma 4B or similar lightweight medical model compatible with P100
REASONING_MODEL_ID = "unsloth/medgemma-4b-it-bnb-4bit" # Optimized 4-bit version if available
# Fallback to standard if unsloth not found, but we will try to load optimized
# Note: For this implementation, we will assume standard loading with BNB if specific unsloth binary isn't present, 
# but pointing to the base google/medgemma-4b-it with BNB config is safer if unsloth path is uncertain.
# Let's stick to the reliable path:
REASONING_MODEL_ID = "google/medgemma-4b-it" 

# 3. RETRIEVAL MODELS
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL = "BAAI/bge-m3"

DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TWEAKABLE PARAMETERS
RELEVANCE_THRESHOLD = 0.3
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5 # Reduced slightly to save context window
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.2 # Lower for medical accuracy

# SECURITY
DEFAULT_AUTH = ("admin", "123456")

print(f"Device: {DEVICE}")

# --- INITIALIZATION ---

# 1. Load Translation Models (FP16 for speed/memory balance)
print("Loading Translation Bridges...")
try:
    # Vi -> En
    vi2en_tokenizer = AutoTokenizer.from_pretrained(VI2EN_MODEL_ID)
    vi2en_model = AutoModelForSeq2SeqLM.from_pretrained(
        VI2EN_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    
    # En -> Vi
    en2vi_tokenizer = AutoTokenizer.from_pretrained(EN2VI_MODEL_ID)
    en2vi_model = AutoModelForSeq2SeqLM.from_pretrained(
        EN2VI_MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    print("✅ Translation models loaded.")
except Exception as e:
    logger.error(f"Error loading translation models: {e}")
    raise e

# 2. Load Retriever & Reranker
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

# 3. Load Medical Reasoning Model (4-bit Quantized)
print(f"Loading Medical Brain ({REASONING_MODEL_ID})...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    reasoning_tokenizer = AutoTokenizer.from_pretrained(REASONING_MODEL_ID)
    reasoning_model = AutoModelForCausalLM.from_pretrained(
        REASONING_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Medical Reasoning Model loaded.")
except Exception as e:
    logger.error(f"Error loading Medical Model: {e}")
    # Fallback logic could go here, but for now we raise
    raise e

# --- HELPER FUNCTIONS ---

def translate(text, tokenizer, model, max_length=1024):
    """Generic translation function."""
    if not text or not text.strip():
        return ""
    
    try:
        input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids.to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=5,  # Increased beams for better quality
                early_stopping=True,
                repetition_penalty=1.2, # Stronger penalty for loops
                no_repeat_ngram_size=3, # Prevent phrase repetition
                length_penalty=1.0 
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text # Return original on failure

def format_system_prompt(context):
    """
    Creates a strict medical system prompt in ENGLISH for MedGemma.
    """
    return (
        "You are an expert medical AI assistant. Answer the user's question based strictly on the provided context.\n"
        "If the information is not in the context, say 'Insufficient information in the provided documents'.\n"
        "Do not hallucinate medical advice.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Reference citations using [Source X] format."
        "Answer concisely and professionally."
    )

def chat(message, history, progress=gr.Progress()):
    """
    5-Stage Pipeline:
    1. Vi -> En
    2. Retrieve
    3. Reason (En)
    4. En -> Vi
    5. Return
    """
    if not retriever:
        yield "Hệ thống chưa có dữ liệu. Vui lòng chạy ingest.py."
        return

    try:
        # STEP 1: Translate Input (Vi -> En)
        progress(0.1, desc="🔍 Bước 1: Dịch câu hỏi sang tiếng Anh...")
        en_query = translate(message, vi2en_tokenizer, vi2en_model)
        yield f"🔄 Đã dịch: {en_query}\n\n⏳ Đang tìm kiếm tài liệu..."
        
        # STEP 2: Retrieval
        progress(0.3, desc="📚 Bước 2: Tìm kiếm dữ liệu y khoa...")
        docs = retriever.invoke(en_query) # Retrieve using English query
        
        if not docs:
            yield "Không tìm thấy tài liệu phù hợp."
            return

        # Rerank
        doc_texts = [d.page_content for d in docs]
        scores = reranker.predict([[en_query, t] for t in doc_texts])
        top_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > RELEVANCE_THRESHOLD][:TOP_K_RERANK]
        top_docs = [docs[i] for i in top_indices]
        
        if not top_docs:
             yield "Không tìm thấy tài liệu đủ độ tin cậy."
             return

        # Build Context
        context_parts = []
        sources_list = []
        for i, doc in enumerate(top_docs):
            src = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', '?')
            context_parts.append(f"[Source {i+1}] {doc.page_content} (File: {src}, Page: {page})")
            sources_list.append(f"- [Source {i+1}] {src} (Trang {page})")
            
        context_str = "\n\n".join(context_parts)
        
        # STEP 3: Medical Reasoning (English)
        progress(0.5, desc="🧠 Bước 3: Phân tích y khoa (MedGemma)...")
        
        system_prompt = format_system_prompt(context_str)
        # Apply chat template
        messages = [
            {"role": "user", "content": system_prompt + f"\n\nQuestion: {en_query}"}
        ]
        
        inputs = reasoning_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(DEVICE)

        streamer = TextIteratorStreamer(reasoning_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
        )
        
        t = Thread(target=reasoning_model.generate, kwargs=generate_kwargs)
        t.start()
        
        en_response = ""
        for token in streamer:
            en_response += token
            # Optional: Show thinking process if desired, but might be messy
        
        # STEP 4: Translate Output (En -> Vi)
        progress(0.8, desc="🇻🇳 Bước 4: Dịch câu trả lời sang tiếng Việt...")
        vi_response = translate(en_response, en2vi_tokenizer, en2vi_model)
        
        # STEP 5: Final Output
        final_output = (
            f"{vi_response}\n\n"
            f"---\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources_list) + "\n\n"
            f"*(Original Reasoning: {en_response[:100]}...)*" # Optional debug info
        )
        
        yield final_output

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        yield f"❌ Lỗi hệ thống: {str(e)}"

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft(), title="Medical RAG System (MedGemma)", fill_height=True) as demo:
    gr.Markdown(
        "# 🏥 Hệ thống Trợ lý Y khoa Chuyên sâu\n"
        "**Kiến trúc:** 5-Stage Pipeline (Vi-En Bridge + MedGemma Reasoning)\n"
        "**Mô hình:** VinAI-Translate & MedGemma-4B-IT"
    )
    
    with gr.Accordion("ℹ️ Lưu ý quan trọng", open=False):
        gr.Markdown(
            "- Hệ thống sử dụng mô hình dịch thuật để tận dụng kiến thức y khoa tiếng Anh.\n"
            "- Thời gian phản hồi có thể lâu hơn (10-15s) do quy trình xử lý đa bước.\n"
            "- Luôn kiểm tra lại với bác sĩ chuyên khoa."
        )

    gr.ChatInterface(
        fn=chat,
        description="Hỏi đáp y khoa với quy trình suy luận chuyên sâu.",
        examples=[
            "Triệu chứng của bệnh tiểu đường type 2 là gì?",
            "Tác dụng phụ của thuốc aspirin?",
            "Làm sao để phòng ngừa bệnh tim mạch?"
        ]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0", auth=DEFAULT_AUTH)
