import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from threading import Thread
import numpy as np
import logging
from transformers import BitsAndBytesConfig

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MEDICAL_DISCLAIMER = """
### ⚠️ CẢNH BÁO Y TẾ QUAN TRỌNG / IMPORTANT MEDICAL DISCLAIMER
1. **Mục đích tham khảo**: Công cụ này chỉ cung cấp thông tin y tế tổng quát để tham khảo, được trích xuất từ tài liệu có sẵn.
2. **Không thay thế bác sĩ**: Thông tin **KHÔNG** có giá trị chẩn đoán, điều trị hay tư vấn y khoa chính thức.
3. **Miễn trừ trách nhiệm**: Người dùng tự chịu trách nhiệm khi sử dụng thông tin. Luôn tham khảo ý kiến bác sĩ hoặc chuyên gia y tế cho các vấn đề sức khỏe cụ thể.
"""
RELEVANCE_THRESHOLD = 0.3

# Configuration
MODEL_ID = "unsloth/gpt-oss-20b"
DB_PATH = os.path.join(os.getcwd(), "chroma_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# Load Retriever
print("Loading Vector Database...")
# Must match the model used in ingest.py
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'})
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
# Retrieve more candidates first, then rerank
retriever = db.as_retriever(search_kwargs={"k": 10})

# Load Reranker
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLM-v2-L12-H384-v1"
print(f"Loading Reranker {RERANKER_MODEL}...")
reranker = CrossEncoder(RERANKER_MODEL, device="cpu") # Reranker is usually small enough for CPU or put on GPU if available

# Load Model with 4-bit Quantization
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
        model.to("cpu")
except Exception as e:
    logger.error(f"Error loading {MODEL_ID}: {e}")
    print("Please ensure you have the model access and hardware requirements.")
    raise e

def format_prompt(message, history, context):
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
    
    # Construct conversation history
    # Note: GPT-OSS-20B might expect a specific chat template.
    # Using a generic flexible format or the tokenizer's chat template if available.
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add history
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    
    # Add current message with context
    content_with_context = f"Context:\n{context}\n\nQuestion: {message}"
    messages.append({"role": "user", "content": content_with_context})
    
    return messages

def chat(message, history):
    try:
        # Retrieve context (fetch top 10)
        docs = retriever.invoke(message)
        
        # Reranking Logic
        doc_texts = [doc.page_content for doc in docs]
        top_docs = []
        
        if doc_texts:
            pairs = [[message, doc_text] for doc_text in doc_texts]
            scores = reranker.predict(pairs)
            
            # Sort by score descending
            sorted_indices = np.argsort(scores)[::-1]
            
            # Filter by Threshold and Take top 8
            top_k_indices = []
            for i in sorted_indices:
                if scores[i] > RELEVANCE_THRESHOLD:
                    top_k_indices.append(i)
                if len(top_k_indices) >= 8:
                    break
            
            top_docs = [docs[i] for i in top_k_indices]
        
        if not top_docs:
            yield "Xin lỗi, tôi không tìm thấy thông tin đủ độ tin cậy trong tài liệu để trả lời câu hỏi của bạn."
            return

            # Format context with metadata
        context_pieces = []
        sources_list = []
        
        for i, doc in enumerate(top_docs):
            # Extract metadata
            source_path = doc.metadata.get('source', 'Unknown File')
            filename = os.path.basename(source_path)
            page = doc.metadata.get('page', 'Unknown Page') + 1 # Page is usually 0-indexed
            
            # Create context string
            context_pieces.append(f"[Source {i+1}]: {doc.page_content}\n(Reference: {filename}, Page {page})")
            sources_list.append(f"- [Source {i+1}]: {filename} (Page {page})")
            
        context = "\n\n".join(context_pieces) if context_pieces else ""
        
        # Format input
        messages = format_prompt(message, history, context)
        
        # Apply template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        partial_response = ""
        for new_token in streamer:
            partial_response += new_token
            yield partial_response

        # Append sources at the end
        if sources_list:
            yield partial_response + "\n\n**Tài liệu tham khảo:**\n" + "\n".join(sources_list)

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        yield "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng kiểm tra log để biết thêm chi tiết."

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(MEDICAL_DISCLAIMER)
    gr.Markdown("# Medical RAG Assistant (GPT-OSS-20B)")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Ask a question about the medical documents")
            clear = gr.Button("Clear")
        with gr.Column(scale=1):
            gr.Markdown("### Ví dụ câu hỏi:")
            gr.Examples(
                examples=[
                    "Triệu chứng của bệnh tiểu đường type 2 là gì?",
                    "Cách phòng ngừa bệnh tim mạch?",
                    "Tác dụng phụ của aspirin?"
                ],
                inputs=msg
            )

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
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0")
