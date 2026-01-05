import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from threading import Thread
import numpy as np

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

# Load Model
print(f"Loading Model {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    if DEVICE == "cpu":
        model.to("cpu")
except Exception as e:
    print(f"Error loading {MODEL_ID}: {e}")
    print("Please ensure you have the model access and hardware requirements.")
    raise e

def format_prompt(message, history, context):
    system_prompt = (
        "You are a helpful medical assistant. The user will ask questions in Vietnamese. "
        "The context provided includes strictly numbered sources (e.g., [Source 1], [Source 2]). "
        "You must answer strictly in Vietnamese. "
        "CRITICAL: When answering, you MUST cite the specific source ID that supports your statement (e.g., 'Theo nguồn [1]...'). "
        "If you don't know the answer based on the context, say you don't know in Vietnamese."
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
        
        # Take top 8 after reranking
        top_k_indices = sorted_indices[:8]
        top_docs = [docs[i] for i in top_k_indices]
    
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

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Medical RAG Assistant (GPT-OSS-20B)")
    
    chatbot = gr.Chatbot(height=600)
    msg = gr.Textbox(label="Ask a question about the medical documents")
    clear = gr.Button("Clear")

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
