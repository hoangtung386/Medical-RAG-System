# Hệ Thống RAG Y Tế (Medical RAG System) - Single-Model Architecture

Dự án này là một ứng dụng **Retrieval Augmented Generation (RAG)** chuyên sâu cho lĩnh vực y tế, sử dụng kiến trúc **Single-Model** tối giản nhưng mạnh mẽ, loại bỏ hoàn toàn module dịch thuật trung gian để tăng độ chính xác và tốc độ phản hồi.

## 🚀 Kiến Trúc Mới: "Direct Vietnamese Processing"

Thay vì phải dịch qua lại (Vi-En-Vi), hệ thống sử dụng các mô hình ngôn ngữ lớn (LLM) thế hệ mới có khả năng hiểu và trả lời tiếng Việt tự nhiên cực tốt.

**Quy trình xử lý đơn giản hóa (3 Bước):**

1.  **Retrieval**: Tìm kiếm tài liệu y khoa liên quan từ cơ sở dữ liệu bằng **BGE-M3**.
2.  **Reasoning**: Mô hình AI (Gemma 3 27B / Qwen 2.5 32B) phân tích tài liệu và suy luận trực tiếp bằng tiếng Việt.
3.  **Response**: Trả về câu trả lời chuyên sâu kèm trích dẫn nguồn.

## 🧠 Các Mô Hình Cốt Lõi

1.  **Medical Logic & Reasoning:**
    *   [**unsloth/gemma-2-9b-it-bnb-4bit**](https://huggingface.co/unsloth/gemma-2-9b-it-bnb-4bit) (Hiện tại): Model cân bằng hoàn hảo giữa tốc độ và độ chính xác suy luận y khoa. Chạy mượt mà trên P100.
    *   *Tất cả đều được tối ưu hóa (4-bit Quantization) để chạy trên GPU 16GB.*

2.  **Embedding:** [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
    *   Giữ nguyên do hiệu năng vượt trội trong tìm kiếm đa ngôn ngữ.

## 🖥️ Giao Diện Hệ Thống

Dưới đây là hình ảnh thực tế của hệ thống:

**1. Màn hình Đăng nhập (Login)**
Bảo mật cơ bản với tài khoản `admin` / `123456`.
![Giao diện đăng nhập](/Images/Login_interface.png)

**2. Giao diện Làm việc (Chat Interface)**
Nơi bác sĩ đặt câu hỏi và nhận câu trả lời từ AI.
![Giao diện làm việc](/Images/Working_interface.png)

## 📦 Cài Đặt & Sử Dụng

### 1. Yêu Cầu
*   Python 3.10+
*   NVIDIA GPU (CUDA) - VRAM **16GB** (Tesla P100/T4)

### ⚠️ Quan Trọng: Cấp Quyền Model
Mô hình **Gemma 2** yêu cầu xin quyền truy cập. 
1. Truy cập [Hugging Face Gemma 2](https://huggingface.co/google/gemma-2-9b-it).
2. Nhấn "Request Access".
3. Đăng nhập terminal: `huggingface-cli login`

### 2. Cài Đặt
```bash
pip install -r requirements.txt
```

### 3. Nạp Dữ Liệu (Ingest)
Copy file PDF tài liệu y khoa vào thư mục `Medical_documents/` và chạy:
```bash
python ingest.py
```

### 4. Khởi Chạy
```bash
python app.py
```
*   Truy cập Web UI tại: `http://localhost:7860`

## 📂 Cấu Trúc Dự Án
*   `app.py`: Logic chính (Single-Model Pipeline).
*   `ingest.py`: Xử lý và vector hóa tài liệu.
*   `Medical_documents/`: Thư mục chứa PDF.
*   `chroma_db/`: Cơ sở dữ liệu Vector.
*   `Images/`: Thư mục chứa ảnh giao diện.

---
**Cảnh báo y tế**: Hệ thống chỉ mang tính chất tham khảo thông tin, không thay thế chẩn đoán của bác sĩ.
