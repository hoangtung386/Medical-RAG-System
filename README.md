# Hệ Thống RAG Y Tế (Medical RAG System) - Phiên Bản Nâng Cấp

Dự án này là một ứng dụng **Retrieval Augmented Generation (RAG)** chuyên sâu cho lĩnh vực y tế, đã được **nâng cấp toàn diện** để sử dụng các công nghệ tiên tiến nhất hiện nay. Hệ thống tra cứu tài liệu y khoa (PDF) và trả lời câu hỏi chuyên sâu bằng tiếng Việt với khả năng suy luận logic (Reasoning).

## 🚀 Công Nghệ Cốt Lõi (Mới)

Dự án hiện tại sử dụng bộ đôi mô hình mạnh mẽ nhất trong phân khúc Open Source:

1.  **Reasoning Model (Tư Duy):** [**DeepSeek-R1-Distill-Qwen-7B**](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
    *   Khả năng "tư duy" (Chain-of-Thought) trước khi trả lời.
    *   Phân tích vấn đề y khoa theo từng bước logic, chéo kiểm thông tin và đưa ra kết luận thận trọng.
    *   Hiệu nâng vượt trội so với các mô hình 7B/8B thông thường (Llama 3.1, v.v.).

2.  **Embedding Model (Vector hóa):** [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
    *   Mô hình embedding đa ngôn ngữ (Multilingual) tốt nhất hiện nay.
    *   Hỗ trợ ngữ nghĩa tiếng Việt và tiếng Anh cực tốt.
    *   Tối ưu hóa cho việc tìm kiếm thông tin y tế dày đặc.

## ✨ Tính Năng Nổi Bật

-   **Deep Reasoning (Suy luận sâu)**: Hệ thống không chỉ trích xuất thông tin mà còn tổng hợp và phân tích logic để trả lời các câu hỏi phức tạp (Ví dụ: So sánh thuốc, cơ chế bệnh sinh).
-   **Tra cứu chính xác (High Precision)**:
    -   Sử dụng **Smart Chunking**: Cắt văn bản thông minh (1500 tokens) để giữ trọn vẹn ngữ cảnh y khoa.
    -   **Cross-Encoder Reranking**: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) lọc lại kết quả tìm kiếm để đảm bảo độ tin cậy cao nhất.
-   **Minh bạch & An toàn**:
    -   Trích dẫn nguồn gốc (Source Citations) rõ ràng cho từng ý.
    -   Cảnh báo y tế và từ chối đưa ra lời khuyên điều trị cụ thể.
-   **Giao diện hiện đại**: Gradio UI cải tiến với thanh tiến trình hiển thị các bước: *Tìm kiếm -> Rerank -> Suy luận*.
-   **Tối ưu phần cứng**: Chạy mượt mà trên GPU tầm trung (VRAM 12GB+) nhờ kỹ thuật Quantization 4-bit (bitsandbytes).

## 🛠 Yêu Cầu Hệ Thống

-   **Hệ điều hành**: Windows / Linux
-   **Python**: 3.10 trở lên
-   **GPU**: NVIDIA GPU vói VRAM tối thiểu **8GB** (Khuyến nghị 12GB+ để chạy tốt DeepSeek-R1 + BGE-M3).
-   **CUDA**: 12.1+

## 📦 Cài Đặt

1.  **Clone dự án và cài đặt thư viện**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chuẩn bị dữ liệu (Ingest)**:
    *   Bỏ các file PDF tài liệu y khoa vào thư mục `Medical_documents/`.
    *   Chạy script để tạo cơ sở dữ liệu vector (Lần đầu chạy sẽ tải model BGE-M3, mất vài phút):
    ```bash
    python ingest.py
    ```
    *Lưu ý: Nếu bạn có thêm tài liệu mới, hãy chạy lại lệnh này.*

3.  **Khởi chạy Dịch vụ**:
    ```bash
    python app.py
    ```
    *   Lần đầu chạy sẽ tải model DeepSeek-R1 (~5GB), vui lòng kiên nhẫn.
    *   Truy cập Web UI tại: `http://localhost:7860`

## 🔑 Đăng Nhập Mặc Định

Dự án tích hợp bảo mật cơ bản:
-   **Username**: `admin`
-   **Password**: `123456`
*(Bạn có thể thay đổi thông tin này trong file `app.py`)*

## 📂 Cấu Trúc File

-   `Medical_documents/`: Thư mục chứa tài liệu gốc.
-   `chroma_db/`: Database chứa vector (Đừng xóa thủ công trừ khi muốn reset).
-   `ingest.py`: Script xử lý dữ liệu (Sử dụng BGE-M3 + Smart Chunking).
-   `app.py`: Ứng dụng chính (Chứa logic Reasoning + Gradio Interface).

## ⚠️ Lưu Ý Quan Trọng

-   **Tốc độ**: Vì sử dụng Reasoning Model, chatbot có thể mất **5-15 giây** để "suy nghĩ" trước khi bắt đầu trả lời. Đây là tính năng, không phải lỗi.
-   **Y tế**: Hệ thống chỉ mang tính chất tham khảo học thuật. **TUYỆT ĐỐI KHÔNG** sử dụng thay thế bác sĩ trong các trường hợp cấp cứu hoặc chẩn đoán thực tế.
