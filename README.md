# Hệ Thống RAG Y Tế (Medical RAG System) - Phiên Bản Ministral Reasoning

Dự án này là một ứng dụng **Retrieval Augmented Generation (RAG)** chuyên sâu cho lĩnh vực y tế, được tối ưu hóa đặc biệt cho **tiếng Việt** và khả năng **suy luận logic (Reasoning)**. Hệ thống tra cứu tài liệu y khoa (PDF) và trả lời câu hỏi chuyên sâu, chính xác.

## 🚀 Công Nghệ Cốt Lõi

Hệ thống sử dụng các mô hình tiên tiến nhất (SOTA) trong phân khúc Open Source:

1.  **Reasoning Model (Tư Duy):** [**mistralai/Ministral-3-8B-Reasoning-2512**](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512)
    *   Mô hình ngôn ngữ thế hệ mới với khả năng suy luận mạnh mẽ.
    *   **Tối ưu hóa đa ngôn ngữ**, đặc biệt là khả năng xử lý và trả lời tiếng Việt tự nhiên, chính xác hơn nhiều so với các phiên bản trước.
    *   Tuân thủ nghiêm ngặt các hướng dẫn an toàn và cấu trúc trả lời.

2.  **Embedding Model (Vector hóa):** [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
    *   Mô hình embedding đa ngôn ngữ tốt nhất hiện nay.
    *   Hỗ trợ vector mật độ cao (Dense Retrieval) và thưa (Sparse Retrieval), tối ưu cho tìm kiếm y khoa.

## ✨ Tính Năng Nổi Bật

-   **Vietnamese First:** Hệ thống được tinh chỉnh để **luôn trả lời bằng tiếng Việt**, loại bỏ hiện tượng pha trộn ngôn ngữ (Anh/Việt) thường gặp.
-   **Deep Reasoning:** Không chỉ tìm kiếm, mô hình còn phân tích, tổng hợp và suy luận từ nhiều nguồn thông tin để trả lời các câu hỏi phức tạp (Ví dụ: So sánh thuốc, phác đồ điều trị).
-   **Độ Chính Xác Cao**:
    -   Quy trình 3 bước: **Tìm kiếm (Retrieve) -> Xếp hạng lại (Rerank) -> Suy luận (Reason)**.
    -   Sử dụng Cross-Encoder để lọc bỏ thông tin nhiễu.
-   **Minh Bạch Nguồn Tin**: Mọi thông tin đưa ra đều đi kèm trích dẫn cụ thể `[Source X]` (Tên file, Số trang).
-   **Giao diện Thông Minh**: Gradio UI hiển thị trạng thái xử lý chi tiết và các mẹo đặt câu hỏi hiệu quả.

## 🛠 Yêu Cầu Hệ Thống

-   **OS**: Windows / Linux
-   **Python**: 3.10+
-   **GPU**: NVIDIA GPU (Khuyến nghị **VRAM 12GB+** để chạy mượt mà Ministral-3-8B ở chế độ 4-bit + BGE-M3).
-   **RAM**: 16GB+

## 📦 Cài Đặt & Sử Dụng

1.  **Cài đặt thư viện**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Chuẩn bị dữ liệu (Ingest)**:
    *   Copy file PDF tài liệu y khoa vào thư mục `Medical_documents/`.
    *   Chạy lệnh nạp dữ liệu (tạo vector DB):
    ```bash
    python ingest.py
    ```
    *(Chạy lại lệnh này mỗi khi có tài liệu mới)*

3.  **Khởi chạy Chatbot**:
    ```bash
    python app.py
    ```
    *   Lần đầu chạy sẽ tải model (~5-6GB).
    *   Truy cập Web UI tại: `http://localhost:7860`

## 🔑 Tài Khoản Truy Cập

Hệ thống có bảo mật đăng nhập cơ bản:
-   **Username**: `admin`
-   **Password**: `123456`
*(Thông tin này có thể đổi trong file `app.py`)*

## 📂 Cấu Trúc Dự Án

-   `Medical_documents/`: Thư mục chứa tài liệu PDF đầu vào.
-   `chroma_db/`: Cơ sở dữ liệu vector (ChromaDB).
-   `ingest.py`: Script xử lý tài liệu (Sử dụng BGE-M3 + Smart Chunking 1500 tokens).
-   `app.py`: Ứng dụng chính (Gradio UI + Ministral Reasoning Logic).

## ⚠️ Lưu Ý

-   **Thời gian phản hồi**: Với các câu hỏi phức tạp, mô hình cần **10-15 giây** để "suy nghĩ" và tổng hợp thông tin.
-   **Cảnh báo y tế**: Hệ thống là công cụ hỗ trợ tra cứu tham khảo. **KHÔNG** sử dụng thay thế bác sĩ trong chẩn đoán và điều trị thực tế.
