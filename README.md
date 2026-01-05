# Hệ Thống RAG Y Tế (Medical RAG System)

Dự án này là một ứng dụng **Retrieval Augmented Generation (RAG)** chạy offline, giúp tra cứu thông tin từ các tài liệu y khoa (PDF) và trả lời câu hỏi bằng tiếng Việt sử dụng mô hình ngôn ngữ lớn **GPT-OSS-20B**.

## Tính Năng
- **Tra cứu thông minh**: Tìm kiếm thông tin liên quan từ kho dữ liệu PDF tiếng Anh.
- **Hỗ trợ Tiếng Việt**: Người dùng hỏi bằng tiếng Việt, hệ thống tìm kiếm trong tài liệu tiếng Anh và trả lời lại bằng tiếng Việt.
- **Trích Dẫn Chính Xác**: Hiển thị nguồn tài liệu cụ thể (Source ID, Tên file, Trang) cho mỗi thông tin được đưa ra.
- **An Toàn & Bảo Mật**:
    - Tích hợp đăng nhập (Authentication) để bảo vệ truy cập.
    - Kiểm soát đầu vào (Input Sanitization) để chống Spam/Attack.
    - Tích hợp các quy tắc an toàn y tế (Medical Safety Guidelines).
- **Reranking Tối Ưu**: Sử dụng Cross-Encoder để lọc kết quả, chỉ lấy thông tin có độ tin cậy >30% và xếp hạng lại Top 8.
- **Tối Ưu Hiệu Năng**: Hỗ trợ 4-bit Quantization giúp chạy mượt mà trên GPU tầm trung.
- **Offline**: Chạy hoàn toàn trên máy cá nhân, đảm bảo bảo mật dữ liệu.
- **Giao diện thân thiện**: Gradio Chat Interface với thanh tiến trình (Progress Bar), Streaming và hỗ trợ chia sẻ từ xa (Remote Share).

## Yêu Cầu Hệ Thống
- **Python 3.10+**
- **GPU**: Khuyên dùng NVIDIA GPU (VRAM > 12GB) để chạy mô hình 20B parameters mượt mà (sử dụng 4-bit quantization nếu cần). Nếu chạy CPU sẽ rất chậm.

## Cài Đặt

1. **Cài đặt thư viện**:
   Mở terminal tại thư mục dự án và chạy lệnh:
   ```bash
   pip install -r requirements.txt
   ```

2. **Nạp dữ liệu (Ingest)**:
   Bước này sẽ đọc các file PDF trong thư mục `Medical_documents`, tạo vector embeddings và lưu vào `chroma_db`.
   ```bash
   python ingest.py
   ```
   *Lưu ý: Cần chạy lại lệnh này mỗi khi bạn thêm/sửa tài liệu mới.*

3. **Chạy ứng dụng**:
   Khởi động chatbot:
   ```bash
   python app.py
   ```
   Sau khi model load xong, truy cập đường dẫn hiện ra (thường là `http://127.0.0.1:7860`).
   
   **Thông tin đăng nhập mặc định**:
   - Username: `admin`
   - Password: `123456`
   *(Bạn có thể đổi mật khẩu trong file `app.py`, dòng `DEFAULT_AUTH`)*

## Cấu Trúc Thư Mục
- `Medical_documents/`: Nơi chứa các file PDF tài liệu y khoa.
- `chroma_db/`: Cơ sở dữ liệu vector lưu trữ thông tin đã xử lý.
- `ingest.py`: Script xử lý và nạp dữ liệu.
- `app.py`: Ứng dụng chính (Gradio UI + RAG logic).
- `requirements.txt`: Danh sách các thư viện cần thiết.

## Lưu Ý
- Mô hình **GPT-OSS-20B** khá nặng. Lần đầu chạy sẽ mất thời gian tải model.
- Để tối ưu hóa tốc độ và bộ nhớ, dự án sử dụng thư viện `unsloth` và `bitsandbytes` (nếu có GPU).
