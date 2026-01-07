# Hệ Thống RAG Y Tế (Medical RAG System) - Phiên Bản Translation Bridge

Dự án này là một ứng dụng **Retrieval Augmented Generation (RAG)** chuyên sâu cho lĩnh vực y tế, sử dụng kiến trúc **Pipeline 5 Tầng** độc đáo để kết hợp khả năng suy luận y khoa chuẩn xác của mô hình quốc tế với trải nghiệm tiếng Việt mượt mà.

## 🚀 Kiến Trúc "Translation Bridge"

Để tối ưu hóa độ chính xác y khoa trên phần cứng giới hạn (**Tesla P100 16GB**), hệ thống sử dụng quy trình xử lý 5 bước:

1.  **Input**: Câu hỏi tiếng Việt.
2.  **Bridge 1 (Vi → En)**: Dịch câu hỏi sang tiếng Anh chuyên ngành y bằng **VinAI-Translate**.
3.  **Retrieval**: Tìm kiếm tài liệu y khoa tiếng Anh (độ chính xác cao hơn tiếng Việt) bằng **BGE-M3**.
4.  **Reasoning**: Suy luận và trả lời bằng **MedGemma-4B** (Mô hình chuyên y tế của Google).
5.  **Bridge 2 (En → Vi)**: Dịch câu trả lời về tiếng Việt bằng **VinAI-Translate**.

## 🧠 Các Mô Hình Cốt Lõi

1.  **Medical Reasoning:** [**google/medgemma-4b-it**](https://huggingface.co/google/medgemma-4b-it)
    *   Tối ưu hóa (Quantization 4-bit) để chạy mượt trên GPU 16GB.
    *   Được huấn luyện chuyên sâu trên dữ liệu y khoa (Medical Papers, Guidelines).

2.  **Translation Bridge:** [**vinai/vinai-translate**](https://huggingface.co/vinai/vinai-translate-vi2en)
    *   Mô hình dịch máy tốt nhất cho cặp câu Việt-Anh hiện nay.
    *   Hiểu rõ thuật ngữ y khoa Việt Nam.

3.  **Embedding:** [**BAAI/bge-m3**](https://huggingface.co/BAAI/bge-m3)
    *   Giữ nguyên từ phiên bản trước do hiệu năng vượt trội.

## ✨ Điểm Mạnh & Lưu Ý

### ✅ Điểm Mạnh
*   **Độ Chính Xác Y Khoa**: Sử dụng nguồn tri thức y học chuẩn tiếng Anh và mô hình MedGemma chuyên dụng.
*   **Tiếng Việt Tự Nhiên**: Không bị "lơ lớ" nhờ module dịch thuật chuyên biệt của VinAI.
*   **Minh Bạch**: Trích dẫn nguồn tài liệu `[Source X]` rõ ràng.

### ⚠️ Lưu Ý Quan Trọng
*   **Độ Trễ (Latency)**: Do phải qua 2 bước dịch thuật và 1 bước suy luận, thời gian phản hồi sẽ khoảng **10-15 giây/câu**.
*   **Cấu Hình**: Yêu cầu GPU tối thiểu **12GB VRAM** (Khuyến nghị 16GB P100/T4).

## 📦 Cài Đặt & Sử Dụng

### 1. Yêu Cầu
*   Python 3.10+
*   NVIDIA GPU (CUDA)

### 2. Cài Đặt
```bash
pip install -r requirements.txt
```
*Lưu ý: Cần cài đặt `sentencepiece` và `sacremoses` (đã có trong requirements.txt).*

### 3. Nạp Dữ Liệu (Ingest)
Copy file PDF tài liệu y khoa vào thư mục `Medical_documents/` và chạy:
```bash
python ingest.py
```

### 4. Khởi Chạy
```bash
python app.py
```
*   Lần đầu chạy sẽ tải khoảng **8-10GB** models.
*   Truy cập Web UI tại: `http://localhost:7860`

## 📂 Cấu Trúc Dự Án
*   `app.py`: Pipeline 5 bước (Translation -> Retrieval -> Reasoning).
*   `ingest.py`: Xử lý và vector hóa tài liệu.
*   `Medical_documents/`: Thư mục chứa PDF.
*   `chroma_db/`: Cơ sở dữ liệu Vector.

---
**Cảnh báo y tế**: Hệ thống chỉ mang tính chất tham khảo thông tin, không thay thế chẩn đoán của bác sĩ.
