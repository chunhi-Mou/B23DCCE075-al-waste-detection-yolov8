# Demo: Phân loại rác thải tái chế (YOLOv8n · Active Learning · PTIT 2026)

## Giới thiệu

Ứng dụng web Gradio minh hoạ mô hình YOLOv8n phát hiện và phân loại rác thải theo
vật liệu (5 lớp: bìa carton, giấy, thuỷ tinh, kim loại, nhựa). Có 3 tab: **Phân tích**
(tải ảnh hoặc chụp webcam để nhận diện và xem mẹo tái chế), **Kết quả huấn luyện**
(biểu đồ so sánh 4 chiến lược Active Learning đọc từ `results/`), và **Dev / So sánh**
(chọn model bất kỳ, chỉnh ngưỡng, xem chi tiết phát hiện).

## Cài đặt và chạy

1. Cài các gói đã pin trong `requirements.txt` của repo (`ultralytics`, `torch`,
   `gradio`, `opencv-python-headless`, `numpy`, `matplotlib`):

   ```bash
   pip install -r requirements.txt
   ```

2. Chạy ứng dụng từ thư mục gốc của repo (nơi chứa `results/`):

   ```bash
   python -m demo.app                # local + link chia sẻ tạm thời
   python -m demo.app --no-share     # chỉ chạy local
   ```

3. Mở trình duyệt tại `http://127.0.0.1:7860`. Thêm `--host`, `--port`,
   `--no-browser` để tuỳ chỉnh.

Để dùng `results/` ở vị trí khác, đặt biến môi trường `DEMO_RESULTS_DIR` trỏ tới nó.
