# Active Learning cho phát hiện rác thải sinh hoạt với YOLOv8

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![YOLOv8n](https://img.shields.io/badge/YOLOv8n-Ultralytics-00BFA5)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316?logo=gradio&logoColor=white)
![PTIT](https://img.shields.io/badge/PTIT-2026-C2185B)

## Mục lục

- [Giới thiệu dự án](#giới-thiệu-dự-án)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
  - [Môi trường](#môi-trường)
  - [Dữ liệu](#dữ-liệu)
  - [Các lệnh chính](#các-lệnh-chính)
  - [Demo](#demo)

## Giới thiệu dự án

Dự án so sánh bốn chiến lược Active Learning trên cùng một mô hình YOLOv8n để phát hiện rác thải sinh hoạt theo vật liệu, trong điều kiện ngân sách gán nhãn bị giới hạn. Câu hỏi trọng tâm là một chiến lược chọn mẫu thông minh có giúp giảm lượng nhãn cần gán so với chọn ngẫu nhiên hay không.

Tập dữ liệu gồm 12.621 ảnh, năm lớp vật liệu tái chế, tách 80/10/10 với hạt giống 13. Tập kiểm thử được giữ cố định, không tham gia vòng lặp huấn luyện. Hai đặc tính chi phối kết quả là mất cân bằng lớp khoảng 16:1 và 93,3% ảnh huấn luyện chỉ chứa một vật thể.

| Chiến lược | Ý tưởng | Nguồn |
|---|---|---|
| ![S0](https://img.shields.io/badge/S0-Random-9E9E9E) | Chọn ngẫu nhiên phân phối đều, làm đối chứng | Settles, 2009 |
| ![S1](https://img.shields.io/badge/S1-Uncertainty-1E88E5) | Bình phương biên khoảng cách hai lớp xác suất cao nhất | Brust, VISAPP 2019, arXiv:1809.09875 |
| ![S2](https://img.shields.io/badge/S2-CoreSet-43A047) | Vector toàn cục mỗi ảnh, k-Center-Greedy phủ đặc trưng | Sener & Savarese, ICLR 2018, arXiv:1708.00489 |
| ![S3](https://img.shields.io/badge/S3-PPAL-8E24AA) | Đường ống hai giai đoạn bất định kết hợp đa dạng | Yang, CVPR 2024, arXiv:2211.11612 |

Bốn chiến lược dùng chung tập khởi tạo theo từng hạt giống và cùng cấu hình huấn luyện, nên khác biệt đo được chỉ đến từ cách chọn mẫu. Quy trình chạy ba lượt (hạt giống 13, 42, 1337), khởi tạo 5% kho dữ liệu, mỗi vòng thêm 2,5% đến trần 20%, mỗi vòng 30 epoch không dừng sớm, ảnh 416, batch 64.

Vòng lặp Active Learning:

<img width="540" height="300" alt="diagram_al_loop" src="https://github.com/user-attachments/assets/054cf025-d377-454c-a745-0ce3ac95d652" />


Mô hình huấn luyện trên 100% tập dữ liệu đạt mAP@50 bằng 0,9383, làm mốc giới hạn trên. Trên miền dữ liệu đơn vật thể này, không chiến lược chủ động nào vượt chọn ngẫu nhiên ở mức có ý nghĩa thống kê: chỉ cặp Uncertainty với CoreSet đạt p-value 0,043, mọi so sánh với Random đều cho p-value lớn hơn 0,49. Đường cong học tập của bốn chiến lược bám sát và đan xen nhau.

Đường cong mAP50 theo ngân sách:

<img width="420" height="300" alt="curve_test_mAP50" src="https://github.com/user-attachments/assets/814d0867-c7d8-4caa-9017-ea746f99f6b7" />

Kết quả sự khác biệt không đáng kể giữa các AL này khoanh vùng được loại dữ liệu mà Active Learning chưa bù lại chi phí cài đặt, nhất quán với Gashi 2024 (arXiv:2403.14800). Đóng góp thực tiễn là bản chuyển đổi PPAL từ RetinaNet sang YOLOv8, kèm một ứng dụng web demo Gradio nhận diện và thống kê vật liệu từ ảnh.

## Cài đặt và chạy

### Yêu cầu

Máy tính cần cài đặt sẵn Python. Các tác vụ còn lại được các tệp `.bat` trong thư mục `bat\` thực hiện tự động, bao gồm cài đặt thư viện còn thiếu theo `requirements.txt`, thiết lập đường dẫn và truyền tham số phù hợp. Trường hợp sử dụng môi trường ảo, hãy tạo `.venv` trước khi chạy, các tệp này sẽ tự động nhận diện.

### Dữ liệu

Quy trình đọc dữ liệu từ `export/data.yaml`, một bộ dữ liệu YOLO gồm ba thư mục đã được cố định theo hạt giống 13. Dữ liệu được đặt tại thư mục gốc của dự án theo cấu trúc sau:

```
export/
  data.yaml                     # khai báo train val test, nc=5, names
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Nhãn tuân theo chuẩn YOLO. Mỗi ảnh tương ứng một tệp `.txt` cùng tên, mỗi dòng có dạng `class cx cy w h` đã chuẩn hóa về khoảng 0..1. Năm lớp được giữ cố định theo thứ tự cardboard 0, paper 1, glass 2, metal 3, plastic 4.

Phép tách dữ liệu đã được cố định nên thông thường không cần tạo lại. Trường hợp cần dựng lại `export/` từ ảnh gốc, hãy đặt ảnh vào `Dataset/images/` và nhãn vào `Dataset/labels/` trong cùng một thư mục, sau đó chạy `bat\00_prepare_data.bat`.

### Các bước chạy

Mỗi tệp `.bat` có thể được thực thi bằng cách nhấp đúp hoặc gõ tên tệp trong Command Prompt. Các bước được thực hiện lần lượt theo thứ tự sau.

| File | Dùng để | Khi nào chạy |
|---|---|---|
| `00_prepare_data.bat` | Dựng `export/` từ `Dataset/` gốc (kiểm định, tách tập, xuất dữ liệu, biểu đồ) | Tùy chọn, bỏ qua nếu đã có `export/` |
| `01_baseline.bat` | Huấn luyện trên 100% dữ liệu | Bước 1 |
| `02_al_random.bat` | S0 Random, ba hạt giống | Bước 2 |
| `03_al_uncertainty.bat` | S1 Uncertainty, ba hạt giống | Bước 3 |
| `04_al_coreset.bat` | S2 CoreSet, ba hạt giống | Bước 4 |
| `05_al_ppal.bat` | S3 PPAL, ba hạt giống | Bước 5 |
| `06_report.bat` | Tổng hợp biểu đồ, bảng AUBC, t-test | Sau khi xong baseline và cả bốn chiến lược |
| `07_demo.bat` | Mở ứng dụng web demo Gradio | Yêu cầu đã huấn luyện baseline |

Bốn tệp AL từ 02 đến 05 độc lập với nhau và có thể chạy theo thứ tự bất kỳ, nhưng cần hoàn thành cả bốn trước khi chạy `06_report.bat`.

### Kaggle mẫu
Đây là link Kaggle được sử dụng để huấn luyện cho Random, với file code và tập dữ liệu ảnh được up lên ở mục input của Kaggle.
Tuy nhiên có thể đổi config ở cell 2 để chạy các AL khác.
[![Kaggle](https://img.shields.io/badge/Kaggle-Click%20Me-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/nhichu/b23dcce075-al-kaggle)
### Demo

Tệp `07_demo.bat` khởi chạy ứng dụng demo tại `http://127.0.0.1:7860`. 
