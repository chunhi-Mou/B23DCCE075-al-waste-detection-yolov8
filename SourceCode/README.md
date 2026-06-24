# Mã nguồn pipeline Active Learning phát hiện rác thải với YOLOv8

Thư mục này chứa toàn bộ mã nguồn cần thiết để tái lập pipeline của khóa luận, bao gồm chuẩn bị dữ liệu, huấn luyện mô hình đối chứng, chạy bốn chiến lược Active Learning, tổng hợp báo cáo và ứng dụng web demo. Thư mục không bao gồm dữ liệu và trọng số mô hình.

## Cấu trúc thư mục

```
README.md                 Tài liệu này
requirements.txt          Danh sách thư viện kèm phiên bản đã ghim
requirements.lock.txt     Bản khóa đầy đủ của mọi phụ thuộc
pyproject.toml            Khai báo gói albench
albench/                  Gói Python lõi của benchmark
bat/                      Tệp chạy tự động trên Windows
configs/                  Tham số cấu hình
scripts/                  Các script dòng lệnh
kaggle/                   Notebook chạy trên Kaggle
demo/                     Ứng dụng web demo Gradio
```

### albench/

Gói Python triển khai toàn bộ logic benchmark.

- `config.py` đọc và hợp nhất cấu hình từ thư mục `configs/`.
- `repro.py` cố định hạt giống và tính mã băm nhằm bảo đảm khả năng tái lập.
- `al/` chứa engine Active Learning.
  - `loop.py` là vòng lặp chính của mỗi lần chạy, gồm các bước chấm điểm kho dữ liệu, chọn lô ngân sách, huấn luyện lại và lặp.
  - `select_random.py`, `select_uncertainty.py`, `select_coreset.py`, `select_ppal.py` lần lượt là bốn chiến lược chọn mẫu S0 đến S3.
  - `ppal_difficulty.py`, `ppal_features.py`, `ppal_stage2.py`, `ccms.py` là các thành phần riêng của PPAL, gồm độ bất định hiệu chỉnh theo độ khó, trích xuất đặc trưng, chọn đa dạng giai đoạn hai và tái cân bằng theo lớp.
  - `score_predictor.py` chạy suy luận để chấm điểm kho dữ liệu chưa gán nhãn.
  - `init_set.py` tạo tập khởi tạo bảo đảm phủ lớp.
  - `dataset.py` liệt kê kho dữ liệu, xác định lớp của từng ảnh và ghi tệp `data.yaml` cho mỗi vòng.
  - `state.py` lưu và đọc trạng thái từng vòng để có thể tiếp tục khi gián đoạn.
  - `device.py` chọn thiết bị tính toán theo thứ tự CUDA, MPS, CPU.
  - `metrics.py`, `stats.py`, `health.py` đo mAP, chạy kiểm định thống kê và kiểm tra tính hợp lệ của kết quả.
  - `charts.py`, `ppal_charts.py`, `tables.py`, `report_io.py` vẽ biểu đồ, lập bảng AUBC và đọc ghi tệp kết quả.
- `data/` xử lý dữ liệu thô.
  - `labels.py`, `audit.py`, `split.py` quét nhãn, kiểm định và tách tập train, val, test.

### bat/

Các tệp chạy tự động trên Windows, mỗi tệp phụ trách một bước, chạy riêng lẻ. Mỗi tệp tự chuyển về thư mục gốc của bundle, thiết lập đường dẫn, cài thư viện còn thiếu và truyền tham số phù hợp.

| Tệp | Chức năng |
|---|---|
| `00_prepare_data.bat` | Dựng `export/` từ `Dataset/` gốc |
| `01_baseline.bat` | Huấn luyện mô hình đối chứng trên 100% dữ liệu |
| `02_al_random.bat` | Chạy chiến lược S0 Random |
| `03_al_uncertainty.bat` | Chạy chiến lược S1 Uncertainty |
| `04_al_coreset.bat` | Chạy chiến lược S2 CoreSet |
| `05_al_ppal.bat` | Chạy chiến lược S3 PPAL |
| `06_report.bat` | Tổng hợp biểu đồ, bảng AUBC và kiểm định t-test |
| `07_demo.bat` | Khởi chạy ứng dụng web demo |

### configs/

- `benchmark.yaml` chứa mọi tham số của pipeline, gồm thông tin dữ liệu, cấu hình huấn luyện, lịch trình Active Learning và tham số từng chiến lược.
- `seeds.yaml` khai báo tập hạt giống dùng cho thí nghiệm.

### scripts/

Các script dòng lệnh được các tệp trong `bat/` và notebook trong `kaggle/` gọi tới.

- `01_audit_dataset.py` kiểm định dataset thô và xuất báo cáo.
- `02_make_splits.py` tách tập train, val, test đã cố định và ghi mã băm.
- `03_train_baseline.py` huấn luyện mô hình đối chứng.
- `06_export_dataset.py` xuất dữ liệu thành cấu trúc ba thư mục độc lập trong `export/`.
- `07_distribution_charts.py` vẽ biểu đồ phân bố lớp.
- `10_run_al.py` điều phối việc chạy các chiến lược Active Learning theo từng hạt giống.
- `11_al_report.py` tổng hợp kết quả thành biểu đồ, bảng và kiểm định thống kê.

### kaggle/

Hai notebook để chạy trên Kaggle với GPU P100, đọc mã nguồn và dữ liệu từ các Kaggle Dataset.

- `kaggle_baseline.ipynb` huấn luyện mô hình đối chứng trên 100% dữ liệu.
- `kaggle_al.ipynb` chạy bốn chiến lược Active Learning và tổng hợp báo cáo.

### demo/

Ứng dụng web demo viết bằng Gradio, chỉ suy luận và không huấn luyện.

- `app.py` dựng giao diện và kết nối các thành phần.
- `engine.py` tìm trọng số mô hình và chạy suy luận trên ảnh tải lên.
- `content.py` định dạng kết quả tiếng Việt và các thành phần hiển thị.
- `charts.py` vẽ biểu đồ kết quả benchmark trong ứng dụng.
- `icons.py` chứa biểu tượng dạng SVG.
- `examples/` chứa ảnh mẫu cho phần thử nhanh.
- `README.md` hướng dẫn riêng cho ứng dụng demo.

## Chạy pipeline

Mã nguồn không kèm dữ liệu. Trước khi chạy, đặt thư mục dữ liệu `export/` cùng cấp với `albench/` và `bat/`, theo cấu trúc sau.

```
export/
├─ data.yaml
├─ train/images/  train/labels/
├─ val/images/    val/labels/
└─ test/images/   test/labels/
```

Trên Windows, chạy lần lượt các tệp trong `bat/` theo thứ tự `01_baseline.bat`, bốn tệp AL từ `02` đến `05`, sau đó `06_report.bat` để tổng hợp kết quả và `07_demo.bat` để mở ứng dụng demo.

## Dữ liệu gốc

Thư mục `export/` ở trên đã được chia sẵn và đóng băng nên thông thường đủ để chạy toàn bộ pipeline. Chỉ khi cần dựng lại `export/` từ đầu bằng `00_prepare_data.bat` mới cần dữ liệu gốc, đặt vào thư mục `Dataset/` cùng cấp với `albench/` và `bat/`, theo cấu trúc phẳng sau.

```
Dataset/
├─ images/      Toàn bộ ảnh định dạng .jpg .jpeg hoặc .png, chưa chia tập
└─ labels/      Nhãn .txt tương ứng, cùng tên tệp với ảnh
```

Mỗi ảnh có một tệp nhãn cùng tên, ghép theo phần tên không tính phần mở rộng. Nhãn tuân theo chuẩn YOLO, mỗi dòng gồm đúng năm trường `class cx cy w h`. Trường `class` là số nguyên từ 0 đến 4 theo thứ tự cardboard, paper, glass, metal, plastic, còn `cx cy w h` đã chuẩn hóa về khoảng 0..1. Khác với `export/`, thư mục `Dataset/` để toàn bộ ảnh trong một thư mục duy nhất, việc chia train, val, test do `00_prepare_data.bat` thực hiện.
