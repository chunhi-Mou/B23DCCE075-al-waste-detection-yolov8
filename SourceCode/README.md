# Mã nguồn pipeline Active Learning phát hiện rác thải với YOLOv8

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![YOLOv8n](https://img.shields.io/badge/YOLOv8n-Ultralytics-00BFA5)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316?logo=gradio&logoColor=white)
![Windows](https://img.shields.io/badge/Run-Windows%20.bat-0078D6?logo=windows&logoColor=white)
![PTIT](https://img.shields.io/badge/PTIT-2026-C2185B)

Thư mục này chứa toàn bộ mã nguồn cần thiết để tái lập pipeline của đề tài: chuẩn bị dữ liệu, huấn luyện mô hình đối chứng, chạy bốn chiến lược Active Learning, tổng hợp báo cáo và ứng dụng web demo. Thư mục không bao gồm dữ liệu và trọng số mô hình.

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Chạy pipeline](#chạy-pipeline)
- [Dữ liệu gốc](#dữ-liệu-gốc)

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

### Gói `albench/`

Gói Python triển khai toàn bộ logic benchmark.

| Thành phần | Vai trò |
|---|---|
| `config.py` | Đọc và hợp nhất cấu hình từ thư mục `configs/` |
| `repro.py` | Cố định hạt giống và tính mã băm để bảo đảm tái lập |
| `al/` | Engine Active Learning (xem bảng dưới) |
| `data/` | Quét, kiểm định và tách dữ liệu thô |

**`albench/al/` — engine Active Learning**

| Tệp | Vai trò |
|---|---|
| `loop.py` | Vòng lặp chính mỗi lần chạy: chấm điểm kho dữ liệu, chọn lô ngân sách, huấn luyện lại, lặp |
| `select_random.py` | ![S0](https://img.shields.io/badge/S0-Random-9E9E9E) Chọn ngẫu nhiên phân phối đều |
| `select_uncertainty.py` | ![S1](https://img.shields.io/badge/S1-Uncertainty-1E88E5) Chọn theo độ bất định |
| `select_coreset.py` | ![S2](https://img.shields.io/badge/S2-CoreSet-43A047) Chọn theo tính đa dạng |
| `select_ppal.py` | ![S3](https://img.shields.io/badge/S3-PPAL-8E24AA) Chiến lược kết hợp hai giai đoạn |
| `ppal_difficulty.py`, `ppal_features.py`, `ppal_stage2.py`, `ccms.py` | Thành phần riêng của PPAL: độ bất định hiệu chỉnh theo độ khó, trích xuất đặc trưng, chọn đa dạng giai đoạn hai, tái cân bằng theo lớp |
| `score_predictor.py` | Suy luận để chấm điểm kho dữ liệu chưa gán nhãn |
| `init_set.py` | Tạo tập khởi tạo bảo đảm phủ lớp |
| `dataset.py` | Liệt kê kho dữ liệu, xác định lớp từng ảnh, ghi `data.yaml` cho mỗi vòng |
| `state.py` | Lưu và đọc trạng thái từng vòng để tiếp tục khi gián đoạn |
| `device.py` | Chọn thiết bị theo thứ tự CUDA, MPS, CPU |
| `metrics.py`, `stats.py`, `health.py` | Đo mAP, kiểm định thống kê, kiểm tra tính hợp lệ kết quả |
| `charts.py`, `ppal_charts.py`, `tables.py`, `report_io.py` | Vẽ biểu đồ, lập bảng AUBC, đọc ghi tệp kết quả |

**`albench/data/` — xử lý dữ liệu thô**

| Tệp | Vai trò |
|---|---|
| `labels.py` | Quét ảnh và nhãn |
| `audit.py` | Kiểm định dataset |
| `split.py` | Tách tập train, val, test |

### `bat/`

Các tệp chạy tự động trên Windows, mỗi tệp phụ trách một bước và chạy riêng lẻ. Mỗi tệp tự chuyển về thư mục gốc của bundle, thiết lập đường dẫn, cài thư viện còn thiếu và truyền tham số phù hợp.

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

### `configs/`

| Tệp | Vai trò |
|---|---|
| `benchmark.yaml` | Mọi tham số pipeline: dữ liệu, cấu hình huấn luyện, lịch trình Active Learning, tham số từng chiến lược |
| `seeds.yaml` | Khai báo tập hạt giống dùng cho thí nghiệm |

### `scripts/`

Các script dòng lệnh được các tệp `bat/` và notebook `kaggle/` gọi tới.

| Script | Vai trò |
|---|---|
| `01_audit_dataset.py` | Kiểm định dataset thô và xuất báo cáo |
| `02_make_splits.py` | Tách tập train, val, test đã cố định và ghi mã băm |
| `03_train_baseline.py` | Huấn luyện mô hình đối chứng |
| `06_export_dataset.py` | Xuất dữ liệu thành cấu trúc ba thư mục độc lập trong `export/` |
| `07_distribution_charts.py` | Vẽ biểu đồ phân bố lớp |
| `10_run_al.py` | Điều phối chạy các chiến lược Active Learning theo từng hạt giống |
| `11_al_report.py` | Tổng hợp kết quả thành biểu đồ, bảng và kiểm định thống kê |

### `kaggle/`

Hai notebook chạy trên Kaggle với GPU T4, đọc mã nguồn và dữ liệu từ các Kaggle Dataset.

| Notebook | Vai trò |
|---|---|
| `kaggle_baseline.ipynb` | Huấn luyện mô hình đối chứng trên 100% dữ liệu |
| `kaggle_al.ipynb` | Chạy bốn chiến lược Active Learning và tổng hợp báo cáo |

### `demo/`

Ứng dụng web demo viết bằng Gradio.

| Thành phần | Vai trò |
|---|---|
| `app.py` | Dựng giao diện và kết nối các thành phần |
| `engine.py` | Tìm trọng số mô hình và chạy suy luận trên ảnh tải lên |
| `content.py` | Định dạng kết quả tiếng Việt và các thành phần hiển thị |
| `charts.py` | Vẽ biểu đồ kết quả benchmark trong ứng dụng |
| `icons.py` | Biểu tượng dạng SVG |
| `examples/` | Ảnh mẫu cho phần thử nhanh |
| `README.md` | Hướng dẫn riêng cho ứng dụng demo |

## Chạy pipeline

Mã nguồn không kèm dữ liệu. Trước khi chạy, đặt thư mục dữ liệu `export/` cùng cấp với `albench/` và `bat/`, theo cấu trúc sau.

```
export/
├─ data.yaml
├─ train/images/  train/labels/
├─ val/images/    val/labels/
└─ test/images/   test/labels/
```

Trên Windows, chạy lần lượt các tệp trong `bat/` theo thứ tự `01_baseline.bat`, bốn tệp AL từ `02` đến `05`, sau đó `06_report.bat` để tổng hợp kết quả và `07_demo.bat` để mở ứng dụng demo. Bốn tệp AL độc lập với nhau và có thể chạy theo thứ tự bất kỳ, nhưng cần hoàn thành cả bốn trước khi chạy `06_report.bat`.

## Dữ liệu gốc

Thư mục `export/` ở trên đã được chia sẵn và cố định nên thông thường đủ để chạy toàn bộ pipeline. Chỉ khi cần dựng lại `export/` từ đầu bằng `00_prepare_data.bat` mới cần dữ liệu gốc, đặt vào thư mục `Dataset/` cùng cấp với `albench/` và `bat/`, theo cấu trúc phẳng sau.

```
Dataset/
├─ images/      Toàn bộ ảnh .jpg .jpeg hoặc .png, chưa chia tập
└─ labels/      Nhãn .txt tương ứng, cùng tên tệp với ảnh
```

Mỗi ảnh có một tệp nhãn cùng tên, ghép theo phần tên không tính phần mở rộng. Nhãn tuân theo chuẩn YOLO, mỗi dòng gồm đúng năm trường `class cx cy w h`. Trường `class` là số nguyên từ 0 đến 4 theo thứ tự cardboard, paper, glass, metal, plastic, còn `cx cy w h` đã chuẩn hóa về khoảng 0..1. Khác với `export/`, thư mục `Dataset/` để toàn bộ ảnh trong một thư mục duy nhất, việc chia train, val, test do `00_prepare_data.bat` thực hiện.
