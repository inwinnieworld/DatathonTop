# DATATHON 2026: THE GRIDBREAKER
## Dự Báo Doanh Thu Thương Mại Điện Tử Thời Trang

**Nhóm thực hiện:** ontop (Trương Nguyên Đại Thắng, Lê Thị Hồng Đang, Lương Thị Hồng Nhung, Trần Võ Minh Hùng)  
**Tài liệu tham khảo chính:** Báo cáo Kỹ thuật (`report_neurips.pdf`)  
**Kết quả tốt nhất (Public Leaderboard):** MAE = **669.087** (Pipeline `v47`)

---

## 1. Cấu trúc thư mục (Repository Structure)

```text
DATATHONTOP/
│
├── figures/                     # Lưu trữ toàn bộ biểu đồ xuất ra từ code
│   ├── eda/                     # Biểu đồ từ quá trình Khám phá dữ liệu (EDA.ipynb)
│   ├── modelling/               # Biểu đồ chẩn đoán mô hình, Cross-validation
│   └── shap/                    # Biểu đồ giải thích mô hình (SHAP values)
│
├── EDA.ipynb                    # Notebook thực hiện EDA, phân tích Cohort, Pareto, Margin
├── modelling.ipynb              # Notebook chứa toàn bộ quá trình huấn luyện và lịch sử thử nghiệm
│
├── report_neurips.pdf           # Báo cáo kỹ thuật chính thức (giới hạn 4 trang main + Phụ lục)
├── report_neurips.tex           # Mã nguồn LaTeX của báo cáo
├── neurips_2025.sty             # File style template của NeurIPS
│
└── submission_v47_10seed.csv    # File kết quả dự báo TỐT NHẤT nộp trên Kaggle (từ pipeline v47)
```

---

## 2. Yêu cầu hệ thống 

Để chạy lại kết quả, môi trường máy tính cần cài đặt Python 3.10+ và các thư viện sau:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
pip install lightgbm xgboost scikit-learn prophet shap
```
*(Lưu ý: Trong notebook `modelling.ipynb` có các thử nghiệm với `chronos-forecasting` và `neuralforecast` ở một số version. Tuy nhiên, kết quả nộp bài tốt nhất - **v47** - KHÔNG sử dụng 2 thư viện này. Do đó, không bắt buộc phải cài đặt chúng nếu chỉ muốn tái lập kết quả submission chính).*

---

## 3. Hướng dẫn chạy lại kết quả 

Mã nguồn gốc được phát triển trên môi trường **Kaggle Notebook**. Để chạy trên máy tính cá nhân (Local) hoặc môi trường khác của Ban giám khảo, vui lòng thực hiện theo các bước sau:

### Bước 1: Chuẩn bị dữ liệu
Tải toàn bộ 15 file `.csv` của ban tổ chức cấp và đặt vào một thư mục chung (ví dụ: `./data/`).

### Bước 2: Cấu hình lại đường dẫn (Quan trọng)
Mở cả 2 file `EDA.ipynb` và `modelling.ipynb`, tìm đến các cell đầu tiên chứa cấu hình đường dẫn và thay đổi lại cho phù hợp với máy cá nhân:

```python
# Sửa lại đường dẫn này trỏ tới thư mục chứa file CSV dữ liệu
BASE_DIR = '/đường/dẫn/tới/thư/mục/chứa/data/' 
# Ví dụ local: BASE_DIR = './data/'

# Sửa lại đường dẫn này trỏ tới nơi bạn muốn lưu file output (CSV, Hình ảnh)
OUT_DIR  = '/đường/dẫn/tới/thư/mục/đầu/ra/'     
# Ví dụ local: OUT_DIR = './'
```

### Bước 3: Chạy phần Khám phá Dữ liệu (EDA)
* Mở và chạy `Run All` file `EDA.ipynb`.
* Kết quả: Code sẽ tự động tính toán các chỉ số thống kê và lưu các biểu đồ EDA vào đường dẫn `OUT_DIR` (như đã thiết lập ở Bước 2).

### Bước 4: Chạy Mô hình và Tái lập Submission tốt nhất (v47)
File `modelling.ipynb` là một hành trình thử nghiệm dài từ bản `v22` đến `v50`. Để tái lập chính xác kết quả của file `submission_v47_10seed.csv`, cần thực hiện như sau:

1. Chạy các cell cài đặt thư viện và hàm `build_features` ở đầu notebook.
2. Cuộn xuống và chạy Cell có tiêu đề: **`submission_v47_master.csv`** (Đây là bản mã nguồn tổng hợp đầy đủ: *LGB Q-Specialist 10 seeds + Ridge + Prophet + Level Correction*).
3. Kết quả đầu ra sẽ tự động sinh ra file `submission_v47_10seed.csv`.

### Bước 5: Chạy phần giải thích mô hình (SHAP)
* Ngay bên dưới cell của `v47` trong `modelling.ipynb`, nhóm đã tách riêng phần **SHAP EXPLAINABILITY ANALYSIS**.
* Chạy các cell này để tái tạo lại toàn bộ biểu đồ Beeswarm, Waterfall, Dependence Plots và Ma trận Tương quan được dùng trong báo cáo NeurIPS.

---

## 4. Tổng quan về Pipeline Dự báo (v47)
Dưới đây là tóm tắt kiến trúc của mô hình đem lại kết quả tốt nhất:

1. **Feature Engineering:** Tạo 94 đặc trưng hoàn toàn dựa trên lịch (`calendar-only` & `zero-lag`), bóc tách Fourier Seasonality, Khoảng cách Tết, Chu kỳ Ngày lương (Payday), và lịch Khuyến mãi tái tạo từ `promotions.csv`. Hoàn toàn không dùng dữ liệu bị rò rỉ (leakage).
2. **Kiến trúc Ensemble (Lớp 1):** 
   * **LightGBM Q-Specialist:** Tách mô hình học theo từng Quý (Q1-Q4) để bắt các dị thường (vd: Urban Blowout lỗ gộp ở Q3). Sử dụng 10 random seeds để giảm phương sai. Trọng số 80%.
   * **Ridge Regression:** Neo dự báo vào mức cơ sở tuyến tính. Trọng số 10%.
   * **Prophet:** Bắt xu hướng và tính mùa vụ Multiplicative. Trọng số 10%.
3. **Hiệu chỉnh Mức độ (Lớp 2 - Level Correction):** Sử dụng giả định tăng trưởng YoY của thị trường TMĐT Việt Nam từ báo cáo MoIT (2022) kết hợp Calibration Ratio (CR) sinh ra từ tập validation (Fold A) để hiệu chỉnh mức doanh thu mục tiêu.
4. **COGS Mapping:** Không dùng mô hình độc lập cho COGS để tránh sai số lũy kế. COGS được suy diễn trực tiếp từ Revenue dự báo và tỷ lệ biên lợi nhuận lịch sử (`Gross Margin per segment`).

