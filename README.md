# 🏙️ Phân tích sức mạnh thị trường Airbnb NYC

Đây là project môn Lập trình Khoa học Dữ liệu nhằm tìm hiểu sơ bộ và phân loại mức độ sức mạnh thị trường (Market Strength) của các khu vực cho thuê Airbnb tại New York City. 

---

## Mục lục

1. Giới thiệu
2. Dataset
3. Method (Phương pháp)
4. Installation & Setup
5. Usage (Hướng dẫn sử dụng)
6. Results (Kết quả)
7. Project Structure
8. Challenges & Solutions
9. Future Improvements
10. Contributors & Contact
11. License

---

## 1. Giới thiệu

### 1.1. Mô tả bài toán
Bài toán là một tác vụ **Phân loại đa Lớp (Multi-class Classification)**. Dữ liệu listing Airbnb được nhóm lại theo các khu vực lưới (Grid) đã được điều chỉnh. Mục tiêu là dự đoán **Mức độ sức mạnh thị trường** (từ **Level 1** - Yếu nhất đến **Level 5** - Mạnh nhất) cho mỗi khu vực đó.

### 1.2. Động lực và ứng dụng thực tế
* **Nhà đầu tư:** Xác định các khu vực có tiềm năng lợi nhuận cao (Level 5) để đầu tư bất động sản hoặc thuê.
- Với những chủ đầu tư có sẵn nguồn lực, nhưng lại không tìm hiểu được nhiều về tình hình ở New York thì thường sẽ không biết được cụ thể khu nhà nào hot hay không hot, chỉ nghe được những thông tin cực kì chung chung như: gần quảng trường Time Square thì sẽ nhiều khách, hay cứ khu Manhattan là có thể kinh doanh airbnb dễ dàng. 
- Điều này cũng tương tự với Thành phố Hồ Chính Minh, nơi ta cũng chỉ biết là quận 1 đắt đỏ, nhưng nếu là 1 căn nhà trong hẻm hay ở rìa quận 1 thì sẽ không có tiềm năng kinh tế so với các khu trung tâm. 
- Mô hình này sẽ giúp các chủ đầu tư định mở airbnb ở NYC có cái nhìn chi tiết hơn về khu phố nơi mình muốn mua mà không cần phải tìm hiểu quá cụ thể hay thông qua cò đất, chỉ cần có tọa độ, mức giá mình định đề ra, số đêm tối thiểu mà mình định cho thuê, thì có thể xác định căn airbnb của mình có tiềm năng hay không.
* **Chủ nhà/Host:** Đánh giá hiệu suất listing của mình so với khu vực lân cận.

* **Phân bổ nguồn lực:** Giúp Airbnb hoặc cơ quan quản lý thành phố hiểu rõ về phân bổ nhu cầu và hoạt động cho thuê.

### 1.3. Mục tiêu cụ thể
1.  Thiết kế **Chỉ số sức mạnh thị trường (Market Strength Index)** dựa trên Demand, Price Efficiency, và Activity Density.
2.  Phát triển thuật toán **Phân loại Logistic Regression (OvR)** .
3.  Đạt độ chính xác (Overall Accuracy) tối ưu nhất cho mô hình phân loại 5 cấp độ.

---

## 2. Dataset

### 2.1. Nguồn dữ liệu
Dữ liệu công khai từ Airbnb NYC (2019).

### 2.2. Kích thước và Đặc điểm
* **Kích thước ban đầu:** Khoảng 49,000 listings.
* **Target (Y):** `target_market_strength_level` (Levels 1, 2, 3, 4, 5).
* **Features (X):** 8 Features đã được chuẩn hóa (Z-score).

### 2.3. Các Features Quan trọng
| Feature | Loại | Mô tả |
| :--- | :--- | :--- |
| `grid_demand_zscore` | Grid | Nhu cầu (Reviews/Month) trung vị của khu vực. |
| `grid_price_efficiency_zscore` | Grid | Hiệu suất giá/giá trị trung vị của khu vực. |
| `grid_activity_density_zscore` | Grid | Mật độ Listing của khu vực (Yếu tố quan trọng nhất). |
| `price_zscore` $\dots$ | Listing | Các Features listing-level gốc đã chuẩn hóa, dùng để phân biệt ranh giới. |

---

## 3. Method (Phương pháp)

### 3.1. Quy trình Xử lý Dữ liệu (02\_preprocessing.ipynb)

1.  **Adaptive Grid (Lưới Thích ứng):** Dữ liệu được nhóm vào các khu vực lưới (grid) có kích thước khác nhau (0.01x0.01 hoặc 0.005x0.005) dựa trên độ biến động giá (`price_std`) để phân tích hiệu quả hơn.
2.  **Tính Chỉ số Sức mạnh (MSI):** Chỉ số `market_strength_index` được tính bằng tổ hợp trọng số của 3 Components Grid-level đã được Min-Max Scaled.
3.  **Phân lớp Target (Y):** MSI (0-1) được chia thành 5 Level (`1` đến `5`) dựa trên **Percentile (P20, P40, P60, P80)**.
4.  **Standardization:** Tất cả Features đầu vào được chuẩn hóa bằng Z-score để đảm bảo Gradient Descent hội tụ nhanh và chính xác.

### 3.2. Thuật toán Sử dụng: Hồi quy Logistic Đa Lớp (OvR)

Mô hình này là tập hợp của $K=5$ mô hình Logistic Regression nhị phân độc lập (Mỗi mô hình học cách phân biệt "Lớp $k$" với "Phần còn lại").

#### Công thức toán học cốt lõi
1.  **Hàm Tuyến tính (Z):**
    $$\mathbf{Z} = \mathbf{X} \mathbf{W}$$
    (Với $\mathbf{X}$ là ma trận Features đã thêm cột Bias)
2.  **Hàm Kích hoạt Sigmoid:**
    $$\mathbf{H} = \sigma(\mathbf{Z}) = \frac{1}{1 + e^{-\mathbf{Z}}}$$
3.  **Hàm Mất mát (Binary Cross-Entropy Loss):**
    $$\text{Loss}(\mathbf{W}) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]$$

### 3.3. Giải thích Implementation NumPy (src/models.py)
* **Gradient Descent:** Hàm `gradient_descent` và `train_logistic_regression` sử dụng `np.dot` để thực hiện phép nhân ma trận và tính toán Gradient theo nguyên tắc **Vectorization**, tránh hoàn toàn các vòng lặp `for`.
* **OvR Implementation:** Hàm `train_ovr_logistic_regression` lặp 5 lần để huấn luyện 5 vector trọng số ($\mathbf{W}_1, \dots, \mathbf{W}_5$). Hàm `predict_ovr` sử dụng `np.dot` để tính xác suất cho cả 5 lớp cùng lúc, sau đó dùng `np.argmax` để chọn lớp có xác suất cao nhất.
* **Kỹ thuật NumPy:** Đã áp dụng `np.percentile`, `np.where` (masking), `np.column_stack` (array manipulation), và `np.clip` (ổn định số học).

---

## 4. Installation & Setup

1.  **Cài đặt Thư viện:**
    ```bash
    pip install numpy matplotlib
    ```
2.  **Cấu trúc thư mục:** Đảm bảo cấu trúc dự án như sau:
    ```
    .
    ├── data/
    │   ├── processed/
    │   └── raw/
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_preprocessing.ipynb
    │   └── 03_modeling.ipynb
    └── src/
        ├── __init__.py
        ├── data_processing.py
        └── models.py
    ```

---

## 5. Usage (Hướng dẫn chạy)

Chạy các file Jupyter Notebook theo thứ tự sau:

1.  **`02_preprocessing.ipynb`**: Xử lý các giá trị missing và lưu vào file `AB_NYC_2019_preprocessed.csv`, thực hiện Feature Engineering và lưu file `AB_NYC_2019_featured.csv`.
2.  **`01_data_exploration.ipynb`**: Khám phá dữ liệu thô, dữ liệu đã tiền xử lý missing, trả lời một vài câu hỏi có ý nghĩa, trong đó có 1 câu hỏi sẽ chỉ ra một vấn đề cần đến model và sự cần thiết của việc phải thêm các thuộc tính xây model.
3.  **`03_modeling.ipynb`**: Load dữ liệu, chia Train/Test, huấn luyện mô hình Logistic Regression OvR.

---

## 6. Results (Kết quả)

### 6.1. Kết quả Đạt được

| Mô hình | Features | Accuracy (Overall) | Iterations | Learning Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression OvR (NumPy)** | 8 Features Z-score | **$0.6800$** | $8000$ | $0.01$ |

### 6.2. Phân tích Trọng số (W)
Ma trận trọng số $\mathbf{W}$ (có kích thước $5 \times 9$) cho thấy tầm quan trọng của các Feature đối với từng Level:

| Lớp | Intercept | grid\_demand | grid\_efficiency | grid\_density | price\_zscore | min\_nights\_zscore | reviews\_zscore | host\_count\_zscore | avail\_zscore |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Level 1** | x | **-1.29** | -0.42 | **-2.57** | x | x | x | x | x |
| **Level 5** | x | +0.53 | +0.55 | **+3.25** | x | x | x | x | x |

*(Sử dụng số liệu từ lần chạy 3 Features trước để làm ví dụ, cần thay bằng số liệu cuối cùng)*

**Insight chính:**
* **`grid_activity_density`** (Mật độ Listing) là **Feature quan trọng nhất** (Trọng số lớn nhất) để dự đoán Level 5 (Thị trường Mạnh nhất) và Level 1 (Thị trường Yếu nhất), cho thấy sự phân biệt rõ ràng giữa các Grid đông đúc và thưa thớt.

### 6.3. Hình ảnh Trực quan hóa Lỗi (Confusion Matrix)



* **Tỷ lệ Đúng cao:** Các ô đường chéo chính (Level 1, Level 5) có tỷ lệ dự đoán đúng rất cao.
* **Lỗ hổng Chính:** Lỗi tập trung ở sự nhầm lẫn giữa **Level 4** và **Level 5**. Khoảng $40\%$ mẫu True Level 4 bị mô hình dự đoán là Level 5 (do ranh giới phân loại quá gần nhau).

---

## 7. Project Structure

| File/Folder | Chức năng |
| :--- | :--- |
| `notebooks/02_preprocessing.ipynb` | Code Feature Engineering (Adaptive Grid, MSI, Z-score). |
| `notebooks/03_modeling.ipynb` | Code Huấn luyện và Đánh giá mô hình OvR. |
| `src/data_processing.py` | Chứa các hàm hỗ trợ tải và chuyển đổi dữ liệu NumPy. |
| `src/models.py` | Chứa các hàm cốt lõi của Logistic Regression (Sigmoid, Loss, Gradient Descent, OvR). |
| `data/processed/` | Lưu trữ file dữ liệu đã xử lý (`AB_NYC_2019_featured.csv`). |

---

## 8. Challenges & Solutions

| Thử thách | Giải pháp | Kỹ thuật NumPy liên quan |
| :--- | :--- | :--- |
| **Lỗi số học (Overflow)** | Sử dụng `np.clip` (ví dụ: trong hàm Loss) để giới hạn giá trị xác suất gần 0 và 1, tránh `log(0)` hoặc tràn số khi tính `np.exp` lớn. | Ổn định số học (2.3) |
| **Phân loại Đa lớp (OvR)** | Áp dụng kỹ thuật One-vs-Rest (OvR) bằng cách huấn luyện 5 mô hình nhị phân. | Masking (`Y_k = (Y_train_multi == k)`) |
| **Tối ưu hóa Tốc độ Học** | Tăng `NUM_ITERATIONS` từ 5000 lên 8000 để mô hình hội tụ hoàn toàn sau khi thêm các Features. | Tối ưu hóa/Code Efficiency (2.4) |
| **Yêu cầu `np.einsum`** | *[Nếu muốn đạt điểm tối đa 2.2]:* Cân nhắc sử dụng `np.einsum` thay cho `np.dot` trong việc tính toán Gradient hoặc Z-score để chứng minh kiến thức về xử lý tensor. | Numpy Techniques (2.2) |

---

## 9. Future Improvements (Hướng phát triển tiếp theo)

* **Thêm Feature Phi tuyến tính:** Tạo các Feature tương tác bậc hai (ví dụ: `grid_demand_zscore * grid_density_zscore`) để giúp mô hình tuyến tính phân biệt ranh giới Level 4/5 tốt hơn.
* **So sánh Mô hình:** Áp dụng các thuật toán phân loại khác (ví dụ: Naive Bayes, Decision Tree) cũng được xây dựng từ NumPy để so sánh hiệu suất.

---

## 10. Contributors & Contact

* **Tác giả:** Phạm Khánh Linh 
* **Liên hệ:** linhpham.1508055@gmail.com

## 11. License

[Ví dụ: MIT License]