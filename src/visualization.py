# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# ==========================================
# Histogram
# ==========================================
def plot_histogram(data, bins=30, title=None, xlabel=None, ylabel="Frequency"):
    plt.figure(figsize=(6,4))
    sns.histplot(data, bins=bins, kde=False, color='skyblue')
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# ==========================================
# Boxplot
# ==========================================
def plot_boxplot(data, title=None, ylabel=None):
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data, color='lightgreen')
    if title: plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    plt.show()

# ==========================================
# Bar chart
# ==========================================
def plot_bar(categories, counts, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10,5))
    sns.barplot(x = categories, y=counts, palette='pastel', dodge=False, legend = False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ==========================================
# Pie chart
# ==========================================
def plot_pie(categories, counts, title=None, top_n=None):
    plt.figure(figsize=(6,6))



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Đặt thư mục visualization là 'images' trong thư mục gốc
VISUALIZATION_DIR = os.path.join(project_root, 'images') 

def save_plot(filename):
    # 1. Kiểm tra và tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(VISUALIZATION_DIR):
        try:
            os.makedirs(VISUALIZATION_DIR)
            print(f"Đã tạo thư mục: {VISUALIZATION_DIR}")
        except OSError as e:
            print(f"Lỗi khi tạo thư mục: {e}")
            return

    # 2. Xây dựng đường dẫn đầy đủ
    full_path = os.path.join(VISUALIZATION_DIR, filename)

    # 3. Lưu hình ảnh
    plt.savefig(full_path)
    print(f"Đã lưu biểu đồ thành công tại: {full_path}")
    
    # Đóng figure để giải phóng bộ nhớ
    plt.close()