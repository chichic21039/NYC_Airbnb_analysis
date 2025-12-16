# src/models.py

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. CÁC HÀM CƠ SỞ (CORE FUNCTIONS)
# ==========================================================

def sigmoid(z):
    """
    Hàm kích hoạt Sigmoid: H = 1 / (1 + exp(-z))
    Args:
        z (np.array): Đầu vào tuyến tính (X * W).
    """
    return 1 / (1 + np.exp(-z))

def compute_loss(Y, H):
    """
    Tính Binary Cross-Entropy Loss (Hàm mất mát)
    Args:
        Y (np.array): Target nhị phân (0 hoặc 1).
        H (np.array): Đầu ra dự đoán (xác suất Sigmoid).
    """
    m = len(Y)
    epsilon = 1e-15
    H_clipped = np.clip(H, epsilon, 1 - epsilon)
    
    # Vectorization cho Cross-Entropy Loss
    loss = (-1/m) * np.sum(Y * np.log(H_clipped) + (1 - Y) * np.log(1 - H_clipped))
    return loss

def gradient_descent(X, Y, W, learning_rate):
    """
    Tính toán Gradient và cập nhật Trọng số W cho một mô hình nhị phân.
    """
    m = len(Y)
    Z = np.dot(X, W)
    H = sigmoid(Z)
    gradient = (1/m) * np.dot(X.T, (H - Y)) 
    W = W - learning_rate * gradient
    
    return W

def train_logistic_regression(X_train, Y_train_binary, num_iterations, learning_rate):
    """
    Hàm huấn luyện chính cho một mô hình nhị phân.
    """
    num_features = X_train.shape[1]
    W = np.zeros(num_features) 
    loss_history = []
    
    for i in range(num_iterations):
        W = gradient_descent(X_train, Y_train_binary, W, learning_rate)
        
        # Tính Loss
        Z = np.dot(X_train, W)
        H = sigmoid(Z)
        loss = compute_loss(Y_train_binary, H)
        loss_history.append(loss)
        
        if i % (num_iterations // 10) == 0:
            print(f"  > Iteration {i}/{num_iterations}, Loss: {loss:.4f}")
            
    return W, loss_history

# ==========================================================
# 2. HÀM ĐA LỚP (ONE-VS-REST / OVR)
# ==========================================================

def train_ovr_logistic_regression(X_train, Y_train_multi, num_iterations, learning_rate):
    """
    Huấn luyện K mô hình Logistic Regression (OvR) cho Target 5 mức (1-5).
    """
    # Lấy các lớp duy nhất (phải là 1, 2, 3, 4, 5)
    unique_classes = np.unique(Y_train_multi) 
    K = len(unique_classes)
    W_ovr = [] 
    
    print(f"Bắt đầu Huấn luyện {K} Mô hình (One-vs-Rest)...")
    
    for k in unique_classes: 
        print(f"\n--- Huấn luyện Mô hình cho Lớp {int(k)} vs Phần còn lại ---")
        
        # 1. Tạo Target nhị phân Y_k: 1 nếu là lớp k, 0 nếu không phải
        Y_k = (Y_train_multi == k).astype(np.int8) 
        
        # 2. Huấn luyện mô hình nhị phân cho Y_k
        W_k, _ = train_logistic_regression(X_train, Y_k, num_iterations, learning_rate)
        W_ovr.append(W_k)
        
    return np.array(W_ovr)

def predict_ovr(X, W_ovr):
    """
    Dự đoán lớp (1-5) bằng cách chọn mô hình có xác suất cao nhất.
    """
    # Z_scores: X * W_ovr.T (Kích thước: số mẫu x K)
    Z_scores = np.dot(X, W_ovr.T)
    
    # Xác suất (Kích thước: số mẫu x K)
    probabilities = sigmoid(Z_scores) 
    
    # Chọn chỉ mục (index) của xác suất cao nhất (index 0 -> Class 1)
    # np.argmax trả về index (0 đến 4). Lớp thực tế là index + 1
    predicted_classes = np.argmax(probabilities, axis=1) + 1 
    
    return predicted_classes

# ==========================================================
# 3. HÀM ĐÁNH GIÁ (METRICS VÀ TRỰC QUAN)
# ==========================================================

def calculate_accuracy(Y_true, Y_pred):
    """
    Tính Accuracy (Độ chính xác) cho phân loại đa lớp.
    """
    return np.mean(Y_true == Y_pred)

def plot_loss(loss_history):
    """
    Vẽ biểu đồ Loss History cho một mô hình nhị phân.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Lịch sử Loss (Binary Cross-Entropy)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # src/models.py (Thêm vào phần 3. HÀM ĐÁNH GIÁ (METRICS))

def compute_confusion_matrix(Y_true, Y_pred, num_classes=5):
    """
    Tính Confusion Matrix cho phân loại đa lớp (5x5).
    """
    # Khởi tạo ma trận rỗng (số lớp x số lớp)
    # Hàng là True Class (thực tế), Cột là Predicted Class (dự đoán)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Duyệt qua từng cặp giá trị (True, Predicted)
    for true_val, pred_val in zip(Y_true, Y_pred):
        # Vì lớp bắt đầu từ 1, ta trừ 1 để lấy index (0 đến 4)
        true_index = int(true_val) - 1
        pred_index = int(pred_val) - 1
        
        # Tăng số lượng tại ô (True Class, Predicted Class)
        cm[true_index, pred_index] += 1
        
    return cm

def plot_confusion_matrix(cm, class_labels):
    """
    Vẽ Confusion Matrix (sử dụng matplotlib/numpy).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # Sử dụng cax=ax.matshow để vẽ ma trận màu
    cax = ax.matshow(cm, cmap=plt.cm.Blues) 
    fig.colorbar(cax)
    
    # Thiết lập nhãn
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix (OvR Logistic Regression)')
    
    # Thêm số vào ô
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='red' if i != j else 'black')
            
    plt.show()