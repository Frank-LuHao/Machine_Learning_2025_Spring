import numpy as np
import time
from cuml.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def train(dim=10, C=1.0, gamma='scale'):
    # 加载数据
    X_pca = np.load(f'./data/mnist_pca_dim{dim}.npy').astype(np.float32)
    y = np.load(f'./data/mnist_labels_dim{dim}.npy').astype(np.int32)

    # 标准化
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    all_y_true = []
    all_y_pred = []
    all_train_times = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_pca, y), 1):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        start_time = time.time()
        svm.fit(X_train, y_train)
        end_time = time.time()
        y_pred = svm.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_train_times.append(end_time - start_time)

        print(f"Fold {fold} 训练时间: {end_time - start_time:.2f} 秒")
        print(f"Fold {fold} 准确率: {acc:.4f}")
        print(f"Fold {fold} F1分数（macro）: {f1:.4f}")
        del svm

    print(f"\n平均准确率: {np.mean(acc_list):.4f}")
    print(f"平均F1分数（macro）: {np.mean(f1_list):.4f}")
    print(f"平均训练时间: {np.mean(all_train_times):.2f} 秒")
    print("总体分类报告:\n", classification_report(all_y_true, all_y_pred))

if __name__ == "__main__":
    for dim in [10, 20, 30, 50]:
        print(f"\n=== 训练 SVM 模型，降维维度: {dim} ===")
        train(dim)