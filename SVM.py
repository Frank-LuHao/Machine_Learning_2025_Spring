import numpy as np
import time
from cuml.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def train(X_trainval, y_trainval, C=1.0, gamma='scale'):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    all_y_true = []
    all_y_pred = []
    all_train_times = []

    for fold, (train_idx, validation_idx) in enumerate(kf.split(X_trainval, y_trainval), 1):
        X_train, X_validation = X_trainval[train_idx], X_trainval[validation_idx]
        y_train, y_validation = y_trainval[train_idx], y_trainval[validation_idx]
        
        svm = SVC(kernel='rbf', C=C, gamma=gamma)
        start_time = time.time()
        svm.fit(X_train, y_train)
        end_time = time.time()
        y_pred = svm.predict(X_validation)
        
        acc = accuracy_score(y_validation, y_pred)
        f1 = f1_score(y_validation, y_pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)
        all_y_true.extend(y_validation)
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
    trainval_idx = np.load('./data/mnist_trainval_idx.npy')
    test_idx = np.load('./data/mnist_test_idx.npy')
    y = np.load('./data/mnist_y.npy').astype(np.int32)

    # 维度选择
    print("=== 训练 SVM 模型, 使用不同的降维维度 ===")
    for dim in [10, 20, 30, 50]:
        print(f"\n=== 训练 SVM 模型，降维维度: {dim} ===")
        X_pca = np.load(f'./data/mnist_pca_dim{dim}.npy').astype(np.float32)
        scaler = StandardScaler()
        X_pca = scaler.fit_transform(X_pca)
        X_trainval, _ = X_pca[trainval_idx], X_pca[test_idx]
        y_trainval, _ = y[trainval_idx], y[test_idx]
        train(X_trainval, y_trainval)

    # 参数C选择
    print("\n=== 训练 SVM 模型, 使用不同的C参数 ===")
    X_pca = np.load(f'./data/mnist_pca_dim50.npy').astype(np.float32)
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca)
    X_trainval, _ = X_pca[trainval_idx], X_pca[test_idx]
    y_trainval, _ = y[trainval_idx], y[test_idx]
    for C in [0.1, 1.0, 10.0, 100.0]:
        print(f"\n=== 训练 SVM 模型, C参数: {C} ===")
        train(X_trainval, y_trainval, C=C)

    # 测试集测试
    print("\n=== 在测试集上评估 SVM 模型 ===")
    X_pca = np.load(f'./data/mnist_pca_dim50.npy').astype(np.float32)
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca)
    X_trainval, X_test = X_pca[trainval_idx], X_pca[test_idx]
    y_trainval, y_test = y[trainval_idx], y[test_idx]
    
    svm = SVC(kernel='rbf', C=10.0, gamma='scale')
    svm.fit(X_trainval, y_trainval)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"测试集准确率: {acc:.4f}")
    print(f"测试集F1分数（macro）: {f1:.4f}")
    print("测试集分类报告:\n", classification_report(y_test, y_pred))