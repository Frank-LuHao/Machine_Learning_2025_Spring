import os
import time
import numpy as np
import matplotlib.pyplot as plt
from cuml.decomposition import PCA
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def dem_reduce(dim, visualize=False):
    pca_result_path = f'./data/mnist_pca_dim{dim}.npy'
    label_path = './data/mnist_y.npy'

    # cuML PCA降维
    pca = PCA(n_components=dim)
    start_time = time.time()
    X_pca = pca.fit_transform(X)
    end_time = time.time()
    np.save(pca_result_path, X_pca)
    del pca  
    print("降维后结果已保存。")
    print("降维后形状：", X_pca.shape)
    print(f"PCA降维耗时: {end_time - start_time:.2f} 秒")
    
    if visualize:
        X_pca = np.load(pca_result_path)
        y = np.load(label_path)
        
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
        plt.colorbar()
        plt.title("PCA Visualization of MNIST (Train + Test)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 合并训练集和测试集
    X = np.concatenate([
        mnist_train.data.reshape(len(mnist_train), -1).numpy(),
        mnist_test.data.reshape(len(mnist_test), -1).numpy()
    ], axis=0).astype(np.float32)
    y = np.concatenate([
        mnist_train.targets.numpy(),
        mnist_test.targets.numpy()
    ], axis=0)

    idx = np.arange(len(X))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y
    )
    np.save('./data/mnist_trainval_idx.npy', trainval_idx)
    np.save('./data/mnist_test_idx.npy', test_idx)
    np.save('./data/mnist_y.npy', y)  # 保存标签

    for dim in [10, 20, 30, 50]:
        print(f"\n=== PCA降维, 维度: {dim} ===")
        dem_reduce(dim, visualize=False)