import os
import time
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from cuml.decomposition import PCA  # 导入 cuML 的 PCA

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

pca_result_path = './data/mnist_pca_dim10.npy'
label_path = './data/mnist_labels_dim10.npy'

if os.path.exists(pca_result_path) and os.path.exists(label_path):
    X_pca = np.load(pca_result_path)
    y = np.load(label_path)
    print("已加载降维结果。")
else:
    # cuML PCA降维
    pca = PCA(n_components=10)
    start_time = time.time()
    X_pca = pca.fit_transform(X)
    end_time = time.time()
    print(f"PCA降维耗时: {end_time - start_time:.2f} 秒")
    np.save(pca_result_path, X_pca)
    np.save(label_path, y)
    print("降维后结果已保存。")
    del pca  

print("降维后形状：", X_pca.shape)
# 可视化前两维
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar()
plt.title("PCA Visualization of MNIST (Train + Test)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()