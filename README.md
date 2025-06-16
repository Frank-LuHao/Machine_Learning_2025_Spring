# MNIST 手写数字识别实验(2025_Spring_SCU 机器学习引论)

## 项目简介

本项目基于 MNIST 手写数字数据集，完成了数据降维（PCA）、分类（KNN、SVM）等机器学习实验。项目包含数据预处理、降维、模型训练与评估等完整流程，并对不同参数和方法进行了对比分析。

## 文件结构

```
Machine Learning/
│
├── data/                        # 数据与降维结果
│   ├── mnist_pca_dim10.npy
│   ├── mnist_pca_dim20.npy
│   ├── mnist_pca_dim30.npy
│   ├── mnist_pca_dim50.npy
│   ├── mnist_test_idx.npy
│   ├── mnist_trainval_idx.npy
│   ├── mnist_y.npy
│   └── MNIST/raw/               # 原始MNIST数据
│
├── result/                      # 实验结果与可视化
│   ├── out_info/
│   │   ├── KNN.txt
│   │   ├── PCA.txt
│   │   └── SVM.txt
│   └── picture/
│       └── PCA.png
│
├── requirement/                 # 项目说明与实验要求
│   ├── 说明.ipynb
│   └── 说明.txt
│
├── KNN.py                       # KNN分类实验代码
├── PCA.py                       # PCA降维及数据预处理代码
├── SVM.py                       # SVM分类实验代码
├── analysis.ipynb               # 其他降维与分类实验分析
└── README.md                    # 项目说明文件
```

## 环境配置

建议使用 Anaconda 或 Miniconda 管理环境，推荐 Python 3.8 及以上版本。

### 主要依赖包

- numpy
- matplotlib
- scikit-learn
- torch
- torchvision
- cuml（需NVIDIA GPU，支持GPU加速的PCA、KNN、SVM等）

## 快速开始

1. 运行 `PCA.py` 进行数据预处理和降维，生成降维后的数据文件。
2. 运行 `KNN.py` 或 `SVM.py` 进行模型训练与评估。
3. 查看 `result/` 目录下的实验结果和可视化图片。
4. 可参考 `analysis.ipynb` 进行更多方法的实验与分析。

## 说明

- 数据集自动下载并保存于 `data/` 目录。
- 代码默认使用 GPU 加速（需支持的NVIDIA显卡和驱动）。
- 详细实验要求与说明见 `requirement/说明.txt` 和 `requirement/说明.ipynb`。