#
这是一个基于#深度学习#的肺部结节侵袭性分类系统，结合3D医学影像与放射组学特征进行多模态分析。项目提供完整的数据可视化、模型训练和评估流程，专门针对医学影像数据的特点进行了优化。
#
A deep learning-based lung nodule invasiveness classification system that combines 3D medical imaging with radiomics features for multimodal analysis. The project provides a complete pipeline for data visualization, model training, and evaluation, specifically optimized for medical imaging data characteristics.


Epoch 25/25 Summary:
Train Loss: 0.0159 | Acc: 0.9963
Val Loss: 0.0087 | Acc: 0.9963 | AUC: 1.0000
Val F1: 0.9963 | Precision: 0.9926 | Recall: 1.0000
--------------------------------------------------

<img width="1131" height="738" alt="image" src="https://github.com/user-attachments/assets/766e3bb0-18d8-4cf2-b00c-af174abb744a" /># Lung-Nodule-Classification-Project

lung-nodule-classification/
📊 GetData.py              # 数据加载和可视化工具
🤖 TrainModel.py           # 模型训练和评估主程序
📁 utils/                  # 工具函数目录
📁 config/                 # 配置文件目录
📁 data/                   # 数据目录（需要自行添加）
📁 models/                 # 模型保存目录（训练后生成）
📁 results/                # 训练结果目录（训练后生成）
📄 requirements.txt        # 依赖包列表
📄 README.md              # 项目说明文档

# 创建conda环境
conda create -n lung-nodule python=3.8
conda activate lung-nodule

# 安装依赖
pip install -r requirements.txt

torch==1.13.0
torchvision==0.14.0
nibabel==4.0.2
numpy==1.21.6
pandas==1.3.5
matplotlib==3.5.3
scikit-learn==1.0.2
tqdm==4.64.1







# ⚡ 关键技术特点
## 🛡️ 稳定性处理
- StableBatchNorm3d: 处理3D卷积的小批量情况
- StableBatchNorm1d: 处理全连接层的小批量情况
- 自动跳过batch_size=1的训练批次

## 🔧 数据增强策略
- 随机亮度调整(0.8-1.2倍)
- 随机对比度调整(0.8-1.2倍)  
- 高斯噪声添加(概率0.5, 强度0.05)

## 📈 训练监控
- 实时进度条显示(tqdm)
- 多指标跟踪(损失/准确率/AUC/F1等)
- 学习率动态调整(ReduceLROnPlateau)
- 最佳模型保存机制

## 🎨 可视化功能
- 训练曲线可视化(6个子图)
- 混淆矩阵热力图
- 数据样本预览
- 中文标签支持
