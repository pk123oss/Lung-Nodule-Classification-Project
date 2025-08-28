import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1定义与训练代码中相同的类和函数

class To3D64x64x64(object):
    """将3D图像调整为64x64x64"""
    def __call__(self, image):
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(image_tensor, size=(64, 64, 64), 
                              mode='trilinear', align_corners=False)
        return resized.squeeze().numpy()

class LungNoduleDataset(Dataset):
    """自定义数据集类，用于加载和预处理肺部结节数据"""
    def __init__(self, base_dir, dataset_type='derivation', transform=None):
        """初始化数据集"""
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.transform = transform
        
        # 确定数据路径
        if dataset_type == 'derivation':
            self.image_dir = os.path.join(base_dir, 'test/dl/derivation')
            self.radiomics_dir = os.path.join(base_dir, 'test/radiomics/derivation')
            self.label_file = os.path.join(self.radiomics_dir, 'deri set.csv')
            self.enhance_file = os.path.join(self.radiomics_dir, 'derivation enhance.csv')
        else:
            self.image_dir = os.path.join(base_dir, f'test/dl/{dataset_type}')
            self.radiomics_dir = os.path.join(base_dir, f'test/radiomics/{dataset_type}')
            self.label_file = os.path.join(self.radiomics_dir, f'valid {dataset_type.split()[-1]}.csv')
            self.enhance_file = os.path.join(self.radiomics_dir, 'vali enhance.csv')
        
        # 加载标签数据
        self.label_df = pd.read_csv(self.label_file, header=None, names=['id', 'label'])
        self.label_df['label'] = pd.to_numeric(self.label_df['label'], errors='coerce')
        self.label_df = self.label_df.dropna()
        
        # 获取所有样本ID
        self.sample_ids = []
        for folder in os.listdir(self.image_dir):
            if folder in self.label_df['id'].astype(str).values:
                self.sample_ids.append(folder)
        
        if len(self.sample_ids) == 0:
            raise ValueError("未找到匹配的样本，请检查数据路径和文件名匹配")
        
        print(f"测试集 '{dataset_type}' 样本数: {len(self.sample_ids)}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        img_folder = os.path.join(self.image_dir, sample_id)
        nii_files = [f for f in os.listdir(img_folder) if 'PSIR_roi' in f and f.endswith('.nii')]
        
        if not nii_files:
            raise FileNotFoundError(f"未找到PSIR_roi文件在 {img_folder}")
            
        nii_file = os.path.join(img_folder, nii_files[0])
        img_data = nib.load(nii_file).get_fdata()
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        image_3d = np.moveaxis(img_data, -1, 0)
        
        enhance_df = pd.read_csv(self.enhance_file)
        sample_features = enhance_df[enhance_df.iloc[:, 0].astype(str).str.strip() == str(sample_id).strip()]
        
        if len(sample_features) == 0:
            raise ValueError(f"未找到样本 {sample_id} 的放射组学特征")
        
        radiomics_features = sample_features.iloc[:, 1:].values.flatten().astype(np.float32)
        
        label_row = self.label_df[self.label_df['id'].astype(str).str.strip() == str(sample_id).strip()]
        if len(label_row) == 0:
            raise ValueError(f"未找到样本 {sample_id} 的标签")
        
        label = float(label_row['label'].values[0])
        
        if self.transform:
            image_3d = self.transform(image_3d)
        
        image_tensor = torch.FloatTensor(image_3d).unsqueeze(0)
        features_tensor = torch.FloatTensor(radiomics_features)
        label_tensor = torch.FloatTensor([label])
        
        return image_tensor, features_tensor, label_tensor, sample_id

class StableBatchNorm1d(nn.Module):
    """与训练代码中相同的StableBatchNorm1d实现"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(StableBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if input.dim() != 2:
            raise ValueError(f'expected 2D input (got {input.dim()}D input)')
            
        if self.training:
            mean = input.mean(dim=0)
            var = input.var(dim=0, unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
            
            if input.size(0) == 1:
                if self.track_running_stats:
                    norm = (input - self.running_mean[None, :]) / torch.sqrt(self.running_var[None, :] + self.eps)
                else:
                    norm = input
            else:
                norm = (input - mean[None, :]) / torch.sqrt(var[None, :] + self.eps)
        else:
            if self.track_running_stats:
                norm = (input - self.running_mean[None, :]) / torch.sqrt(self.running_var[None, :] + self.eps)
            else:
                norm = input
        
        if self.affine:
            norm = norm * self.weight[None, :] + self.bias[None, :]
            
        return norm

class StableBatchNorm3d(nn.Module):
    """与训练代码中相同的StableBatchNorm3d实现"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(StableBatchNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if input.dim() != 5:
            raise ValueError(f'expected 5D input (got {input.dim()}D input)')
            
        if self.training:
            mean = input.mean(dim=[0, 2, 3, 4])
            var = input.var(dim=[0, 2, 3, 4], unbiased=False)
            
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
            
            if input.size(0) == 1:
                if self.track_running_stats:
                    norm = (input - self.running_mean[None, :, None, None, None]) / \
                           torch.sqrt(self.running_var[None, :, None, None, None] + self.eps)
                else:
                    norm = input
            else:
                norm = (input - mean[None, :, None, None, None]) / \
                       torch.sqrt(var[None, :, None, None, None] + self.eps)
        else:
            if self.track_running_stats:
                norm = (input - self.running_mean[None, :, None, None, None]) / \
                       torch.sqrt(self.running_var[None, :, None, None, None] + self.eps)
            else:
                norm = input
        
        if self.affine:
            norm = norm * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
            
        return norm

class HybridModel(nn.Module):
    """与训练代码中相同的混合模型"""
    def __init__(self, num_radiomics_features):
        super(HybridModel, self).__init__()
        
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            StableBatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            StableBatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            StableBatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            StableBatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Flatten()
        )
        
        # 放射组学特征处理部分
        self.radiomics_fc = nn.Sequential(
            nn.Linear(num_radiomics_features, 256),
            StableBatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            StableBatchNorm1d(128),
            nn.ReLU()
        )
        
        # 联合分类器
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4*4 + 128, 256),
            StableBatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, radiomics):
        cnn_features = self.cnn(image)
        radio_features = self.radiomics_fc(radiomics)
        combined = torch.cat([cnn_features, radio_features], dim=1)
        output = self.classifier(combined)
        return output

# 2. 加载模型

def load_model(model_path, num_radiomics_features):
    """加载保存的模型"""
    model = HybridModel(num_radiomics_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 3. 预测函数

def predict_on_dataset(model, dataset, batch_size=8):
    """在指定数据集上进行预测"""
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    all_sample_ids = []
    all_labels = []
    all_probs = []
    all_preds = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with torch.no_grad():
        for images, radiomics, labels, sample_ids in data_loader:
            images = images.to(device)
            radiomics = radiomics.to(device)
            
            outputs = model(images, radiomics)
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_sample_ids.extend(sample_ids)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs)
            all_preds.extend(preds)
    
    results = pd.DataFrame({
        'sample_id': all_sample_ids,
        'true_label': all_labels,
        'pred_prob': all_probs,
        'pred_label': all_preds
    })
    
    return results

# 4. 主函数

def main():
    # 设置参数
    base_dir = "/kaggle/input/testdatapro"  # 修改为你的数据路径
    model_path = "/kaggle/working/best_model.pth"  # 修改为你的模型路径
    
    # 创建转换
    transform = transforms.Compose([
        To3D64x64x64()
    ])
    
    # 加载模型
    print("加载模型...")
    
    # 首先需要确定放射组学特征的数量
    # 我们可以通过加载一个样本来获取这个信息
    temp_dataset = LungNoduleDataset(base_dir, dataset_type='derivation', transform=transform)
    num_radiomics_features = temp_dataset[0][1].shape[0]
    print(f"放射组学特征数量: {num_radiomics_features}")
    
    model = load_model(model_path, num_radiomics_features)
    print("模型加载完成")
    
    # 在三个数据集上进行预测
    datasets = [  'vali set1']
    all_results = []
    
    for dataset_type in datasets:
        print(f"\n正在处理数据集: {dataset_type}")
        
        # 加载数据集
        dataset = LungNoduleDataset(base_dir, dataset_type=dataset_type, transform=transform)
        
        # 进行预测
        results = predict_on_dataset(model, dataset)
        results['dataset'] = dataset_type
        all_results.append(results)
        
        # 计算并显示性能指标
        accuracy = (results['true_label'] == results['pred_label']).mean()
        print(f"{dataset_type} 准确率: {accuracy:.4f}")
        print(f"{dataset_type} 预测结果示例:")
        print(results.head())
        
        # 保存预测结果
        output_file = f"predictions_{dataset_type.replace(' ', '_')}.csv"
        results.to_csv(output_file, index=False)
        print(f"预测结果已保存到 {output_file}")
    
    # 合并所有结果
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_csv("all_predictions.csv", index=False)
    print("\n所有预测结果已保存到 all_predictions.csv")
    
    # 显示总体性能
    overall_accuracy = (final_results['true_label'] == final_results['pred_label']).mean()
    print(f"\n总体准确率: {overall_accuracy:.4f}")
    print("\n预测完成")

if __name__ == "__main__":
    main()