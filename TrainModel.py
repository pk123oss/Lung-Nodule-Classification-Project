# 最终版的
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
from tqdm import tqdm
from torchsummary import summary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


torch.manual_seed(42)
np.random.seed(42)


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



class LungNoduleDataset(Dataset):
    """加载和预处理肺部结节数据"""
    def __init__(self, base_dir, dataset_type='derivation', transform=None, augment=False, debug=False):
        """初始化数据集"""
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.augment = augment
        self.debug = debug
        
        if dataset_type == 'derivation':
            self.image_dir = os.path.join(base_dir, 'test/dl/derivation')
            self.radiomics_dir = os.path.join(base_dir, 'test/radiomics/derivation')
            self.label_file = os.path.join(self.radiomics_dir, 'deri set.csv')
            self.enhance_file = os.path.join(self.radiomics_dir, 'derivation enhance.csv')
        else:
            self.image_dir = os.path.join(base_dir, f'test/dl/{dataset_type}')
            self.radiomics_dir = os.path.join(base_dir, f'test/radiomics/{dataset_type}')
            self.label_file = os.path.join(self.radiomics_dir, f'vali {dataset_type.split()[-1]}.csv')
            self.enhance_file = os.path.join(self.radiomics_dir, 'vali enhance.csv')
        
        self.label_df = pd.read_csv(self.label_file, header=None, names=['id', 'label'])
        self.label_df['label'] = pd.to_numeric(self.label_df['label'], errors='coerce')
        self.label_df = self.label_df.dropna()
        
        # 获取样本ID
        self.sample_ids = []
        for folder in os.listdir(self.image_dir):
            if folder in self.label_df['id'].astype(str).values:
                self.sample_ids.append(folder)
        
        if len(self.sample_ids) == 0:
            raise ValueError("未找到匹配的样本，请检查数据路径和文件名匹配")
        
        print(f"初始样本数: {len(self.sample_ids)}")
        
        # 类别平衡
        self._balance_classes()
        
        if self.augment:
            self._augment_dataset(scale_factor=2)
        
        if self.debug:
            self._visualize_samples()
    
    def _balance_classes(self):
        class_0 = [id_ for id_ in self.sample_ids 
                  if self.label_df[self.label_df['id'].astype(str) == str(id_)]['label'].values[0] == 0]
        class_1 = [id_ for id_ in self.sample_ids 
                  if self.label_df[self.label_df['id'].astype(str) == str(id_)]['label'].values[0] == 1]
        
        print(f"原始类别分布 - 0类: {len(class_0)}, 1类: {len(class_1)}")
        
        if len(class_0) > len(class_1):
            class_1 = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        else:
            class_0 = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
        
        self.sample_ids = class_0 + class_1
        np.random.shuffle(self.sample_ids)
        
        print(f"类别平衡后样本数: {len(self.sample_ids)} (0类: {len(class_0)}, 1类: {len(class_1)})")
    
    def _augment_dataset(self, scale_factor=1):

        original_ids = self.sample_ids.copy()
        augmented_ids = []
        
        for _ in range(scale_factor - 1):
            augmented_ids.extend(original_ids)
        
        self.sample_ids.extend(augmented_ids)
        np.random.shuffle(self.sample_ids)
        print(f"数据增强后样本总数: {len(self.sample_ids)}")
    
    def _visualize_samples(self, num_samples=3):
        plt.figure(figsize=(15, 5*num_samples))
        for i in range(min(num_samples, len(self.sample_ids))):
            sample_id = self.sample_ids[i]
            label = self.label_df[self.label_df['id'].astype(str) == str(sample_id)]['label'].values[0]
            img_folder = os.path.join(self.image_dir, sample_id)
            nii_files = [f for f in os.listdir(img_folder) if 'PSIR_roi' in f and f.endswith('.nii')]
            
            if not nii_files:
                print(f"警告: 未找到PSIR_roi文件在 {img_folder}")
                continue
                
            nii_file = os.path.join(img_folder, nii_files[0])
            img_data = nib.load(nii_file).get_fdata()
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            
            mid_slice = img_data.shape[2] // 2
            
            plt.subplot(num_samples, 2, 2*i+1)
            plt.imshow(img_data[:, :, mid_slice], cmap='gray')
            plt.title(f"DataSample {sample_id} (label: {label})\n origin Image-cut{mid_slice}")
            
            if self.transform:
                try:
                    transformed = self.transform(img_data)
                    plt.subplot(num_samples, 2, 2*i+2)
                    plt.imshow(transformed[32, :, :], cmap='gray')
                    plt.title(f"DataSample {sample_id} (label: {label})\n after transorm-cut32")
                except Exception as e:
                    print(f"转换图像时出错: {e}")
        
        plt.tight_layout()
        plt.show()
    
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
        
        return image_tensor, features_tensor, label_tensor

class To3D64x64x64(object):

    def __init__(self, augment=False):
        self.augment = augment
    
    def __call__(self, image):
        image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        if self.augment:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            image_tensor = image_tensor * brightness
            image_tensor = (image_tensor - image_tensor.mean()) * contrast + image_tensor.mean()
            
            if np.random.rand() > 0.5:
                noise = torch.randn_like(image_tensor) * 0.05
                image_tensor = image_tensor + noise
        
        resized = F.interpolate(image_tensor, size=(64, 64, 64), 
                              mode='trilinear', align_corners=False)
        
        return resized.squeeze().numpy()

## 模型架构

class StableBatchNorm1d(nn.Module):

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
            
            # 对小batch的特殊处理 直接丢弃
            if input.size(0) == 1:
                
                if self.track_running_stats:
                    norm = (input - self.running_mean[None, :]) / torch.sqrt(self.running_var[None, :] + self.eps)
                else:
                    # 如果没有running stats，直接返回原始值
                    norm = input
            else:
                # 正常batch标准化
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
            
            # 对小batch的特殊处理
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
    #混合模型
    def __init__(self, num_radiomics_features):
        super(HybridModel, self).__init__()
        
        # CNN部分 - 使用StableBatchNorm3d
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
        
        # 放射组学特征处理部分 - 使用StableBatchNorm1d
        self.radiomics_fc = nn.Sequential(
            nn.Linear(num_radiomics_features, 256),
            StableBatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            StableBatchNorm1d(128),
            nn.ReLU()
        )
        
        # 联合分类器 - 使用StableBatchNorm1d
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
    
    def visualize(self):
        self.eval()
        with torch.no_grad():
            image_input = torch.randn(1, 1, 64, 64, 64)
            radio_input = torch.randn(1, num_radiomics_features)
            
            print("\n模型结构摘要:")
            print("图像输入形状:", image_input.shape)
            print("放射组学特征输入形状:", radio_input.shape)
            
            print("\nCNN部分:")
            x = image_input
            for i, layer in enumerate(self.cnn):
                x = layer(x)
                print(f"层 {i+1}: {layer.__class__.__name__}, 输出形状: {x.shape}")
            
            print("\n放射组学特征处理部分:")
            x = radio_input
            for i, layer in enumerate(self.radiomics_fc):
                x = layer(x)
                print(f"层 {i+1}: {layer.__class__.__name__}, 输出形状: {x.shape}")
            
            print("\n分类器部分:")
            cnn_features = torch.randn(1, 128*4*4*4)
            radio_features = torch.randn(1, 128)
            combined = torch.cat([cnn_features, radio_features], dim=1)
            x = combined
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                print(f"层 {i+1}: {layer.__class__.__name__}, 输出形状: {x.shape}")
        
        self.train()



base_dir = "/kaggle/input/testdatapro"
dataset_type = 'derivation'

transform = transforms.Compose([
    To3D64x64x64(augment=True)
])

full_dataset = LungNoduleDataset(
    base_dir, 
    dataset_type=dataset_type, 
    transform=transform, 
    augment=True,
    debug=True
)

print(f"数据集大小: {len(full_dataset)}")
sample = full_dataset[0]
print(f"样本图像形状: {sample[0].shape}")
print(f"放射组学特征数量: {sample[1].shape[0]}")
print(f"标签值: {sample[2].item()}")

num_radiomics_features = full_dataset[0][1].shape[0]
print(f"\n放射组学特征数量: {num_radiomics_features}")

train_idx, val_idx = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    random_state=42,
    stratify=[full_dataset[i][2].item() for i in range(len(full_dataset))]
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

batch_size = 8
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = HybridModel(num_radiomics_features)
model.visualize()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5):

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    best_val_auc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        processed_samples = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, radiomics, labels) in enumerate(train_bar):
            if batch_idx == len(train_loader) - 1 and images.size(0) == 1:
                print(f"\n跳过训练集的最后一个小批量（batch_size=1）")
                continue
                
            images = images.to(device)
            radiomics = radiomics.to(device)
            labels = labels.to(device)
            
            outputs = model(images, radiomics)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).float()
            correct = (preds == labels).sum().item()
            
            running_loss += loss.item() * images.size(0)
            running_corrects += correct
            processed_samples += images.size(0)

            current_loss = running_loss / processed_samples if processed_samples > 0 else 0
            current_acc = running_corrects / processed_samples if processed_samples > 0 else 0
            train_bar.set_postfix({
                'loss': current_loss,
                'acc': current_acc
            })

        epoch_train_loss = running_loss / processed_samples if processed_samples > 0 else float('nan')
        epoch_train_acc = running_corrects / processed_samples if processed_samples > 0 else float('nan')
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_labels = []
        all_probs = []
        all_preds = []
        val_processed_samples = 0
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch_idx, (images, radiomics, labels) in enumerate(val_bar):
                # 跳过最后一个小批量（如果batch_size=1）
                if batch_idx == len(val_loader) - 1 and images.size(0) == 1:
                    print(f"\n跳过验证集的最后一个小批量（batch_size=1）")
                    continue
                    
                images = images.to(device)
                radiomics = radiomics.to(device)
                labels = labels.to(device)
                
                outputs = model(images, radiomics)
                loss = criterion(outputs, labels)
                
                preds = (outputs > 0.5).float()
                correct = (preds == labels).sum().item()
                
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                val_corrects += correct
                val_processed_samples += batch_size
                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())

                current_val_loss = val_loss / val_processed_samples if val_processed_samples > 0 else 0
                current_val_acc = val_corrects / val_processed_samples if val_processed_samples > 0 else 0
                val_bar.set_postfix({
                    'loss': current_val_loss,
                    'acc': current_val_acc
                })

        if val_processed_samples > 0:
            epoch_val_loss = val_loss / val_processed_samples
            epoch_val_acc = val_corrects / val_processed_samples

            all_labels = np.array(all_labels).astype(int)
            all_preds = np.array(all_preds).astype(int)

            if len(np.unique(all_labels)) > 1:
                epoch_val_auc = roc_auc_score(all_labels, all_probs)
                epoch_val_f1 = f1_score(all_labels, all_preds)
                epoch_val_precision = precision_score(all_labels, all_preds)
                epoch_val_recall = recall_score(all_labels, all_preds)
            else:
                epoch_val_auc = float('nan')
                epoch_val_f1 = float('nan')
                epoch_val_precision = float('nan')
                epoch_val_recall = float('nan')
        else:
            epoch_val_loss = float('nan')
            epoch_val_acc = float('nan')
            epoch_val_auc = float('nan')
            epoch_val_f1 = float('nan')
            epoch_val_precision = float('nan')
            epoch_val_recall = float('nan')
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_auc'].append(epoch_val_auc)
        history['val_f1'].append(epoch_val_f1)
        history['val_precision'].append(epoch_val_precision)
        history['val_recall'].append(epoch_val_recall)

        if not torch.isnan(torch.tensor(epoch_val_loss)):
            scheduler.step(epoch_val_loss)

        if not torch.isnan(torch.tensor(epoch_val_auc)) and epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_model_wts = model.state_dict().copy()

            torch.save(model.state_dict(), 'best_model.pth')

            torch.save({
                'cnn': model.cnn.state_dict(),
                'radiomics_fc': model.radiomics_fc.state_dict(),
                'classifier': model.classifier.state_dict()
            }, 'best_model_parts.pth')

        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'训练样本: {processed_samples}/{len(train_loader.dataset)}')
        print(f'验证样本: {val_processed_samples}/{len(val_loader.dataset)}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | AUC: {epoch_val_auc:.4f}')
        print(f'Val F1: {epoch_val_f1:.4f} | Precision: {epoch_val_precision:.4f} | Recall: {epoch_val_recall:.4f}')
        print('-' * 50)

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    else:
        print("警告: 没有保存的最佳模型权重")

    torch.save(model.state_dict(), 'final_model.pth')
    torch.save({
        'cnn': model.cnn.state_dict(),
        'radiomics_fc': model.radiomics_fc.state_dict(),
        'classifier': model.classifier.state_dict()
    }, 'final_model_parts.pth')
    
    return model, history


print("\n开始训练...")
trained_model, history = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=5
    
)
print("训练完成，模型已保存")


plt.figure(figsize=(18, 12))
plt.subplot(2, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(2, 3, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(2, 3, 3)
plt.plot(history['val_auc'], label='Val AUC', color='green')
plt.title('Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()


plt.subplot(2, 3, 4)
plt.plot(history['val_f1'], label='Val F1', color='red')
plt.title('Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# Precision曲线
plt.subplot(2, 3, 5)
plt.plot(history['val_precision'], label='Val Precision', color='purple')
plt.title('Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()


plt.subplot(2, 3, 6)
plt.plot(history['val_recall'], label='Val Recall', color='orange')
plt.title('Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()



def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []
    
    eval_bar = tqdm(data_loader, desc='Evaluating')
    with torch.no_grad():
        for images, radiomics, labels in eval_bar:
            images = images.to(device)
            radiomics = radiomics.to(device)
            labels = labels.to(device)
            
            outputs = model(images, radiomics)
            probs = outputs.cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs)
            all_preds.extend(preds)
    
    # 确保标签是整数类型
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)
    
    metrics = {
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float('nan'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'report': classification_report(all_labels, all_preds, target_names=['非侵袭性', '侵袭性']),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return metrics

print("\n评估验证集...")
val_results = evaluate_model(trained_model, val_loader)

print("\n验证集性能:")
print(f"AUC: {val_results['auc']:.4f}")
print(f"Accuracy: {val_results['accuracy']:.4f}")
print(f"F1 Score: {val_results['f1']:.4f}")
print(f"Precision: {val_results['precision']:.4f}")
print(f"Recall: {val_results['recall']:.4f}")
print("\n分类报告:")
print(val_results['report'])
print("\n混淆矩阵:")
print(val_results['confusion_matrix'])

plt.figure(figsize=(8, 6))
plt.imshow(val_results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵\nAUC: {:.4f}, Accuracy: {:.4f}'.format(
    val_results['auc'], val_results['accuracy']))
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['非侵袭性', '侵袭性'])
plt.yticks(tick_marks, ['非侵袭性', '侵袭性'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')

thresh = val_results['confusion_matrix'].max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, val_results['confusion_matrix'][i, j],
                horizontalalignment="center",
                color="white" if val_results['confusion_matrix'][i, j] > thresh else "black")

plt.tight_layout()
plt.show()