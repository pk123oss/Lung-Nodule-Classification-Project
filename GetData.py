import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
def load_and_visualize_paired_data(image_root, label_root, sample_id):
    """
    加载并可视化跨目录配对的PNG图像和NIfTI标签数据
    参数:
        image_root: 包含PNG子文件夹的根目录（all_slices_png）
        label_root: 包含NIfTI子文件夹的根目录（data）
        sample_id: 要可视化的样本ID（子文件夹名）
    """

    png_dir = os.path.join(image_root, sample_id)
    label_dir = os.path.join(label_root, sample_id)
    if not os.path.exists(png_dir):
        print(f"错误: 图像目录 {png_dir} 不存在")
        return
    if not os.path.exists(label_dir):
        print(f"错误: 标签目录 {label_dir} 不存在")
        return
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and f.startswith(sample_id)],
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
    nii_file = os.path.join(label_dir, 'PSIR_roi-label.nii')
    if not os.path.exists(nii_file):
        nii_file = os.path.join(label_dir, 'PSIR_roi.nii')
    label_data = nib.load(nii_file).get_fdata()
    

    fig, axes = plt.subplots(2, min(6, len(png_files)), figsize=(18, 6))
    fig.suptitle(f'Sample  {sample_id} Image_item and LabelMask', y=1.05, fontsize=16)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']  # 多个字体备选
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    for i, png_file in enumerate(png_files[:6]):  

        img_path = os.path.join(png_dir, png_file)
        img = np.array(Image.open(img_path))
        
        label_slice = label_data[:, :, i] if i < label_data.shape[2] else None

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Cut_item {i+1}\n{png_file}')
        axes[0, i].axis('off')

        if label_slice is not None:
            axes[1, i].imshow(label_slice, cmap='jet', alpha=0.5)
            axes[1, i].set_title(f'Cut_item_Mask {i+1}\n{os.path.basename(nii_file)}')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
            axes[1, i].text(0.5, 0.5, 'NoTarget', ha='center')
    
    plt.tight_layout()
    plt.show()

    print(f"\n样本 {sample_id} 信息:")
    print(f"PNG切片数量: {len(png_files)}")
    print(f"NIfTI标签形状: {label_data.shape}")
    print(f"唯一标签值: {np.unique(label_data)}")


base_dir = "/kaggle/input/test-data/test/dl/train set"
image_root = os.path.join(base_dir, "all_slices_png")  # 包含编号子文件夹的PNG图像
label_root = os.path.join(base_dir, "data")           # 包含编号子文件夹的NIfTI标签

image_samples = set(os.listdir(image_root))
label_samples = set(os.listdir(label_root))
valid_samples = sorted(list(image_samples & label_samples))

for sample_id in valid_samples[:3]:
    print(f"\n{'='*50}")
    print(f"处理样本: {sample_id}")
    load_and_visualize_paired_data(image_root, label_root, sample_id)