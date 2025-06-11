import numpy as np
import cv2
from matplotlib import pyplot as plt
import rasterio
from rasterio.plot import show

def process_remote_sensing_image(image_path, output_path):
    """
    多光谱影像可视化处理工具
    功能：
    - 读取多波段遥感影像
    - 提取可见光波段
    - 执行自适应对比度增强
    - 输出标准RGB图像
    
    保持原有输入输出路径不变
    """
    # 读取影像数据
    with rasterio.open(image_path) as src:
        # 获取所有波段数据并调整维度顺序
        img_data = np.rollaxis(src.read(), 0, 3).astype('float32')
        
        # 显示基本信息
        print(f"影像尺寸: {src.height}×{src.width}")
        print(f"波段数量: {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        show(src, ax=axes[0], title='原始多光谱影像')
        
        # 提取RGB波段 (假设前三个波段)
        rgb_data = img_data[..., :3]
        
        # 显示原始数值范围
        print("\n原始波段范围:")
        for i, color in enumerate(['红', '绿', '蓝']):
            band = rgb_data[..., i]
            print(f"{color}波段: {band.min():.1f}-{band.max():.1f}")

        # 增强对比度 - 使用动态范围拉伸
        print("\n执行对比度增强:")
        low = np.percentile(rgb_data, 2, axis=(0,1))
        high = np.percentile(rgb_data, 98, axis=(0,1))
        
        for i, color in enumerate(['红', '绿', '蓝']):
            print(f"{color}波段拉伸范围: {low[i]:.1f}-{high[i]:.1f}")
        
        # 执行归一化处理
        norm_img = np.empty_like(rgb_data)
        for ch in range(3):
            diff = high[ch] - low[ch]
            if diff < 1e-6: diff = 1  # 避免除以零
            norm_img[..., ch] = np.clip(
                (rgb_data[..., ch] - low[ch]) / diff * 255, 
                0, 255
            )
        
        # 转换为8位整型
        uint8_img = norm_img.astype('uint8')
        
        # 保存结果 (保持BGR格式)
        cv2.imwrite(output_path, cv2.cvtColor(uint8_img, cv2.COLOR_RGB2BGR))
        
        # 显示处理结果
        axes[1].imshow(uint8_img)
        axes[1].set_title('增强后RGB影像')
        axes[1].axis('off')
        
        # 保存对比图
        plt.tight_layout()
        plt.savefig('processing_comparison.png', dpi=120)
        plt.show()
        
        print(f"\n处理完成! 结果保存至: {output_path}")

# 保持原有调用方式不变
if __name__ == "__main__":
    input_image = "C:\Users\Administrator\Desktop\day1\photo"  # 保持原路径不变
    output_image = "output_rgb.jpg"  # 保持原路径不变
    
    process_remote_sensing_image(input_image, output_image)