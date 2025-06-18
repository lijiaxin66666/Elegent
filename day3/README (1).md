# DAY3实习课程笔记

### (下列内容为本日实习课程笔记)

---

# 1 常见激活函数详解

### 1.1 激活函数的作用
- 为神经网络引入非线性特性，使模型能够学习和表示复杂的非线性映射关系
- 决定神经元的激活状态，影响网络的学习能力和表达能力

### 1.2 主要激活函数介绍

#### Sigmoid激活函数
- **特点**：
  - 输出范围在(0,1)之间，适合用于概率输出
  - 函数平滑连续，便于求导
- **缺点**：
  - 存在严重的梯度消失问题（当|x|较大时梯度趋近于0）
  - 计算复杂度高（包含指数运算）
  - 输出非零中心化，可能导致梯度更新方向单一
- **适用场景**：二分类任务输出层、早期神经网络

#### Tanh激活函数
- **特点**：
  - 输出范围在(-1,1)之间，实现零中心化
  - 比Sigmoid具有更强的表达能力
- **缺点**：
  - 仍然存在梯度消失问题
  - 计算复杂度高
- **适用场景**：RNN等需要中心对称输出的任务

#### ReLU激活函数
- **特点**：
  - 计算效率极高，收敛速度快
  - 有效缓解梯度消失问题
- **缺点**：
  - 存在"死亡ReLU"问题（x<0时梯度为0，神经元永久失活）
- **适用场景**：CNN、深度神经网络的默认激活函数

#### Leaky ReLU激活函数
- **特点**：
  - 解决死亡ReLU问题，引入小斜率α（通常0.01）
  - 保持计算效率的同时避免神经元失活
- **适用场景**：深度CNN、回归任务

#### Parametric ReLU(PReLU)
- **特点**：
  - α为可学习参数，比Leaky ReLU更灵活
  - 适合大型数据集，增强网络表达能力
- **适用场景**：计算机视觉任务（如ImageNet训练）

#### ELU(Exponential Linear Unit)
- **特点**：
  - 负半轴更平滑，梯度表现更好
  - 输出均值接近0，加速收敛
- **缺点**：计算复杂度高于ReLU
- **适用场景**：深层网络，稳定收敛

#### Swish激活函数
- **特点**：
  - 结合ReLU和Sigmoid优点，自适应平滑激活
  - 非线性特性更强，表达能力优越
- **缺点**：计算复杂度高
- **适用场景**：EfficientNet等大规模网络

#### Softmax激活函数
- **特点**：
  - 将输出转换为概率分布，和为1
  - 专门用于多分类任务
- **适用场景**：多分类任务输出层（如图像分类、NLP）

### 1.3 激活函数对比表

| 激活函数 | 输出范围 | 计算复杂度 | 梯度消失 | 额外参数 | 典型应用场景 |
|----------|----------|------------|----------|----------|--------------|
| Sigmoid | (0,1) | 高 | 有 | 无 | 二分类输出层 |
| Tanh | (-1,1) | 高 | 有 | 无 | RNN、零均值数据 |
| ReLU | [0,∞) | 低 | 有（部分区域） | 无 | CNN、默认激活函数 |
| Leaky ReLU | (-∞,∞) | 低 | 无 | 有(α) | 防止神经元死亡 |
| PReLU | (-∞,∞) | 低 | 无 | 有(α) | 计算机视觉 |
| ELU | (-∞,∞) | 中等 | 无 | 有(α) | 深层网络稳定收敛 |
| Swish | (-∞,∞) | 高 | 无 | 有(β) | 高效神经网络 |
| Softmax | (0,1)且归一化 | 高 | 无 | 无 | 多分类任务 |

---

# 2 数据集处理与模型训练

## 2.1 数据集预处理流程

### 2.1.1 数据划分
##### 1) 划分方式
- 使用train_test_split按比例划分训练集和验证集，确保数据分布均匀。
```python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
```

##### 2) 路径处理
- 数据集路径需明确，训练集和验证集路径分别设置，便于后续操作。
```python
train_dir = r'/image2/train'
val_dir = r'/image2/val'
```

##### 3) 数据划分脚本 - deal_with_datasets.py
```python
import os
import shutil
from sklearn.model_selection import train_test_split
import random

# 设置随机种子保证结果可复现
random.seed(42)

# 原始数据集路径（需替换为实际路径）
dataset_dir = r'D:\Desktop\tcl\dataset\image2'
# 训练集和验证集输出路径
train_dir = r'D:\Desktop\tcl\dataset\image2\train'
val_dir = r'D:\Desktop\tcl\dataset\image2\val'

# 训练集划分比例
train_ratio = 0.7

# 创建输出目录（如果不存在）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别文件夹
for class_name in os.listdir(dataset_dir):
    # 跳过已存在的训练集和验证集目录
    if class_name in ["train", "val"]:
        continue
        
    class_path = os.path.join(dataset_dir, class_name)
    
    # 获取该类别下所有图片文件
    images = [f for f in os.listdir(class_path) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    # 确保图片路径包含类别信息
    images = [os.path.join(class_name, img) for img in images]

    # 划分训练集和验证集
    train_images, val_images = train_test_split(
        images, train_size=train_ratio, random_state=42)

    # 创建类别子文件夹
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # 移动训练集图片
    for img in train_images:
        src = os.path.join(dataset_dir, img)
        dst = os.path.join(train_dir, img)
        shutil.move(src, dst)

    # 移动验证集图片
    for img in val_images:
        src = os.path.join(dataset_dir, img)
        dst = os.path.join(val_dir, img)
        shutil.move(src, dst)
    
    # 删除原始类别文件夹（可选）
    shutil.rmtree(class_path)
```

## 2.2 生成数据索引文件
### 2.2.1 创建txt文件
- 自动生成训练集和验证集的txt文件，记录图片路径和标签。
```python
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")
```

- 指定数据集路径和输出txt文件名，方便后续加载数据。
```python
create_txt_file(r'/image2/train', 'train.txt')
create_txt_file(r'/image2/val', "val.txt")
```

### 2.2.2 生成文件脚本 - prepare.py
```python
import os

# 创建数据集索引文件函数
def create_txt_file(root_dir, txt_filename):
    # 打开文件进行写入
    with open(txt_filename, 'w') as f:
        # 遍历每个类别文件夹
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # 遍历该类别下的所有图片
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    # 写入图片路径和对应标签（格式：路径 标签）
                    f.write(f"{img_path} {label}\n")

# 生成训练集索引文件
create_txt_file(r'D:\Desktop\tcl\dataset\image2\train', 'train.txt')
# 生成验证集索引文件
create_txt_file(r'D:\Desktop\tcl\dataset\image2\val', "val.txt")
```

