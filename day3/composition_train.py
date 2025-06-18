import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, googlenet, mobilenet_v2
from tqdm import tqdm


# 模型选择函数
def get_model(name):
    if name == "resnet18":
        return resnet18(pretrained=False, num_classes=10)
    elif name == "googlenet":
        return googlenet(pretrained=False, num_classes=10, aux_logits=False)
    elif name == "mobilenet_v2":
        model = mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
        return model
    else:
        raise ValueError("Unsupported model name")


# 选择模型
while True:
    model_name = input("请输入模型名称(resnet18, googlenet, mobilenet_v2): ")
    if model_name in ["resnet18", "googlenet", "mobilenet_v2"]:
        break

model = get_model(model_name)
if torch.cuda.is_available():
    model = model.cuda()

# 设置 AMP 自动混合精度
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if use_amp else None

# 准备数据集（添加数据增强）
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_data = torchvision.datasets.CIFAR10(
    root="../dataset_chen",
    train=True,
    transform=transform,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="../dataset_chen",
    train=False,
    transform=test_transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=128, num_workers=4)

# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)

# 训练与测试循环
writer = SummaryWriter("../../../../logs_train")
total_train_step = 0
total_test_step = 0
epoch = 50
best_accuracy = 0.0

for epoch_idx in range(epoch):
    print(f"-----第{epoch_idx + 1}轮训练开始-----")
    model.train()

    # 训练进度条
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}")
    for imgs, targets in train_bar:
        if torch.cuda.is_available():
            imgs, targets = imgs.cuda(), targets.cuda()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

        optim.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        total_train_step += 1
        train_bar.set_postfix(loss=loss.item())

        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    model.eval()
    total_test_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing")
        for imgs, targets in test_bar:
            if torch.cuda.is_available():
                imgs, targets = imgs.cuda(), targets.cuda()

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()

    accuracy = total_correct / len(test_data)
    avg_loss = total_test_loss / len(test_loader)

    print(f"测试集 Loss: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
    writer.add_scalar("test_loss", avg_loss, epoch_idx)
    writer.add_scalar("test_accuracy", accuracy, epoch_idx)

    # 保存最优模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), f"model_save/{model_name}_best.pth")
        print(f"保存最优模型，准确率: {best_accuracy:.4f}")

    scheduler.step()

writer.close()

'''
Resnet18:
整体测试集上的loss: 195.47172996265808
整体测试集上的正确率: 0.6577

Googenet:
整体测试集上的loss: 175.47176936268806
整体测试集上的正确率: 0.7781

mobilenet:
整体测试集上的loss: 223.33217066526413
整体测试集上的正确率: 0.5452
'''