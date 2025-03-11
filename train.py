import argparse
import config.config as config
import torch
from network.CNN import CNNnet
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os

def train_0(args_cli, model, configure):
    logdir = configure["log_dir"]
    train_dataset = configure["data_dir"]
    test_dataset= configure["test_dir"]

    device = args_cli.device

        # 数据预处理（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize(256),                 # 调整大小使短边为256
        transforms.RandomResizedCrop(224),      # 随机裁剪缩放
        transforms.RandomHorizontalFlip(),      # 水平翻转
        transforms.RandomRotation(15),           # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计值
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),             # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
        # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=train_dataset,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=test_dataset,
        transform=val_transform
    )

        # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=configure['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=configure['lr'])
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True
    )
    writer = SummaryWriter(logdir)
    best_val_acc = 0.0


    # 训练循环
    for epoch in range(configure["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{configure["num_epochs"]}')
        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            train_progress.set_postfix({
                'loss': running_loss/(total//labels.size(0)),
                'acc': 100*correct/total
            })

        # 记录训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_loss = val_loss / len(test_loader)
                val_acc = 100 * val_correct / val_total
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
        scheduler.step(val_acc)  # 根据验证指标调整学习率
        
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(configure["save_dir"], 'best_model.pth'))

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    writer.close()
    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")

def train_1(args_cli, model, configure):

    best_val_acc = 0.0  

    device = args_cli.device

    # CIFAR-10标准化参数
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2470, 0.2435, 0.2616]
    
    # 数据增强策略
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=configure['data_dir'],
        train=True,
        download=False,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=configure['data_dir'],
        train=False,
        download=False,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=configure['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=configure['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # 优化器和损失函数
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=configure['lr'],
        momentum=0.9,
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=configure['num_epochs']
    )
    
    criterion = CrossEntropyLoss()
    writer = SummaryWriter(configure["log_dir"])
    
    # 训练循环
    for epoch in range(configure["num_epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{configure["num_epochs"]}')
        for inputs, targets in train_progress:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播 + 反向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            train_progress.set_postfix({
                'loss': running_loss/(total//targets.size(0)),
                'acc': 100.*correct/total
            })
        
        # 学习率调整
        scheduler.step()
        
        # 记录训练指标
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        # 验证阶段
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, os.path.join(configure["save_dir"], 'best_model.pth'))
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # 最终测试
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    writer.close()
    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return loss / len(loader), 100. * correct / total
