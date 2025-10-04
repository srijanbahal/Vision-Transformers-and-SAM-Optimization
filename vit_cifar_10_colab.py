### Colab Notebook: Vision Transformer (ViT) Fine-Tuning on CIFAR-10
# Goal: Highest accuracy possible with minimal compute (using pretrained ViT)

# --- SETUP ---
!pip install timm torchsummary

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import timm
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2

# --- CONFIG ---
class CFG:
    model_name = "vit_base_patch16_224"  # pretrained ViT
    img_size = 224
    batch_size = 64
    epochs = 50
    lr = 5e-5
    weight_decay = 0.05
    num_classes = 10
    smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()

# --- DATASET & AUGMENTATIONS ---
train_transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
valloader = DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

# --- MODEL ---
model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
model.to(cfg.device)

# --- LOSS & OPTIMIZER ---
criterion = LabelSmoothingCrossEntropy(smoothing=cfg.smoothing)
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

# EMA for stability
ema = ModelEmaV2(model, decay=0.999)

# --- TRAIN & EVAL ---
def train_one_epoch(model, loader, optimizer, criterion, device, ema):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        ema.update(model)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss/total, 100.*correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss/total, 100.*correct/total

# --- TRAINING LOOP ---
for epoch in range(cfg.epochs):
    train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, cfg.device, ema)
    val_loss, val_acc = evaluate(model, valloader, criterion, cfg.device)
    scheduler.step()
    print(f"Epoch {epoch+1}/{cfg.epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# Save best model
torch.save(model.state_dict(), "vit_cifar10.pth")
