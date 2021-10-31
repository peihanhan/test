import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder

import glob
import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import natsort 

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# def save_checkpoint(checkpoint_path, model, optimizer):
#     state = {'state_dict': model.state_dict(),
#              'optimizer' : optimizer.state_dict()}
#     torch.save(state, checkpoint_path)
#     print('model saved to %s' % checkpoint_path)
    
# def load_checkpoint(checkpoint_path, model, optimizer):
#     state = torch.load(checkpoint_path)
#     model.load_state_dict(state['state_dict'])
#     optimizer.load_state_dict(state['optimizer'])
#     print('model loaded from %s' % checkpoint_path)

train_tfm = transforms.Compose([
    # transforms.Resize((299,299)),
    transforms.Resize((224,224)),
    # transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomResizedCrop(299,scale=(0.08, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.508, 0.481, 0.431], std=[0.263, 0.255, 0.274]),
])

val_tfm = transforms.Compose([
    # transforms.Resize((299,299)),
    transforms.Resize((224,224)),
    # transforms.CenterCrop(299),
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.503, 0.477, 0.424], std=[0.263, 0.254, 0.275]),
])

train_set = DatasetFolder("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/train", loader=lambda x: Image.open(x), extensions="png", transform=train_tfm)
valid_set = DatasetFolder("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/val", loader=lambda x: Image.open(x), extensions="png", transform=val_tfm)

batch_size = 64
# batch_size = 512
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16_bn(pretrained = True)
model.classifier[6] = nn.Linear(4096, 50)

# model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True, init_weights=True, aux_logits=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 50)
model = model.to(device)

epoch = 300
best_acc = 0.0
# learning_rate = 0.025
learning_rate = 0.00005

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
criterion = nn.CrossEntropyLoss()

for ep in range(epoch):
    print(f"Learning rate = {scheduler.get_last_lr()[0]:.5f}")
    model.train()

    train_loss = []
    train_accs = []

    for batch in valid_loader:
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {ep + 1:03d}/{epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    valid_loss = []
    valid_accs = []
    for batch in valid_loader:
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    scheduler.step()

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(),f'/home/robot/Downloads/DLCV/hw1/model_{ep+1:03d}.ckpt')
        torch.save(model.state_dict(),f'/home/robot/Downloads/DLCV/hw1/model_best.ckpt')
        # state = {'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}
        # torch.save(state, '/home/robot/Downloads/DLCV/hw1/model.pth')
        print(f'model saved to /home/robot/Downloads/DLCV/hw1/model_{ep+1:03d}.ckpt')
        print(f"[ Valid | {ep + 1:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, best acc = {best_acc:.5f}")
        print(f"saving model_{ep} with acc {best_acc:.4f}")
    else:
        print(f"[ Valid | {ep + 1:03d}/{epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}, best acc = {best_acc:.5f}")

same_seeds(0) 

# model1 = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False, init_weights=True, aux_logits=False)
# ftrs = model1.fc.in_features
# model1.fc = nn.Linear(ftrs, 50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = models.vgg16_bn(pretrained = False).to(device)
model1.classifier[6] = nn.Linear(4096, 50).to(device)

state = torch.load(f'/home/robot/Downloads/DLCV/hw1/model_best.ckpt')
model1.load_state_dict(state)
model1.eval()
# model.eval()
predictions = []
check_accs = []
for batch in valid_loader:
    imgs, labels = batch
    with torch.no_grad():
        logits = model1(imgs.to(device))
        # logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    check_accs.append(acc)

check_acc = sum(check_accs) / len(check_accs)
print(check_acc)

image_dir = os.listdir("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/val_50")
image_dir = natsort.natsorted(image_dir,reverse=False)
with open("/home/robot/Downloads/DLCV/hw1/predict.csv","w") as f:
    f.write("image_id,label,predict\n")
    for i in range(len(predictions)):
        f.write(f"{image_dir[i]},{image_dir[i].split('_')[0]},{predictions[i]}\n") 
