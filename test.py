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

valid_set = DatasetFolder("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/val", loader=lambda x: Image.open(x), extensions="png", transform=val_tfm)
batch_size = 128
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = models.vgg16_bn(pretrained = False).to(device)
model1.classifier[6] = nn.Linear(4096, 50).to(device)
# model1 = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False, init_weights=True, aux_logits=False)
# ftrs = model1.fc.in_features
# model1.fc = nn.Linear(ftrs, 50)
# model1 = model1.to(device)

state = torch.load(f'/home/robot/Downloads/DLCV/hw1/model_best.ckpt')
model1.load_state_dict(state)
model1.eval()

criterion = nn.CrossEntropyLoss()

predictions = []
check_loss = []
check_accs = []
for batch in valid_loader:
    imgs, labels = batch

    with torch.no_grad():
        logits = model1(imgs.to(device))
    loss = criterion(logits, labels.to(device))

    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    check_loss.append(loss.item())
    check_accs.append(acc)

check_loss = sum(check_loss) / len(check_loss)
check_acc = sum(check_accs) / len(check_accs)
print(check_loss, check_acc)

image_dir = os.listdir("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/val_50")
image_dir = natsort.natsorted(image_dir,reverse=False)

with open("/home/robot/Downloads/DLCV/hw1/sample_submission.csv","w") as f:
    f.write("image_id,label\n")
    for i in range(len(predictions)):
        f.write(f"{image_dir[i]},{predictions[i]}\n") 
image_dir = os.listdir("/home/robot/Downloads/DLCV/hw1/hw1_data/p1_data/val_50")
# image_dir = natsort.natsorted(image_dir,reverse=False)
# with open("/home/robot/Downloads/DLCV/hw1/predict.csv","w") as f:
#     f.write("image_id,label,predict\n")
#     for i in range(len(predictions)):
#         f.write(f"{image_dir[i]},{image_dir[i].split('_')[0]},{predictions[i]}\n")
