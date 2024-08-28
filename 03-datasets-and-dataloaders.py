import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader

# root: 학습/테스트 데이터가 저장되는 경로
# train: 학습용 또는 테스트용 데이터셋 여부 지정
# download = True: root에 데이터가 없는 경우 인터넷에서 다운로드
# transform과 target_transform: 특징과 정답, 변형 지정

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

# 데이터셋 순회, 시각화
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize = (8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size = (1, )).item() # size (tuple) - a tuple defining the shape of the output tensor
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")
    
plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, names = ['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform
        if self.target_transform:
            label = self.target_transform
            
        return image, label
    
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

while True:
    try:
        train_features, train_labels = next(iter(train_dataloader))

        print(f"\nFeature batch shape: {train_features.size()}\n")
        print(f"Labels batch shape: {train_labels.size()}\n")

        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap = "gray")
        plt.show()

        print(f"\nLabel: {label}\n\n")

    except:
        break