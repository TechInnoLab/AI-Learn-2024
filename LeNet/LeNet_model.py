import torch.nn as nn
import torch.utils
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def data_tensor_get():
    def read_data(train=True):
        data = []
        if train:
            num = 180
            floder = "TrainImage"
        else:
            num = 27
            floder = "TestImage"

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像，单通道
            transforms.ToTensor(),  # 将图像转换为张量
            # 将张量数据归一化到 [-1, 1] 的范围
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        for i in range(num):
            image_path = f"ImageData/{floder}/{i}.jpg"
            image = transform(Image.open(image_path)).unsqueeze(0)
            if i == 0:
                data = image
            else:
                data = torch.cat((data, image), dim=0)
        return data
    train_data = read_data()
    test_data = read_data(False)
    return train_data, test_data


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label_num = self.label[index]
        return sample, label_num


net = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 54 * 54, 240), nn.ReLU(),
    nn.Linear(240, 84), nn.ReLU(),
    nn.Linear(84, 10))


def load_dataset(batch_size):
    train_data, test_data = data_tensor_get()
    with open("ImageData/train_label.txt") as f:
        train_label_data = f.readlines()
        train_label_num = [int(i)-1 for i in train_label_data]

    with open("ImageData/test_label.txt") as f:
        test_label_data = f.readlines()
        test_label_num = [int(i)-1 for i in test_label_data]

    train_dataset = MyDataset(train_data, train_label_num)
    test_dataset = MyDataset(test_data, test_label_num)
    return (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
            DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))


def train_acc_cal(y_hat, y):
    max_indices = torch.argmax(y_hat, dim=1)
    acc = torch.mean((max_indices == y).float()).item()
    return acc


def test_acc_cal(net, data_iter, device):
    acc_list = []
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        acc_list.append(train_acc_cal(y_hat, y))
    acc = (sum(acc_list)/len(acc_list))
    return acc


def train_model(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"epoch:{epoch}, training...")
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss = l.item()
                train_acc = train_acc_cal(y_hat, y)
                test_acc = test_acc_cal(net, test_iter, device)

            with open("train_loss.txt", 'a') as f:
                f.write(f"{train_loss}\n")
            with open("train_acc.txt", 'a') as f:
                f.write(f"{train_acc}\n")
            with open("test_acc.txt", 'a') as f:
                f.write(f"{test_acc}\n")
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("test_acc:", test_acc)
        # animator.add(epoch + 1, (None, None, test_acc))
    torch.save(net.state_dict(), 'LeNet.pth')
    print(f'FINAL: loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')


lr = 10e-3
device = 'cuda'
num_epochs = 10
batch_size = 40

if __name__ == "__main__":
    train_iter, test_iter = load_dataset(batch_size)
    train_model(net, train_iter, test_iter, num_epochs, lr, device)
