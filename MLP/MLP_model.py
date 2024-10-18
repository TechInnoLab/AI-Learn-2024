from torch.utils.data import Dataset, DataLoader
import torch.utils
import torch.nn as nn
import pandas as pd
# import torch
import numpy as np

log_path_list = ['train_loss.txt', 'train_acc.txt', 'test_acc.txt']

# 设置打印选项，使 NumPy 打印全部内容
np.set_printoptions(threshold=np.inf)


def init_files(file_path_list):
    for file_path in file_path_list:
        with open(file_path, 'w') as f:
            f.write("")


def clean_data():
    # 读取Excel文件
    file_path = "data2.xlsx"
    data = pd.read_excel(file_path)

    data_values = data.values

    delete_index = []
    for i in range(len(data_values)):
        if data_values[i, -1] < 85 or data_values[i, -1] > 105:
            delete_index.append(i)
        data_values[i, 0] = int(data_values[i, 0][:2])
    index_list = np.ones(len(data_values), dtype=bool)
    index_list[delete_index] = False
    data_values = data_values[index_list]
    data_values = data_values[:, :-1]

    # 将数组转换为 DataFrame
    df = pd.DataFrame(data_values)

    # 将 DataFrame 写入 Excel 文件
    excel_file = "clean_data1.xlsx"
    df.to_excel(excel_file, index=False)

    print(data_values)


def get_labels():
    file_path1 = "data1.xlsx"
    data1 = pd.read_excel(file_path1)

    data1_values = data1.values

    file_path2 = "clean_data2.xlsx"
    data2 = pd.read_excel(file_path2)

    data2_values = data2.values

    labels = []
    for i in range(len(data2_values)):
        labels.append(int(data1_values[int(data2_values[i, 0])-1, 2]-1))
    return labels


def data_tensor_get():
    # 读取Excel文件
    file_path = "clean_data2.xlsx"
    data = pd.read_excel(file_path)

    # 提取数据并转换为torch.tensor
    data_values = data.values
    torch_data = torch.tensor(data_values, dtype=torch.float32)

    return torch_data


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


def load_dataset(batch_size, step):
    data = data_tensor_get()[:, 1:]
    labels = get_labels()
    begin = 0
    while begin < len(labels):
        if begin+step <= len(labels):
            train_dataset = MyDataset(torch.cat((data[0:begin, :], data[begin+step:, :]), dim=0),
                                      labels[0:begin]+labels[begin+step:])
            test_dataset = MyDataset(data[begin:begin+step, :], 
                                     labels[begin:begin+step])
            begin += step
            yield (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                   DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))
        else:
            train_dataset = MyDataset(data[0:begin, :], labels[0:begin])
            test_dataset = MyDataset(data[begin:, :], labels[begin:])
            begin += step
            yield (DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
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


def train_model(net, num_epochs, batch_size, step, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        data_iter = load_dataset(batch_size, step)
        print(f"epoch:{epoch}, training...")
        net.train()
        for train_iter, test_iter in data_iter:
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
    torch.save(net.state_dict(), 'MLP.pth')
    print(f'FINAL: loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')


def main():
    net = nn.Sequential(
        nn.Linear(14, 10), nn.Sigmoid(),
        nn.Linear(10, 5), nn.Sigmoid(),
        nn.Linear(5, 2))

    lr = 10e-1
    device = 'cuda'
    num_epochs = 8
    batch_size = 40
    step = 14

    train_model(net, num_epochs, batch_size, step, lr, device)


if __name__ == "__main__":
    # 模型训练时解除下面两行注释
    # init_files(log_path_list)
    # main()

    # 获取总样本标签并打印
    labels = get_labels()
    for i in labels:
        print(i)
