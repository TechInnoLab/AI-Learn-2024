import torch
import torchvision
from torch import nn
import torch_snippets
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import tifffile
import segmentation_models_pytorch as smp
import os
from torch.nn.parallel import DataParallel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import torch.multiprocessing as mp

# mp.set_start_method('spawn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# print(device)


def img_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


def read_images(path: str, is_train=True):
    filepath = os.path.join(path, 'Training' if is_train else 'Validation')
    train_images = os.listdir(filepath + ('/image' if is_train else '/images'))
    label_images = os.listdir(filepath + ('/label' if is_train else '/masks'))
    features = []
    labels = []
    for image in train_images:
        features.append(read_tiff(os.path.join(
            filepath, ('image' if is_train else 'images'), image), True))
    for image in label_images:
        labels.append(read_tiff(os.path.join(
            filepath, ('label' if is_train else 'masks'), image), False))
    return features, labels


def read_tiff(file_path, is_train):
    # 读取.tiff文件中的数据
    data = tifffile.imread(file_path)
    # 将数据转换为 torch 张量
    if is_train:
        data_tensor = torch.from_numpy(data.astype('float32'))[:, :, :3].permute(2, 0, 1)
    else:
        data_tensor = torch.from_numpy(data)
    return data_tensor


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, dir_path):
        # self.transform = torchvision.transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, self.labels = read_images(dir_path, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in features]
        # print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return torch.nn.functional.normalize(img, p=2, dim=0)

    # def filter(self, imgs):
    #     return [img for img in imgs if (
    #             img.shape[0] >= self.crop_size[0] and
    #             img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = img_crop(self.features[idx], self.labels[idx],
                                  *self.crop_size)
        return feature, label

    def __len__(self):
        return len(self.features)

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([im[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([mask[None] for mask in masks]).long().to(device)
        return ims, ce_masks


def load_data(batch_size, crop_size, dir_path):
    """加载语义分割数据集"""
    trainDataset = SegDataset(True, crop_size, dir_path)
    valDataset = SegDataset(False, crop_size, dir_path)
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        trainDataset, batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers, collate_fn=trainDataset.collate_fn)
    val_iter = torch.utils.data.DataLoader(
        valDataset, int(batch_size / 2),
        drop_last=True, num_workers=num_workers, collate_fn=valDataset.collate_fn)
    return train_iter, val_iter


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x


# MCC_loss = smp.losses.MCCLoss()
Dice_loss = smp.losses.DiceLoss('binary')


# def iou_loss(y_pred, y_label, smooth=1e-6):
#     intersection = torch.logical_and(y_pred, y_label).sum(dim=(1, 2))
#     union = torch.logical_or(y_pred, y_label).sum(dim=(1, 2))
#
#     iou = (intersection + smooth) / (union + smooth)
#     jaccard_loss = 1 - iou
#     print(jaccard_loss)
#     return jaccard_loss.mean()


def dice_coeff(pred, target, epsilon=1e-5):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice.item()
    # intersection = torch.logical_and(prediction[0] > threshold, target[0] > threshold).sum()
    # union = torch.logical_or(prediction[0] > threshold, target[0] > threshold).sum()
    # return (intersection.float() / union.float()).item()


def dice_loss(pred, target, epsilon=1e-5):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)

    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return 1.0 - dice


def UnetLoss(preds, targets):
    # if not preds.shape == targets.shape:
    #     print("deal with shape conflict############")
    #     targets = F.interpolate(targets, size=preds.size(), mode='bilinear')
    print(preds)
    print(targets)
    ce_loss = dice_loss(preds, targets)
    acc = dice_coeff(preds, targets)
    return ce_loss, acc


def train_batch(model, data, optimizer, criterion):
    model.train()
    images, true_masks = data
    predict_masks = model(images)
    optimizer.zero_grad()
    loss, acc = criterion(predict_masks, true_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    images, true_masks = data
    predict_masks = model(images)
    loss, acc = criterion(predict_masks, true_masks)
    return loss.item(), acc


# model = smp.Unet().to(device)
model = UNet().to(device)
# 将模型包装在DataParallel中
model = DataParallel(model)

# model.load_state_dict(torch.load('pro_vegetation.pth'))


def train_realize():
    criterion = UnetLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with open('pro_vegetation_loss_log.txt', 'r') as f:
        num = len(f.readlines())
    n_epochs = (20 - num) if num <= 20 else 0
    print(n_epochs)

    log = torch_snippets.Report(n_epochs)
    train_iter, val_iter = load_data(4, (224, 224), 'ATLANTIC_FOREST')
    for ex in range(n_epochs):
        N = len(train_iter)
        # print(f"train_iter size : {N}")
        for bx, data in enumerate(train_iter):
            train_loss, train_acc = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=train_loss, trn_acc=train_acc, end='\r')

        N = len(val_iter)
        for bx, data in enumerate(val_iter):
            val_loss, val_acc = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=val_loss, val_acc=val_acc, end='\r')

        result = log.report_avgs(ex + 1)
        print(result)
        train_loss = result['epoch_trn_loss']
        val_loss = result['epoch_val_loss']
        train_acc = result['epoch_trn_acc']
        val_acc = result['epoch_val_acc']
        with open('pro_vegetation_loss_log.txt', 'a') as f:
            f.write(f'{train_loss}\t{val_loss}\n')
        with open('pro_vegetation_acc_log.txt', 'a') as f:
            f.write(f'{train_acc}\t{val_acc}\n')
        torch.save(model.state_dict(), 'pro_vegetation.pth')

    log.plot_epochs(['trn_loss', 'val_loss'])


# def model_test():
#     img = read_tiff("")


if __name__ == "__main__":
    train_realize()
    print("You have finished training!\nCheers!")
    # model_test()
