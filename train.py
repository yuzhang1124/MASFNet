import os
import argparse
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from MASFNet import MASFNet

parser = argparse.ArgumentParser("MASFNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=20,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=12, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def main(args):
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = SAM2UNet(args.hiera_path)
    model.to(device)
    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)

    # ** 选定需要追踪梯度流的参数 **
    selected_params = [
        "encoder.blocks.0.prompt_learn.0.weight",
        "encoder.blocks.0.prompt_learn.2.weight",
        "encoder.blocks.1.prompt_learn.0.weight",
    ]  # **手动指定想要记录梯度的参数**

    grad_flow = {name: [] for name in selected_params}  # 初始化存储
    grad_txt_path = os.path.join(args.save_path, "gradient_flow.txt")

    iteration = 0  # 记录全局迭代次数

    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss = structure_loss(pred0, target) + structure_loss(pred1, target) + structure_loss(pred2, target)
            loss.backward()

            # **每5步记录一次梯度**
            if i % 5 == 0:  # 每5步记录一次
                for name in selected_params:
                    if name in dict(model.named_parameters()):
                        param = dict(model.named_parameters())[name]
                        if param.grad is not None:
                            grad_mean = param.grad.abs().mean().item()
                            grad_flow[name].append(f"{iteration}, {grad_mean}")

            iteration += 1  # 迭代计数
            optim.step()

            if i % 50 == 0:
                print(f"epoch:{epoch + 1}-{i + 1}: loss:{loss.item()}")

        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'MASFNet-{epoch + 1}.pth'))
            print('[Saving Snapshot:]', os.path.join(args.save_path, f'MASFNet-{epoch + 1}.pth'))

    # **写入TXT文件**
    with open(grad_txt_path, "w") as f:
        for name, values in grad_flow.items():
            f.write(f"Parameter: {name}\n")
            f.write("\n".join(values))
            f.write("\n\n")

    # **绘制梯度变化图**
    plt.figure(figsize=(12, 6))
    for name, grad_list in grad_flow.items():
        # 提取数值
        grad_values = [float(line.split(", ")[1]) for line in grad_list]
        plt.plot(grad_values, label=name)

    plt.xlabel("Iterations")
    plt.ylabel("Mean Absolute Gradient")
    plt.title("Gradient Flow Over Time")
    plt.legend()
    plt.savefig(os.path.join(args.save_path, "gradient_flow.png"))
    plt.show()



if __name__ == "__main__":
    main(args)
