import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timm.models.vision_transformer import vit_base_patch16_224
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pytorch_msssim import ssim as ssim_loss_fn
from torchvision.models import MobileNet_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionDataset(Dataset):
    def __init__(self, focus1_list, focus2_list, target_list, transform=None):
        self.focus1_list = focus1_list
        self.focus2_list = focus2_list
        self.target_list = target_list
        self.transform = transform

    def __len__(self):
        return len(self.focus1_list)

    def __getitem__(self, idx):
        img1 = Image.open(self.focus1_list[idx]).convert("RGB")
        img2 = Image.open(self.focus2_list[idx]).convert("RGB")
        target = Image.open(self.target_list[idx]).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)

        target = transforms.Resize((224, 224))(target)
        target = transforms.ToTensor()(target)
        target = target/255.0
        x = torch.cat([img1, img2], dim=0)

        return x, target

class LocalExtractor(nn.Module):
    def __init__(self):
        super(LocalExtractor, self).__init__()
        mobilenet_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.preconv = nn.Conv2d(6, 3, kernel_size=1)
        self.features = mobilenet_model.features[:14]

        for name, param in self.features.named_parameters():
            if "4" in name or "5" in name:
                param.requires_grad = True

    def forward(self, x):
        x = self.preconv(x)
        return self.features(x)


class AViT(nn.Module):
    def __init__(self):
        super(AViT, self).__init__()
        self.preconv = nn.Conv2d(6, 3, kernel_size=1)
        self.vit = vit_base_patch16_224(pretrained=True)
        self.norm = nn.LayerNorm(768)
        self.up = nn.Conv2d(768, 96, kernel_size=1)

    def forward(self, x):
        x = self.preconv(x)
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.norm(x[:, 1:]).mean(1)
        x = self.up(x.unsqueeze(-1).unsqueeze(-1))
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max_], dim=1)
        att = torch.sigmoid(self.conv(x_cat))
        return x * att


class MSFM(nn.Module):
    def forward(self, x):
        top_hat = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        bottom_hat = F.avg_pool2d(x, 3, stride=1, padding=1) - x
        grad = torch.abs(top_hat - bottom_hat)
        return grad


class IterativeRefinement(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


def combined_loss(pred, target, alpha=0.85):
    mse = F.mse_loss(pred, target)
    ssim_val = ssim_loss_fn(pred, target, data_range=1.0)
    return alpha * mse + (1 - alpha) * (1 - ssim_val)


class CAViT_IMSFN(nn.Module):
    def __init__(self):
        super(CAViT_IMSFN, self).__init__()
        self.local = LocalExtractor()
        self.global_model = AViT()
        self.attn = SpatialAttention()
        self.refine = IterativeRefinement(96)
        self.ms = MSFM()
        self.out = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local = self.local(x)
        global_feat = F.interpolate(self.global_model(x), size=local.shape[2:], mode='bilinear', align_corners=False)
        fused = self.attn(local + global_feat)
        refined = self.refine(fused)
        ms_features = self.ms(refined)
        output = self.out(ms_features)
        output = F.interpolate(output, size=(224, 224), mode='bilinear', align_corners=False)

        return output

def prepare_data(root_dir, transform=None):
    focus1 = sorted([os.path.join(root_dir, 'source_1', f) for f in os.listdir(f"{root_dir}/source_1")])
    focus2 = sorted([os.path.join(root_dir, 'source_2', f) for f in os.listdir(f"{root_dir}/source_2")])
    target = sorted([os.path.join(root_dir, 'full_clear', f) for f in os.listdir(f"{root_dir}/full_clear")])

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = FusionDataset(focus1, focus2, target, transform)
    return dataset


def evaluate_metrics(output, target):
    output_np = output.squeeze().detach().cpu().numpy()
    target_np = target.squeeze().detach().cpu().numpy()
    return psnr(target_np, output_np, data_range=1.0), ssim(target_np, output_np, data_range=1.0)


def train_dataset(data_path, epochs=200):
    dataset = prepare_data(data_path)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = CAViT_IMSFN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = combined_loss

    metrics_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        model_path = f"trained.pth"
        torch.save(model.state_dict(), model_path)

        psnr, ssim, avg_time = test_dataset(
            r"D:\pycharm\pythonProject\Split\test",
            model_path=model_path,
            save_dir=None,
            verbose=False
        )

        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, Time: {avg_time:.4f}s")

        metrics_log.append([epoch, avg_loss, psnr, ssim, avg_time])

    with open("meaures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "PSNR", "SSIM", "Time"])
        writer.writerows(metrics_log)

def test_dataset(data_path, model_path, verbose=True):
    dataset = prepare_data(data_path)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CAViT_IMSFN().to(device)
    state_dict = torch.load(model_path, map_location=device,weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_psnr, total_ssim, times = 0, 0, []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            start = time.time()
            pred = model(x)
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            pred = F.interpolate(pred, size=y.shape[-2:], mode='bilinear', align_corners=False)
            end = time.time()

            p, s = evaluate_metrics(pred, y)
            total_psnr += p
            total_ssim += s
            times.append(end - start)


    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_time = np.mean(times)

    if verbose:
        print(f"Test PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, Time: {avg_time:.4f}s")

    return avg_psnr, avg_ssim, avg_time



train_dataset(r"Dataset", epochs=200)
test_dataset(r"Dataset", model_path="trained.pth")

