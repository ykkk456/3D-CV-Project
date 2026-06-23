import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import monai

# ============================================================
# 预处理
# ============================================================

# ── CT 值窗宽窗位 ──
CT_WINDOW_MIN = 0      # HU
CT_WINDOW_MAX = 400    # HU  (颅内动脉瘤 CTA 常用范围)

# ── patch 尺寸 ──
PATCH_D = 64
PATCH_H = 128
PATCH_W = 128


def _load_nii(path):
    """加载 .nii.gz 文件，返回 float32 numpy 数组 (D,H,W) 格式。"""
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    # nibabel 返回 (H, W, D) → 转为 (D, H, W)，和 PyTorch 的 C,D,H,W 一致
    return np.ascontiguousarray(np.transpose(data, (2, 0, 1)))


def norm_ct(img, window_min=CT_WINDOW_MIN, window_max=CT_WINDOW_MAX):
    """CT 窗宽窗位截断后归一化到 [0, 1]。"""
    img = np.clip(img, window_min, window_max)
    img = (img - window_min) / max(window_max - window_min, 1)
    return img


def _find_aneurysm_centers(seg):
    """
    找到所有动脉瘤连通域的质心坐标。
    返回 list of (d, h, w) int 坐标。
    """
    from scipy import ndimage
    labeled, n = ndimage.label(seg > 0)
    centers = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)  # (N, 3) 每行 [d, h, w]
        center = coords.mean(axis=0)        # 质心 [d, h, w]
        centers.append(tuple(center.astype(int)))
    return centers


def _random_crop_around_center(img, seg, center, half_d, half_h, half_w,
                                max_shift=8):
    """
    以 center 为中心，随机平移 max_shift 后截取 patch。
    边界不足时用 0 (img) / 0 (seg) 填充。
    """
    d_mid, h_mid, w_mid = center
    d_mid += np.random.randint(-max_shift, max_shift + 1)
    h_mid += np.random.randint(-max_shift, max_shift + 1)
    w_mid += np.random.randint(-max_shift, max_shift + 1)

    d_start = d_mid - half_d
    h_start = h_mid - half_h
    w_start = w_mid - half_w

    D, H, W = img.shape

    # ── pad if necessary ──
    pad_before = (max(0, -d_start), max(0, -h_start), max(0, -w_start))
    pad_after  = (max(0, d_start + 2 * half_d - D),
                  max(0, h_start + 2 * half_h - H),
                  max(0, w_start + 2 * half_w - W))

    if any(p > 0 for p in pad_before + pad_after):
        img = np.pad(img, list(zip(pad_before, pad_after)),
                     mode='constant', constant_values=0)
        seg = np.pad(seg, list(zip(pad_before, pad_after)),
                     mode='constant', constant_values=0)
        d_start += pad_before[0]
        h_start += pad_before[1]
        w_start += pad_before[2]

    patch_img = img[d_start:d_start+2*half_d,
                    h_start:h_start+2*half_h,
                    w_start:w_start+2*half_w].copy()

    patch_seg = seg[d_start:d_start+2*half_d,
                    h_start:h_start+2*half_h,
                    w_start:w_start+2*half_w].copy()

    return patch_img, patch_seg


class AneurysmDataset(Dataset):
    """颅内动脉瘤 CT 图像数据集。

    每个 epoch：
      - 对每个有动脉瘤的连通域采样 1 个正样本 patch
      - 再随机采样一些负样本 patch（无动脉瘤区域）
    """

    def __init__(self, image_dir, seg_dir, patch_size=(64, 128, 128),
                 positives_per_epoch=1, negatives_per_epoch=4):
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.d, self.h, self.w = patch_size
        self.pos_per_epoch = positives_per_epoch
        self.neg_per_epoch = negatives_per_epoch

        print(f"  → 扫描目录: {image_dir}")
        self.ids = sorted(
            f.replace('.nii.gz', '') for f in os.listdir(image_dir)
            if f.endswith('.nii.gz'))
        print(f"  → 找到 {len(self.ids)} 个样本")

        # 只预计算动脉瘤中心坐标，不预加载图像（单张 ~768 MB，全加载会炸内存）
        self.positive_centers = {}  # id → list of center
        total_aneurysms = 0
        for pid in self.ids:
            seg_path = os.path.join(seg_dir, pid + '.nii.gz')
            seg = _load_nii(seg_path)        # 临时加载 seg
            centers = _find_aneurysm_centers(seg)
            self.positive_centers[pid] = centers
            total_aneurysms += len(centers)
            del seg                           # 立即释放

        print(f"  → 扫描完成，共 {total_aneurysms} 个动脉瘤连通域")

        self.half_d = self.d // 2
        self.half_h = self.h // 2
        self.half_w = self.w // 2

        # 构建采样列表（每个样本 = (id, center or None)）
        self._build_sample_list()
        print(f"  → 每个 epoch 采样 {len(self.samples)} 个 patch")

    def _build_sample_list(self):
        self.samples = []
        for pid in self.ids:
            centers = self.positive_centers[pid]
            # 正样本
            for c in centers:
                for _ in range(self.pos_per_epoch):
                    self.samples.append((pid, c))
            # 负样本：随机位置
            for _ in range(self.neg_per_epoch):
                self.samples.append((pid, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, center = self.samples[idx]

        # 按需从磁盘加载
        img_path = os.path.join(self.image_dir, pid + '.nii.gz')
        seg_path = os.path.join(self.seg_dir, pid + '.nii.gz')
        img = _load_nii(img_path)
        img = norm_ct(img)
        seg = _load_nii(seg_path)

        if center is not None:
            # 正样本：在动脉瘤中心附近随机取 patch
            img_patch, seg_patch = _random_crop_around_center(
                img, seg, center,
                self.half_d, self.half_h, self.half_w,
                max_shift=6)
        else:
            # 负样本：从全图随机取 patch（要保证里面确实没有动脉瘤）
            D, H, W = img.shape
            # 尽可能不碰到动脉瘤：如果有正样本中心，远离它们
            for _ in range(20):  # 最多试 20 次
                d0 = np.random.randint(0, max(1, D - self.d))
                h0 = np.random.randint(0, max(1, H - self.h))
                w0 = np.random.randint(0, max(1, W - self.w))
                patch = seg[d0:d0+self.d, h0:h0+self.h, w0:w0+self.w]
                if patch.sum() < 10:  # 几乎没动脉瘤体素
                    break
            img_patch = img[d0:d0+self.d, h0:h0+self.h, w0:w0+self.w].copy()
            seg_patch = seg[d0:d0+self.d, h0:h0+self.h, w0:w0+self.w].copy()

        # 加 channel 维度： (D,H,W) → (1,D,H,W)
        img_t = torch.from_numpy(img_patch).unsqueeze(0)
        seg_t = torch.from_numpy(seg_patch).unsqueeze(0).float()
        return img_t, seg_t


def get_dataloaders(train_dir, train_seg_dir,
                    test_dir, test_seg_dir,
                    patch_size=(64, 128, 128),
                    batch_size=2, num_workers=0):
    """返回 train_loader, test_loader。"""
    print("=" * 55)
    print("[1/5] 加载训练集...")
    train_ds = AneurysmDataset(train_dir, train_seg_dir, patch_size,
                               positives_per_epoch=1, negatives_per_epoch=4)
    print("[2/5] 加载测试集...")
    test_ds = AneurysmDataset(test_dir, test_seg_dir, patch_size,
                              positives_per_epoch=1, negatives_per_epoch=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    print(f"  → train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    return train_loader, test_loader



# model

class cross_attention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.linear_e = nn.Linear(in_channels, out_channels)
        self.linear_d1 = nn.Linear(in_channels, out_channels)
        self.linear_d2 = nn.Linear(in_channels, out_channels)
        self.num_heads = num_heads
    def forward(self, e, d): # e: [B, C, D, H, W], d: [B, C, D, H, W]
        # 特征变换，做多头注意力
        B, C, D, H, W = e.shape
        N = D * H * W
        out_dim = self.linear_e.out_features
        head_dim = out_dim // self.num_heads

        # 展平空间维度: (B, C, D, H, W) → (B, N, C)
        e_flat = e.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        d_flat = d.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # Q ← linear_e(e), K ← linear_d1(d), V ← linear_d2(d)
        # 投影: (B, N, C) → (B, N, out_dim)
        Q = self.linear_e(e_flat)
        K = self.linear_d1(d_flat)
        V = self.linear_d2(d_flat)

        # 拆多头: (B, N, out_dim) → (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        Q = Q.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # 计算注意力权重
        attn = Q @ K.transpose(-2, -1) / (head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = torch.softmax(attn, dim=-1)

        # 加权 V 并合并多头: (B, num_heads, N, head_dim) → (B, N, out_dim)
        out = attn @ V  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, out_dim)  # (B, N, out_dim)

        # 重塑回原始空间尺寸: (B, N, out_dim) → (B, out_dim, D, H, W)
        out = out.transpose(1, 2).view(B, out_dim, D, H, W)

        return out








class seg_head(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode='trilinear', align_corners=False)
        return x 

class Loss(nn.Module):
    """Deep supervision loss with learnable head weights.

    Each decoder head contributes one loss term. The 4 per-head losses
    are stacked into a [4]-vector and combined by a learnable Linear(4,1).
    """
    def __init__(self, beta=0.5):
        super().__init__()
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True)
        self.cross_entropy_loss = nn.BCEWithLogitsLoss()
        self.beta = beta
        # learnable combination weights for 4 decoder heads
        self.combine = nn.Linear(4, 1, bias=False)

    def forward(self, x1, x2, x3, x4, y):
        # order: coarse -> fine (x4=deepest, x1=final)
        losses = []
        for x in [x4, x3, x2, x1]:
            dice = self.dice_loss(x, y)
            bce = self.cross_entropy_loss(x, y)
            losses.append(dice + self.beta * bce)

        loss_vec = torch.stack(losses)               # [4]
        return self.combine(loss_vec.unsqueeze(0)).squeeze()  # scalar    

def compute_dice_score(pred, target, eps=1e-6):
    """Compute batch Dice coefficient for binary segmentation.

    pred: [B, 1, D, H, W] logits
    target: [B, 1, D, H, W] binary mask (0/1)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3, 4))
    union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    return (2 * intersection / (union + eps)).mean()
class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # encoder
        self.bn1 = nn.BatchNorm3d(1)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(256)
        self.bn4 = nn.BatchNorm3d(512)

        self.resconv1 = nn.Conv3d(1, 128, kernel_size=1)
        self.resconv2 = nn.Conv3d(128, 256, kernel_size=1)
        self.resconv3 = nn.Conv3d(256, 512, kernel_size=1)
        self.resconv4 = nn.Conv3d(512, 1024, kernel_size=1)

        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)

        # decoder
        self.deconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose3d(128, out_channels, kernel_size=2, stride=2)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        # seg-heads: L4→L1  (deepest → shallowest)
        self.seghead4 = seg_head(512, 1, scale_factor=8)
        self.seghead3 = seg_head(256, 1, scale_factor=4)
        self.seghead2 = seg_head(128, 1, scale_factor=2)
        self.seghead1 = seg_head(out_channels, 1, scale_factor=1)

        # cross-attention
        self.attn1 = cross_attention(512, 512, num_heads=8)
        self.attn2 = cross_attention(256, 256, num_heads=8)
        self.attn3 = cross_attention(128, 128, num_heads=8)
        self.attn4 = cross_attention(out_channels, out_channels, num_heads=8)

    def forward(self, x):  # x: [B, C, D, H, W]
        # ── encoder ──
        x = self.bn1(x)
        e1 = self.relu(self.resconv1(x) + self.conv1(x))
        x = self.maxpool(e1)

        x = self.bn2(x)
        e2 = self.relu(self.resconv2(x) + self.conv2(x))
        x = self.maxpool(e2)

        x = self.bn3(x)
        e3 = self.relu(self.resconv3(x) + self.conv3(x))
        x = self.maxpool(e3)

        x = self.bn4(x)
        e4 = self.relu(self.resconv4(x) + self.conv4(x))
        x = self.maxpool(e4)

        # ── decoder ──
        d4 = self.deconv1(x)                           # bottleneck
        p4 = self.seghead4(d4)

        d3 = self.attn1(d4, e3)
        d3 = self.deconv2(d3)
        p3 = self.seghead3(d3)

        d2 = self.attn2(d3, e2)
        d2 = self.deconv3(d2)
        p2 = self.seghead2(d2)

        d1 = self.attn3(d2, e1)
        d1 = self.deconv4(d1)
        p1 = self.seghead1(d1)

        return p1, p2, p3, p4    # shallowest → deepest


# ============================================================
# 训练
# ============================================================

def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    n_batches = len(loader)
    for i, (img, seg) in enumerate(loader):
        img, seg = img.to(device), seg.to(device)
        p1, p2, p3, p4 = model(img)
        loss = loss_fn(p1, p2, p3, p4, seg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += compute_dice_score(p1, seg).item()
        count += img.size(0)

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"  Epoch {epoch:3d} [{i+1:4d}/{n_batches}] "
                  f"loss: {total_loss/(i+1):.4f}  dice: {total_dice/(i+1):.4f}")

    return total_loss / max(count, 1), total_dice / max(count, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    for img, seg in loader:
        img, seg = img.to(device), seg.to(device)
        p1, p2, p3, p4 = model(img)
        loss = loss_fn(p1, p2, p3, p4, seg)
        total_loss += loss.item()
        total_dice += compute_dice_score(p1, seg).item()
        count += img.size(0)

    return total_loss / max(count, 1), total_dice / max(count, 1)


def main():
    # ── 配置 ──
    DATA_ROOT = "datas"
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA 不可用，使用 CPU")

    # ── 数据 ──
    train_loader, test_loader = get_dataloaders(
        train_dir=os.path.join(DATA_ROOT, "data_train"),
        train_seg_dir=os.path.join(DATA_ROOT, "data_annotation_tr"),
        test_dir=os.path.join(DATA_ROOT, "data_test"),
        test_seg_dir=os.path.join(DATA_ROOT, "data_annotationj_ts"),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # ── 模型 ──
    print("=" * 55)
    print("[3/5] 构建模型...")
    model = Unet(in_channels=1, out_channels=1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  → 参数量: {n_params:,}")
    print("[4/5] 初始化损失函数 & 优化器...")
    loss_fn = Loss(beta=0.5).to(DEVICE)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    print(f"  → 优化器: AdamW  lr={LR}  scheduler=CosineAnnealingLR(T={NUM_EPOCHS})")

    print("=" * 55)
    print("[5/5] 开始训练...")
    print("=" * 55)
    best_dice = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE, epoch)
        val_loss, val_dice = validate(model, test_loader, loss_fn, DEVICE)
        scheduler.step()

        print(f"  → Epoch {epoch:3d} summary | "
              f"T loss: {train_loss:.4f}  dice: {train_dice:.4f} | "
              f"V loss: {val_loss:.4f}  dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"       ✅ saved best (dice={best_dice:.4f})")

    print(f"\nDone. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()



        




