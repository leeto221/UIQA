import os
import random
import argparse
import traceback

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

from premodel import EPCFQA


class UnderwaterSyntheticDataset(Dataset):
    """Synthetic underwater image dataset for pretraining"""

    def __init__(self, degraded_dir, clear_dir, params_dir, transform=None, param_size=(256, 256)):
        self.degraded_dir = degraded_dir
        self.clear_dir = clear_dir
        self.params_dir = params_dir
        self.transform = transform
        self.param_size = param_size

        self.image_files = []
        for file in sorted(os.listdir(degraded_dir)):
            if file.endswith(".png"):
                clear_path = os.path.join(clear_dir, file)
                if os.path.exists(clear_path):
                    self.image_files.append(file)

        print(f"Found {len(self.image_files)} valid image pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_files[idx]
        img_base = os.path.splitext(img_name)[0]

        # Load degraded image I
        I_path = os.path.join(self.degraded_dir, img_name)
        I = Image.open(I_path).convert("RGB")

        # Load clear image J
        J_path = os.path.join(self.clear_dir, img_name)
        J = Image.open(J_path).convert("RGB")

        # Load physical parameters
        params_path = os.path.join(self.params_dir, f"{img_base}.npy")
        try:
            params = np.load(params_path, allow_pickle=True).item()
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file not found: {params_path}")

        def _to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.float().clone().contiguous()
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.float32)
            x = np.ascontiguousarray(x).copy()
            return torch.from_numpy(x).float().clone().contiguous()

        d_gt = _to_torch(params["depth"])         # [H, W]
        g_gt = _to_torch(params["g"])             # [H, W]
        L_gt = _to_torch(params["lighting"])      # [H, W]
        beta_D_gt = _to_torch(params["beta_D"])   # [H, W, 3]
        beta_B_gt = _to_torch(params["beta_B"])   # scalar or [3]
        B_inf = _to_torch(params["B_inf"])        # [3]

        if beta_D_gt.ndim == 3 and beta_D_gt.shape[-1] == 3:
            beta_D_gt = beta_D_gt.permute(2, 0, 1).clone()  # [3, H, W]
        else:
            raise ValueError(f"beta_D_gt shape incorrect: {beta_D_gt.shape}, expected [H, W, 3]")

        if d_gt.ndim == 2:
            d_gt = d_gt.unsqueeze(0).clone()  # [1, H, W]
        if g_gt.ndim == 2:
            g_gt = g_gt.unsqueeze(0).clone()  # [1, H, W]
        if L_gt.ndim == 2:
            L_gt = L_gt.unsqueeze(0).clone()  # [1, H, W]

        if beta_B_gt.dim() == 1:
            beta_B_gt = beta_B_gt.view(-1, 1, 1).clone()  # [3,1,1]
        elif beta_B_gt.dim() == 0:
            beta_B_gt = beta_B_gt.view(1, 1, 1).expand(3, 1, 1).clone()

        if B_inf.dim() == 1:
            B_inf = B_inf.view(-1, 1, 1).clone()

        # Resize parameter maps to target size
        d_gt = F.interpolate(d_gt.unsqueeze(0), size=self.param_size, mode="bilinear", align_corners=False).squeeze(0).clone()
        g_gt = F.interpolate(g_gt.unsqueeze(0), size=self.param_size, mode="bilinear", align_corners=False).squeeze(0).clone()
        L_gt = F.interpolate(L_gt.unsqueeze(0), size=self.param_size, mode="bilinear", align_corners=False).squeeze(0).clone()
        beta_D_gt = F.interpolate(beta_D_gt.unsqueeze(0), size=self.param_size, mode="bilinear", align_corners=False).squeeze(0).clone()

        if self.transform:
            I = self.transform(I).clone().contiguous()
            J = self.transform(J).clone().contiguous()

        return {
            "I": I,
            "J": J,
            "d_gt": d_gt.contiguous(),
            "beta_D_gt": beta_D_gt.contiguous(),
            "beta_B_gt": beta_B_gt.contiguous(),
            "g_gt": g_gt.contiguous(),
            "L_gt": L_gt.contiguous(),
            "B_inf": B_inf.contiguous(),
        }


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, path, is_ddp=False):
    """Save checkpoint"""
    model_state_dict = model.module.state_dict() if is_ddp else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, is_ddp=False):
    """Load checkpoint"""
    checkpoint = torch.load(path, map_location="cpu")

    if is_ddp:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]
    return model, optimizer, start_epoch, best_loss


def plot_losses(train_losses, val_losses, save_path):
    """Plot training / validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(save_path)
    plt.close()


def train_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
    """Train one epoch"""
    model.train()
    train_loss = 0.0
    rank = dist.get_rank() if dist.is_initialized() else 0

    train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]") if rank == 0 else train_loader

    for batch in train_progress:
        try:
            I = batch["I"].to(device, non_blocking=True)
            J = batch["J"].to(device, non_blocking=True)
            d_gt = batch["d_gt"].to(device, non_blocking=True)
            beta_D_gt = batch["beta_D_gt"].to(device, non_blocking=True)
            beta_B_gt = batch["beta_B_gt"].to(device, non_blocking=True)
            g_gt = batch["g_gt"].to(device, non_blocking=True)
            L_gt = batch["L_gt"].to(device, non_blocking=True)
            B_inf = batch["B_inf"].to(device, non_blocking=True)

            optimizer.zero_grad()

            if isinstance(model, DDP):
                loss, losses_dict = model.module.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf
                )
            else:
                loss, losses_dict = model.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf
                )

            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss

            if rank == 0:
                train_progress.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "depth": f"{losses_dict['depth_loss'].item():.4f}",
                    "beta": f"{(losses_dict['beta_D_loss'] + losses_dict['beta_B_loss']).item():.4f}",
                    "g": f"{losses_dict['g_loss'].item():.4f}",
                    "L": f"{losses_dict['L_loss'].item():.4f}",
                    "recon": f"{losses_dict['L_recon'].item():.4f}",
                })

        except Exception as e:
            if rank == 0:
                print(f"Error in training batch: {e}")
                traceback.print_exc()
            continue

    train_loss /= len(train_loader)

    if dist.is_initialized():
        train_loss_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()

    return train_loss


@torch.no_grad()
def validate(model, val_loader, device, epoch, total_epochs):
    """Validate one epoch"""
    model.eval()
    val_loss = 0.0
    rank = dist.get_rank() if dist.is_initialized() else 0

    val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Val]") if rank == 0 else val_loader

    for batch in val_progress:
        try:
            I = batch["I"].to(device, non_blocking=True)
            J = batch["J"].to(device, non_blocking=True)
            d_gt = batch["d_gt"].to(device, non_blocking=True)
            beta_D_gt = batch["beta_D_gt"].to(device, non_blocking=True)
            beta_B_gt = batch["beta_B_gt"].to(device, non_blocking=True)
            g_gt = batch["g_gt"].to(device, non_blocking=True)
            L_gt = batch["L_gt"].to(device, non_blocking=True)
            B_inf = batch["B_inf"].to(device, non_blocking=True)

            if isinstance(model, DDP):
                loss, losses_dict = model.module.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf
                )
            else:
                loss, losses_dict = model.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf
                )

            batch_loss = loss.item()
            val_loss += batch_loss

            if rank == 0:
                val_progress.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "depth": f"{losses_dict['depth_loss'].item():.4f}",
                    "beta": f"{(losses_dict['beta_D_loss'] + losses_dict['beta_B_loss']).item():.4f}",
                    "g": f"{losses_dict['g_loss'].item():.4f}",
                    "L": f"{losses_dict['L_loss'].item():.4f}",
                    "recon": f"{losses_dict['L_recon'].item():.4f}",
                })

        except Exception as e:
            if rank == 0:
                print(f"Error in validation batch: {e}")
                traceback.print_exc()
            continue

    val_loss /= len(val_loader)

    if dist.is_initialized():
        val_loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / dist.get_world_size()

    return val_loss


def train_worker(local_rank, world_size, args):
    """Main worker for single-GPU or DDP training"""
    gpu_id = args.gpu_ids[local_rank]

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    set_seed(args.seed + local_rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    if local_rank == 0:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
        print(f"GPU IDs: {args.gpu_ids}")
        print(f"Degraded dir: {args.degraded_dir}")
        print(f"Clear dir: {args.clear_dir}")
        print(f"Params dir: {args.params_dir}")
        print(f"Output dir: {args.output_dir}")

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    full_dataset = UnderwaterSyntheticDataset(
        degraded_dir=args.degraded_dir,
        clear_dir=args.clear_dir,
        params_dir=args.params_dir,
        transform=transform,
        param_size=(256, 256),
    )

    dataset_size = len(full_dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    if local_rank == 0:
        print(f"Dataset size: {dataset_size}, Train: {train_size}, Val: {val_size}")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers_train,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers_val,
        pin_memory=True,
        drop_last=False,
    )

    model = EPCFQA(image_size=256).to(device)

    is_ddp = world_size > 1
    if is_ddp:
        model = DDP(model, device_ids=[gpu_id])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    if args.resume:
        if local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.resume, is_ddp=is_ddp
        )
        if local_rank == 0:
            print(f"Resume from epoch {start_epoch}, best val loss {best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, device, epoch, args.epochs)

        scheduler.step(val_loss)

        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, is_ddp=is_ddp)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, val_loss, best_model_path, is_ddp=is_ddp)
                print(f"Best model saved, val loss: {val_loss:.6f}")

            plot_losses(train_losses, val_losses, os.path.join(args.output_dir, "losses.png"))

            with open(os.path.join(args.output_dir, "losses.txt"), "w") as f:
                for tr, va in zip(train_losses, val_losses):
                    f.write(f"{tr:.6f},{va:.6f}\n")

        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Pretrain R-PPEM module")

    parser.add_argument("--degraded_dir", type=str, required=True, help="Degraded image directory")
    parser.add_argument("--clear_dir", type=str, required=True, help="Clear image directory")
    parser.add_argument("--params_dir", type=str, required=True, help="Parameter directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")

    parser.add_argument("--num_workers_train", type=int, default=8, help="Train dataloader workers")
    parser.add_argument("--num_workers_val", type=int, default=4, help="Validation dataloader workers")

    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0, 1, 2, 3], help="GPU ids to use")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))

    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        world_size = min(len(args.gpu_ids), visible_gpu_count)
        args.gpu_ids = list(range(world_size))
        print(f"Visible GPUs: {visible_gpu_count}, Using: {world_size}")
    else:
        world_size = 1
        args.gpu_ids = [0]
        print("No GPU detected, using CPU")

    if world_size == 1:
        train_worker(0, 1, args)
    else:
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
