import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from model import EPCFQA
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

class UnderwaterSyntheticDataset(Dataset):
    """Synthetic underwater image dataset"""
    def __init__(self, degraded_dir, clear_dir, params_dir, transform=None, param_size=(256, 256)):
        """
        Args:
            degraded_dir (string): Directory with degraded images
            clear_dir (string): Directory with clear images
            params_dir (string): Directory with physical parameter .npy files
            transform (callable, optional): Optional transform to be applied on images
            param_size (tuple): Target size for physical parameters, default (256, 256)
        """
        self.degraded_dir = degraded_dir
        self.clear_dir = clear_dir
        self.params_dir = params_dir
        self.transform = transform
        self.param_size = param_size  # Target size for physical parameters

        # Get all image files
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

        # Get image file name
        img_name = self.image_files[idx]
        img_base = os.path.splitext(img_name)[0]

        # Load degraded image I
        I_path = os.path.join(self.degraded_dir, img_name)
        I = Image.open(I_path).convert('RGB')

        # Load clear image J
        J_path = os.path.join(self.clear_dir, img_name)
        J = Image.open(J_path).convert('RGB')

        # Load physical parameters
        params_path = os.path.join(self.params_dir, f"{img_base}.npy")
        try:
            params = np.load(params_path, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Parameter file not found: {params_path}")
            raise FileNotFoundError(f"Parameter file not found: {params_path}")

        # Convert parameters to torch tensors
        def _to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.float().clone().contiguous()
            if not isinstance(x, np.ndarray):
                x = np.array(x, dtype=np.float32)
            x = np.ascontiguousarray(x)
            x = x.copy()  # Prevent "read-only"
            return torch.from_numpy(x).float().clone().contiguous()

        # Load parameters
        d_gt = _to_torch(params['depth'])       # [H, W]
        g_gt = _to_torch(params['g'])           # [H, W]
        L_gt = _to_torch(params['lighting'])    # [H, W]
        beta_D_gt = _to_torch(params['beta_D']) # [H, W, 3]
        beta_B_gt = _to_torch(params['beta_B']) # scalar or [3]
        B_inf = _to_torch(params['B_inf'])      # [3]

        # Adjust beta_D_gt to [3, H, W]
        if len(beta_D_gt.shape) == 3 and beta_D_gt.shape[-1] == 3:
            beta_D_gt = beta_D_gt.permute(2, 0, 1).clone()  # from [H, W, 3] to [3, H, W]
        else:
            raise ValueError(f"beta_D_gt shape incorrect: {beta_D_gt.shape}, expected [H, W, 3]")

        # Ensure other parameters are [C, H, W]
        if len(d_gt.shape) == 2:
            d_gt = d_gt.unsqueeze(0).clone()  # [1, H, W]
        if len(g_gt.shape) == 2:
            g_gt = g_gt.unsqueeze(0).clone()  # [1, H, W]
        if len(L_gt.shape) == 2:
            L_gt = L_gt.unsqueeze(0).clone()  # [1, H, W]

        # Adjust beta_B_gt and B_inf to [3, 1, 1]
        if beta_B_gt.dim() == 1:
            beta_B_gt = beta_B_gt.view(-1, 1, 1).clone()
        elif beta_B_gt.dim() == 0:
            beta_B_gt = beta_B_gt.view(1, 1, 1).expand(3, 1, 1).clone()
        if B_inf.dim() == 1:
            B_inf = B_inf.view(-1, 1, 1).clone()

        # Resize physical parameters to target size (e.g. 256x256)
        d_gt = F.interpolate(d_gt.unsqueeze(0), size=self.param_size, mode='bilinear', align_corners=False).squeeze(0).clone()
        g_gt = F.interpolate(g_gt.unsqueeze(0), size=self.param_size, mode='bilinear', align_corners=False).squeeze(0).clone()
        L_gt = F.interpolate(L_gt.unsqueeze(0), size=self.param_size, mode='bilinear', align_corners=False).squeeze(0).clone()
        beta_D_gt = F.interpolate(beta_D_gt.unsqueeze(0), size=self.param_size, mode='bilinear', align_corners=False).squeeze(0).clone()

        # Apply image transform
        if self.transform:
            I = self.transform(I).clone().contiguous()
            J = self.transform(J).clone().contiguous()

        # Ensure all tensors are contiguous
        d_gt = d_gt.contiguous()
        g_gt = g_gt.contiguous()
        L_gt = L_gt.contiguous()
        beta_D_gt = beta_D_gt.contiguous()
        beta_B_gt = beta_B_gt.contiguous()
        B_inf = B_inf.contiguous()

        return {
            'I': I,                 # Degraded image [3, 256, 256]
            'J': J,                 # Clear image [3, 256, 256]
            'd_gt': d_gt,           # Depth [1, 256, 256]
            'beta_D_gt': beta_D_gt, # Absorption coefficient [3, 256, 256]
            'beta_B_gt': beta_B_gt, # Background absorption coefficient [3, 1, 1]
            'g_gt': g_gt,           # Anisotropy coefficient [1, 256, 256]
            'L_gt': L_gt,           # Lighting [1, 256, 256]
            'B_inf': B_inf          # Background light [3, 1, 1]
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
    # If DDP model, only save module part
    if is_ddp:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, is_ddp=False):
    """Load checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    if is_ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    return model, optimizer, start_epoch, loss

def plot_losses(train_losses, val_losses, save_path):
    """Plot loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, optimizer, device, args, epoch):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Progress bar only in main process
    if rank == 0:
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    else:
        train_progress = train_loader
    
    for batch in train_progress:
        try:
            # Move data to device
            I = batch['I'].to(device)
            J = batch['J'].to(device)
            d_gt = batch['d_gt'].to(device)
            beta_D_gt = batch['beta_D_gt'].to(device)
            beta_B_gt = batch['beta_B_gt'].to(device)
            g_gt = batch['g_gt'].to(device)
            L_gt = batch['L_gt'].to(device)
            B_inf = batch['B_inf'].to(device)
            
            # Forward and loss
            optimizer.zero_grad()
            
            # Check if DDP model
            if isinstance(model, DDP):
                loss, losses_dict = model.module.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf,
                    lambda_d=args.lambda_d, 
                    lambda_beta=args.lambda_beta,
                    lambda_g=args.lambda_g, 
                    lambda_L=args.lambda_L,
                    lambda_phys=args.lambda_phys
                )
            else:
                loss, losses_dict = model.pretrain_rppem(
                    I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf,
                    lambda_d=args.lambda_d, 
                    lambda_beta=args.lambda_beta,
                    lambda_g=args.lambda_g, 
                    lambda_L=args.lambda_L,
                    lambda_phys=args.lambda_phys
                )
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar and loss
            batch_loss = loss.item()
            train_loss += batch_loss
            
            if rank == 0:
                train_progress.set_postfix({
                    'loss': batch_loss,
                    'depth_loss': losses_dict['depth_loss'].item(),
                    'beta_loss': (losses_dict['beta_D_loss'] + losses_dict['beta_B_loss']).item(),
                    'g_loss': losses_dict['g_loss'].item(),
                    'L_loss': losses_dict['L_loss'].item(),
                    'phys_loss': losses_dict['phys_loss'].item()
                })
        except Exception as e:
            if rank == 0:
                print(f"Error in batch: {str(e)}")
                import traceback
                traceback.print_exc()
            continue
    
    # Average training loss
    train_loss /= len(train_loader)
    
    # If distributed, gather all losses and average
    if dist.is_initialized():
        train_loss_tensor = torch.tensor([train_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / dist.get_world_size()
    
    return train_loss

def validate(model, val_loader, device, args, epoch):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Progress bar only in main process
    if rank == 0:
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
    else:
        val_progress = val_loader
    
    with torch.no_grad():
        for batch in val_progress:
            try:
                # Move data to device
                I = batch['I'].to(device)
                J = batch['J'].to(device)
                d_gt = batch['d_gt'].to(device)
                beta_D_gt = batch['beta_D_gt'].to(device)
                beta_B_gt = batch['beta_B_gt'].to(device)
                g_gt = batch['g_gt'].to(device)
                L_gt = batch['L_gt'].to(device)
                B_inf = batch['B_inf'].to(device)
                
                # Forward and loss
                if isinstance(model, DDP):
                    loss, losses_dict = model.module.pretrain_rppem(
                        I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf,
                        lambda_d=args.lambda_d, 
                        lambda_beta=args.lambda_beta,
                        lambda_g=args.lambda_g, 
                        lambda_L=args.lambda_L,
                        lambda_phys=args.lambda_phys
                    )
                else:
                    loss, losses_dict = model.pretrain_rppem(
                        I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf,
                        lambda_d=args.lambda_d, 
                        lambda_beta=args.lambda_beta,
                        lambda_g=args.lambda_g, 
                        lambda_L=args.lambda_L,
                        lambda_phys=args.lambda_phys
                    )
                
                # Update loss
                batch_loss = loss.item()
                val_loss += batch_loss
                
                if rank == 0:
                    val_progress.set_postfix({
                        'loss': batch_loss,
                        'depth_loss': losses_dict['depth_loss'].item(),
                        'beta_loss': (losses_dict['beta_D_loss'] + losses_dict['beta_B_loss']).item(),
                        'g_loss': losses_dict['g_loss'].item(),
                        'L_loss': losses_dict['L_loss'].item(),
                        'phys_loss': losses_dict['phys_loss'].item()
                    })
            except Exception as e:
                if rank == 0:
                    print(f"Error in validation batch: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue
    
    # Average validation loss
    val_loss /= len(val_loader)
    
    # If distributed, gather all losses and average
    if dist.is_initialized():
        val_loss_tensor = torch.tensor([val_loss], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / dist.get_world_size()
    
    return val_loss


def train(local_rank, world_size, args):
    """Training function"""
    # Map local_rank to actual GPU ID
    gpu_id = args.gpu_ids[local_rank]
    
    # Initialize distributed environment
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    # Set random seed (add rank to ensure different seed for each process)
    set_seed(args.seed + local_rank)
    
    # Set device - use mapped GPU ID
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(gpu_id)
    
    if local_rank == 0:
        print(f"Using device: {device} (GPU ID: {gpu_id})")
        print(f"World size: {world_size}")
        print(f"GPU IDs: {args.gpu_ids}")
        print(f"Degraded image dir: {args.degraded_dir}")
        print(f"Clear image dir: {args.clear_dir}")
        print(f"Params dir: {args.params_dir}")
        print(f"Output dir: {args.output_dir}")
    
    # Create output directory
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    full_dataset = UnderwaterSyntheticDataset(
        degraded_dir=args.degraded_dir,
        clear_dir=args.clear_dir,
        params_dir=args.params_dir,
        transform=transform,
        param_size=(256, 256)
    )

    
    # Split train/val (80%/20%)
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size
    
    if local_rank == 0:
        print(f"Dataset size: {dataset_size}, Train: {train_size}, Val: {val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler
    )
    
    # Create model
    model = EPCFQA(image_size=256).to(device)
    
    # If distributed, wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id])
        is_ddp = True
    else:
        is_ddp = False
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Create LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Init variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Resume from checkpoint if provided
    if args.resume:
        if local_rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.resume, is_ddp
        )
        if local_rank == 0:
            print(f"Start epoch: {start_epoch}, Best val loss: {best_val_loss}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for distributed samplers
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device, args, epoch)
        
        # Adjust LR
        scheduler.step(val_loss)
        
        # Save model and log only in main process
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, is_ddp)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, val_loss, best_model_path, is_ddp)
                print(f"Best model saved, val loss: {val_loss:.6f}")
            
            plot_losses(train_losses, val_losses, os.path.join(args.output_dir, 'losses.png'))
            
            with open(os.path.join(args.output_dir, 'losses.txt'), 'w') as f:
                for t_loss, v_loss in zip(train_losses, val_losses):
                    f.write(f"{t_loss:.6f},{v_loss:.6f}\n")
        
        # Sync all processes
        if world_size > 1:
            dist.barrier()
    
    # Clean up distributed environment
    if world_size > 1:
        dist.destroy_process_group()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pretrain R-PPEM module')
    # Data paths
    parser.add_argument('--degraded_dir', type=str, 
                        default=None,
                        help='Degraded image directory')
    parser.add_argument('--clear_dir', type=str, 
                        default=None,
                        help='Clear image directory')
    parser.add_argument('--params_dir', type=str, 
                        default=None,
                        help='Parameter directory')
    parser.add_argument('--output_dir', type=str, 
                        default=None,
                        help='Output directory')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    
    # Loss weights
    parser.add_argument('--lambda_d', type=float, default=1.0, help='Depth loss weight')
    parser.add_argument('--lambda_beta', type=float, default=0.6, help='Absorption coefficient loss weight')
    parser.add_argument('--lambda_g', type=float, default=0.4, help='Anisotropy coefficient loss weight')
    parser.add_argument('--lambda_L', type=float, default=0.4, help='Lighting loss weight') 
    parser.add_argument('--lambda_phys', type=float, default=1.0, help='Physical consistency loss weight')
    
    # Distributed training
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1, 2, 3], 
                       help='List of GPU IDs to use')
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    
    # Check available GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        world_size = len(args.gpu_ids)
        print(f"Available GPUs: {available_gpus}, Using: {world_size}")
        args.gpu_ids = list(range(world_size))
    else:
        world_size = 1
        args.gpu_ids = [0]
        print("No GPU detected, using CPU for training")
    
    # Single GPU: call train directly
    if world_size == 1:
        train(0, 1, args)
    else:
        # Multi-process distributed training
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
