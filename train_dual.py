import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy import stats  # 用于计算 PLCC, SRCC, KRCC

# 引入你的双流模型
from model_dual import PhysicsGuidedIQA
 
# -----------------------------------------------------------------------------
# Dataset & Utils
# -----------------------------------------------------------------------------

class IQADataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found: {label_file}")

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    score = float(parts[1])
                    self.data.append((img_name, score))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, score = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            I = Image.open(img_path).convert('RGB')
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            I = Image.new('RGB', (256, 256))

        if self.transform:
            I = self.transform(I)

        return I, torch.tensor(score, dtype=torch.float32)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Training & Validation Loops
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train() 
    train_loss = 0.0
    
    # 进度条只在主进程显示，或者简化输出
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    
    for imgs, scores in pbar:
        imgs = imgs.to(device)
        scores = scores.to(device).unsqueeze(1) 

        optimizer.zero_grad()
        preds = model(imgs)
        loss = F.mse_loss(preds, scores) 
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
            
    return train_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    preds_list = []
    scores_list = []
    
    with torch.no_grad():
        for imgs, scores in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            
            preds_list.extend(preds.cpu().numpy().flatten())
            scores_list.extend(scores.numpy().flatten())

    preds_all = np.array(preds_list)
    scores_all = np.array(scores_list)

    # --- 计算 4 个指标 ---
    srcc, _ = stats.spearmanr(preds_all, scores_all)
    plcc, _ = stats.pearsonr(preds_all, scores_all)
    krcc, _ = stats.kendalltau(preds_all, scores_all)
    mse = np.mean((preds_all - scores_all)**2)
    rmse = np.sqrt(mse)

    return rmse, srcc, plcc, krcc

# -----------------------------------------------------------------------------
# Single Session Worker
# -----------------------------------------------------------------------------

def run_session(gpu_id, args, current_seed, result_queue):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    set_seed(current_seed)
    
    print(f"\n>>> Starting Session with Seed: {current_seed}")

    # 1. Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = IQADataset(args.image_dir, args.mos_file, transform=transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(current_seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # 2. Model
    print(f"Loading Pretrained R-PPEM from: {args.rppem_path}")
    model = PhysicsGuidedIQA(pretrained_rppem_path=args.rppem_path).to(device)
    
    # Optimizer: Only train requires_grad=True parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. Logging Setup
    log_file = os.path.join(args.output_dir, f"log_seed_{current_seed}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Val_RMSE,Val_SRCC,Val_PLCC,Val_KRCC\n")

    # 4. Training Loop
    best_stats = {
        'srcc': -1.0,
        'plcc': -1.0,
        'krcc': -1.0,
        'rmse': 999.0,
        'epoch': -1
    }

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        val_rmse, val_srcc, val_plcc, val_krcc = validate(model, val_loader, device)
        
        scheduler.step()

        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_rmse:.6f},{val_srcc:.6f},{val_plcc:.6f},{val_krcc:.6f}\n")

        # Save Best Model (Based on SRCC)
        if val_srcc > best_stats['srcc']:
            best_stats['srcc'] = val_srcc
            best_stats['plcc'] = val_plcc
            best_stats['krcc'] = val_krcc
            best_stats['rmse'] = val_rmse
            best_stats['epoch'] = epoch + 1
            
            save_path = os.path.join(args.output_dir, f"best_model_seed_{current_seed}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': best_stats
            }, save_path)
        
        # [Corrected Print Statement]
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | SRCC={val_srcc:.4f} PLCC={val_plcc:.4f} KRCC={val_krcc:.4f} RMSE={val_rmse:.4f}")

    print(f"<<< Finished Seed {current_seed}. Best SRCC: {best_stats['srcc']:.4f} @ Epoch {best_stats['epoch']}")
    
    # Put result in queue
    result_queue.put((current_seed, best_stats))

# -----------------------------------------------------------------------------
# Main Manager
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--rppem_path', type=str, required=True, help='Path to frozen R-PPEM weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of images')
    parser.add_argument('--mos_file', type=str, required=True, help='Txt file: filename score')
    parser.add_argument('--output_dir', type=str, default='./output_dual')
    
    # Config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_seed', type=int, default=43, help='Starting seed')
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeated runs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use multiprocessing
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    results = []

    for i in range(args.repeats):
        current_seed = args.base_seed + i
        
        p = ctx.Process(target=run_session, args=(args.gpu_id, args, current_seed, result_queue))
        p.start()
        p.join()
        
        if not result_queue.empty():
            res = result_queue.get()
            results.append(res)
        else:
            print(f"Error: Session {current_seed} failed to return results.")

    # Summary
    if results:
        # results structure: [(seed, {'srcc':..., 'plcc':..., ...}), ...]
        srccs = [r[1]['srcc'] for r in results]
        plccs = [r[1]['plcc'] for r in results]
        krccs = [r[1]['krcc'] for r in results]
        rmses = [r[1]['rmse'] for r in results]
        
        print("\n============================================================")
        print("FINAL RESULTS SUMMARY (Best Epoch Metrics)")
        print("============================================================")
        print(f"{'Seed':<6} | {'SRCC':<8} | {'PLCC':<8} | {'KRCC':<8} | {'RMSE':<8}")
        print("-" * 60)
        
        for r in results:
            s = r[0]
            m = r[1]
            print(f"{s:<6} | {m['srcc']:.4f}   | {m['plcc']:.4f}   | {m['krcc']:.4f}   | {m['rmse']:.4f}")
            
        print("-" * 60)
        print(f"AVG    | {np.mean(srccs):.4f}   | {np.mean(plccs):.4f}   | {np.mean(krccs):.4f}   | {np.mean(rmses):.4f}")
        print(f"STD    | {np.std(srccs):.4f}   | {np.std(plccs):.4f}   | {np.std(krccs):.4f}   | {np.std(rmses):.4f}")
        print("============================================================")
        
        # Save summary to file
        with open(os.path.join(args.output_dir, "summary_metrics.txt"), "w") as f:
            f.write("Seed,SRCC,PLCC,KRCC,RMSE\n")
            for r in results:
                m = r[1]
                f.write(f"{r[0]},{m['srcc']:.4f},{m['plcc']:.4f},{m['krcc']:.4f},{m['rmse']:.4f}\n")
            f.write("\n")
            f.write(f"Average SRCC: {np.mean(srccs):.4f} +/- {np.std(srccs):.4f}\n")
            f.write(f"Average PLCC: {np.mean(plccs):.4f} +/- {np.std(plccs):.4f}\n")
            f.write(f"Average KRCC: {np.mean(krccs):.4f} +/- {np.std(krccs):.4f}\n")
            f.write(f"Average RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}\n")

if __name__ == "__main__":
    main()
