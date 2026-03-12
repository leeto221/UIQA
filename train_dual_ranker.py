import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import random
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
import logging
from collections import defaultdict

# 引入你的双流模型
from model_dual import PhysicsGuidedIQA

# -----------------------------------------------------------------------------
# Dataset: 修正后的自动扫描逻辑
# -----------------------------------------------------------------------------

class AutoRankDataset(Dataset):
    def __init__(self, image_dir, n_rank=10, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.n_rank = n_rank
        
        self.groups = self._scan_and_group(image_dir)
        
        if len(self.groups) == 0:
            raise RuntimeError(f"No valid groups found in {image_dir} with n_rank={n_rank}!\n"
                               f"Please check the debug info printed above.")
            
        print(f"Dataset Loaded: Found {len(self.groups)} groups (each has {n_rank} images).")

    def _scan_and_group(self, root):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        all_files = sorted([f for f in os.listdir(root) if os.path.splitext(f)[1].lower() in valid_exts])
        
        # 字典结构: { '100': [(6, '100_GDCP_6.png'), (0, '100_dive+_0.png')...] }
        raw_groups = defaultdict(list)
        
        for fname in all_files:
            name_no_ext = os.path.splitext(fname)[0]
            
            # 必须包含至少一个下划线
            if '_' in name_no_ext:
                try:
                    # 修改 1: 获取 Group ID (第一个下划线之前的部分)
                    # 例如: "100_GDCP_6" -> group_id="100"
                    parts = name_no_ext.split('_')
                    group_id = parts[0] 
                    
                    # 修改 2: 获取 Rank ID (最后一个下划线之后的部分)
                    # 例如: "100_GDCP_6" -> rank_str="6"
                    rank_str = parts[-1]
                    
                    if rank_str.isdigit():
                        rank_val = int(rank_str)
                        raw_groups[group_id].append((rank_val, fname))
                except:
                    continue
        
        valid_groups = []
        for group_id, items in raw_groups.items():
            if len(items) == self.n_rank:
                # 排序: 按后缀数字从小到大排序
                items.sort(key=lambda x: x[0]) 
                sorted_filenames = [x[1] for x in items]
                valid_groups.append(sorted_filenames)
        
        # --- DEBUG ---
        if len(valid_groups) == 0:
            print("\n[DEBUG ERROR] No groups formed. Debugging info:")
            print(f"  - Logic: Group by First('_'), Sort by Last('_')")
            print(f"  - Total images found: {len(all_files)}")
            if len(all_files) > 0:
                print(f"  - Example file: {all_files[0]}")
            
            print(f"  - Raw groups identified: {len(raw_groups)}")
            count = 0
            for k, v in raw_groups.items():
                if count >= 3: break
                # 打印出来看看归类对不对
                print(f"  - Group ID '{k}' has {len(v)} images. Ranks found: {[x[0] for x in v]}")
                count += 1
            print("------------------------------------------------\n")
                
        return valid_groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        names = self.groups[idx]
        images = []
        for img_name in names:
            img_path = os.path.join(self.image_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                img = Image.new('RGB', (256, 256))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images, dim=0)
        return {'images': images}

# -----------------------------------------------------------------------------
# Utils & Training Logic (保持不变)
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_ranking_metrics(pred_scores, n_rank):
    pred_scores = pred_scores.reshape(-1, n_rank)
    ref = np.arange(n_rank, 0, -1)
    srccs, krccs = [], []
    for row in pred_scores:
        s = spearmanr(row, ref)[0]
        k = kendalltau(row, ref)[0]
        if np.isnan(s): s = 0.0
        if np.isnan(k): k = 0.0
        srccs.append(s)
        krccs.append(k)
    return np.mean(srccs), np.mean(krccs)

def train_epoch(model, loader, optimizer, device, epoch, n_rank):
    model.train() 
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch in pbar:
        images = batch['images'].to(device)
        B, _, C, H, W = images.shape
        images_flat = images.view(-1, C, H, W)
        
        optimizer.zero_grad()
        outputs = model(images_flat)
        pred_scores = outputs.view(B, n_rank)
        
        loss = 0.0
        for group_pred in pred_scores:
            ref = torch.arange(n_rank, 0, -1, device=device, dtype=torch.float32)
            group_loss = 0.0
            pair_count = 0
            for i in range(n_rank):
                for j in range(i + 1, n_rank):
                    label = 1.0 if ref[i] > ref[j] else -1.0
                    group_loss += F.margin_ranking_loss(
                        group_pred[i].unsqueeze(0), 
                        group_pred[j].unsqueeze(0),
                        torch.tensor([label], device=device),
                        margin=0.5
                    )
                    pair_count += 1
            if pair_count > 0:
                loss += group_loss / pair_count
        
        loss = loss / B
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

def validate(model, loader, device, n_rank):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            B, _, C, H, W = images.shape
            images_flat = images.view(-1, C, H, W)
            outputs = model(images_flat)
            pred_scores = outputs.view(B, n_rank)
            all_preds.append(pred_scores.cpu().numpy())
            
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        avg_srcc, avg_krcc = compute_ranking_metrics(all_preds, n_rank)
    else:
        avg_srcc, avg_krcc = 0.0, 0.0
    return avg_srcc, avg_krcc

def run_session(gpu_id, args, current_seed, result_queue):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    set_seed(current_seed)
    
    print(f"\n>>> Starting Session | Seed: {current_seed}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        full_dataset = AutoRankDataset(args.image_dir, n_rank=args.n_rank, transform=transform)
    except RuntimeError as e:
        print(e)
        return

    total_groups = len(full_dataset)
    indices = list(range(total_groups))
    random.Random(current_seed).shuffle(indices)
    
    split = int(np.floor(0.2 * total_groups))
    train_indices = indices[split:]
    val_indices = indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    print(f"Loading Pretrained R-PPEM: {args.rppem_path}")
    model = PhysicsGuidedIQA(pretrained_rppem_path=args.rppem_path).to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    log_file = os.path.join(args.output_dir, f"log_seed_{current_seed}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,Train_Loss,Val_SRCC,Val_KRCC\n")

    best_srcc = -1.0
    best_krcc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.n_rank)
        val_srcc, val_krcc = validate(model, val_loader, device, args.n_rank)
        scheduler.step()

        with open(log_file, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_srcc:.6f},{val_krcc:.6f}\n")

        if val_srcc > best_srcc:
            best_srcc = val_srcc
            best_krcc = val_krcc
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, f"best_ranker_seed_{current_seed}.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'srcc': best_srcc, 'krcc': best_krcc}, save_path)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f} | Val SRCC={val_srcc:.4f} KRCC={val_krcc:.4f}")

    print(f"<<< Finished Seed {current_seed}. Best SRCC: {best_srcc:.4f}")
    result_queue.put((current_seed, best_srcc, best_krcc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rppem_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ranker_auto_output')
    parser.add_argument('--n_rank', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--base_seed', type=int, default=44)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    results = []

    for i in range(args.repeats):
        current_seed = args.base_seed + i
        p = ctx.Process(target=run_session, args=(args.gpu_id, args, current_seed, result_queue))
        p.start()
        p.join()
        if not result_queue.empty():
            results.append(result_queue.get())
        else:
            print(f"Session {current_seed} failed.")

    if results:
        srccs = [r[1] for r in results]
        krccs = [r[2] for r in results]
        print("\n============================================")
        print("FINAL RESULTS SUMMARY")
        print("============================================")
        print(f"{'Seed':<6} | {'SRCC':<8} | {'KRCC':<8}")
        for r in results:
            print(f"{r[0]:<6} | {r[1]:.4f}   | {r[2]:.4f}")
        print("============================================")
        print(f"AVG SRCC: {np.mean(srccs):.4f}")
        print(f"AVG KRCC: {np.mean(krccs):.4f}")
        with open(os.path.join(args.output_dir, "summary_auto.txt"), "w") as f:
            f.write(f"Average SRCC: {np.mean(srccs):.4f} +/- {np.std(srccs):.4f}\n")
            f.write(f"Average KRCC: {np.mean(krccs):.4f} +/- {np.std(krccs):.4f}\n")

if __name__ == "__main__":
    main()
