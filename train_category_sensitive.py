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
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy import stats
from collections import defaultdict

# 引入你的双流模型
from model_dual import PhysicsGuidedIQA

# -----------------------------------------------------------------------------
# 0. 类别定义与解析工具
# -----------------------------------------------------------------------------

# 按照你的描述，精确定义10个类别
# 注意：fusion18 和 fusion12 只要全名匹配即可，顺序其实不影响，但为了严谨我们列出全名
CATEGORIES = [
    'waternet', 'funiegan', 'fusion18', 'dive', 'fusion12',
    'raw', 'HP', 'IBLA', 'twostep', 'ACPAB'
]

def get_category_idx(filename):
    """
    根据文件名判断属于哪个类别，返回索引 (0-9)。
    例如: fusion181.bmp -> startswith('fusion18') -> 返回对应的 index
    """
    for idx, cat_name in enumerate(CATEGORIES):
        if filename.startswith(cat_name):
            return idx
    return -1  # 未知类别

# -----------------------------------------------------------------------------
# 1. Dataset (支持传入预处理好的列表)
# -----------------------------------------------------------------------------

class CategoryIQADataset(Dataset):
    def __init__(self, image_dir, data_list, transform=None):
        """
        data_list: list of tuples (filename, score, cat_idx)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, score, cat_idx = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            I = Image.open(img_path).convert('RGB')
        except Exception as e:
            # print(f"Error loading {img_path}: {e}")
            I = Image.new('RGB', (256, 256))

        if self.transform:
            I = self.transform(I)

        # 返回：图像, 分数, 类别索引
        return I, torch.tensor(score, dtype=torch.float32), torch.tensor(cat_idx, dtype=torch.long)

# -----------------------------------------------------------------------------
# 2. 数据准备与分层划分 (Stratified Split)
# -----------------------------------------------------------------------------

def prepare_stratified_data(label_file, seed):
    """
    读取 txt，按类别分组，每组随机 shuffle 后按 1600:400 划分。
    """
    # 1. 读取并解析所有数据
    data_by_class = defaultdict(list)
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                score = float(parts[1])
                
                cat_idx = get_category_idx(name)
                if cat_idx != -1:
                    data_by_class[cat_idx].append((name, score, cat_idx))
                else:
                    print(f"[Warning] Unknown category for file: {name}")

    # 2. 分层划分
    train_data_all = []
    val_data_all = []
    
    rng = random.Random(seed)
    
    print("\nData Split Statistics:")
    print(f"{'Category':<12} | {'Found':<6} | {'Train':<6} | {'Val':<6}")
    print("-" * 40)

    for idx, cat_name in enumerate(CATEGORIES):
        items = data_by_class[idx]
        count = len(items)
        
        # Shuffle
        rng.shuffle(items)
        
        # Hard split: 1600 for train, rest (expected 400) for val
        # 为了鲁棒性，万一数量不足 2000，按比例或截断，这里按你要求的固定数量
        train_subset = items[:1600]
        val_subset = items[1600:]
        
        train_data_all.extend(train_subset)
        val_data_all.extend(val_subset)
        
        print(f"{cat_name:<12} | {count:<6} | {len(train_subset):<6} | {len(val_subset):<6}")
    print("-" * 40)
    print(f"{'TOTAL':<12} | {sum(len(v) for v in data_by_class.values()):<6} | {len(train_data_all):<6} | {len(val_data_all):<6}\n")
    
    return train_data_all, val_data_all

# -----------------------------------------------------------------------------
# 3. Training & Detailed Validation
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

def train_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train() 
    train_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
    
    for imgs, scores, _ in pbar: # 训练时不需要类别信息
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

def validate_per_class(model, loader, device):
    """
    验证函数：按类别分别计算指标
    """
    model.eval()
    
    # 存储结果: results[cat_idx] = {'preds': [], 'targets': []}
    results = defaultdict(lambda: {'preds': [], 'targets': []})
    
    with torch.no_grad():
        for imgs, scores, cat_idxs in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            
            preds_np = preds.cpu().numpy().flatten()
            scores_np = scores.numpy().flatten()
            cats_np = cat_idxs.numpy().flatten()
            
            # 将每个样本分发到对应的类别列表中
            for i in range(len(cats_np)):
                c = cats_np[i]
                results[c]['preds'].append(preds_np[i])
                results[c]['targets'].append(scores_np[i])

    # 计算指标
    metrics_per_class = {}
    sum_srcc = 0.0
    sum_plcc = 0.0
    
    # 遍历所有定义的类别 (0-9)
    for idx, cat_name in enumerate(CATEGORIES):
        data = results[idx]
        if len(data['preds']) < 2:
            # 防止某类样本太少无法计算
            s, p = 0.0, 0.0
        else:
            s, _ = stats.spearmanr(data['preds'], data['targets'])
            p, _ = stats.pearsonr(data['preds'], data['targets'])
        
        metrics_per_class[cat_name] = {'srcc': s, 'plcc': p}
        sum_srcc += s
        sum_plcc += p
        
    avg_srcc = sum_srcc / len(CATEGORIES)
    avg_plcc = sum_plcc / len(CATEGORIES)
    
    return avg_srcc, avg_plcc, metrics_per_class

# -----------------------------------------------------------------------------
# 4. Session Runner
# -----------------------------------------------------------------------------

def run_session(gpu_id, args, current_seed, result_queue):
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    set_seed(current_seed)
    
    print(f"\n>>> Starting Session with Seed: {current_seed}")

    # 1. 准备数据 (手动分层划分)
    train_list, val_list = prepare_stratified_data(args.label_file, current_seed)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CategoryIQADataset(args.image_dir, train_list, transform=transform)
    val_dataset = CategoryIQADataset(args.image_dir, val_list, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # 2. 模型
    print(f"Loading Pretrained R-PPEM from: {args.rppem_path}")
    model = PhysicsGuidedIQA(pretrained_rppem_path=args.rppem_path).to(device)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. 日志 (CSV格式，方便查看)
    log_file = os.path.join(args.output_dir, f"log_seed_{current_seed}.csv")
    with open(log_file, "w") as f:
        # Header: Epoch, Loss, Avg_SRCC, Avg_PLCC, cat1_S, cat1_P, cat2_S, cat2_P ...
        header = "Epoch,Train_Loss,Avg_SRCC,Avg_PLCC," + ",".join([f"{c}_S,{c}_P" for c in CATEGORIES])
        f.write(header + "\n")

    # 4. 训练循环
    best_avg_srcc = -1.0
    best_metrics_detail = {} # 存储最佳epoch的所有详细指标
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        
        # 详细验证
        val_avg_srcc, val_avg_plcc, detail_metrics = validate_per_class(model, val_loader, device)
        
        scheduler.step()

        # 构建日志行
        log_line = f"{epoch+1},{train_loss:.6f},{val_avg_srcc:.6f},{val_avg_plcc:.6f}"
        for cat in CATEGORIES:
            m = detail_metrics[cat]
            log_line += f",{m['srcc']:.4f},{m['plcc']:.4f}"
            
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

        # 保存最佳 (以 SRCC 之和/均值 为标准)
        if val_avg_srcc > best_avg_srcc:
            best_avg_srcc = val_avg_srcc
            best_metrics_detail = detail_metrics
            best_epoch = epoch + 1
            
            save_path = os.path.join(args.output_dir, f"best_model_seed_{current_seed}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'avg_srcc': val_avg_srcc,
                'detail_metrics': detail_metrics
            }, save_path)
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | Avg SRCC={val_avg_srcc:.4f}")

    print(f"<<< Finished Seed {current_seed}. Best Avg SRCC: {best_avg_srcc:.4f} @ Epoch {best_epoch}")
    
    # 结果回传
    result_queue.put((current_seed, best_avg_srcc, best_metrics_detail))

# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rppem_path', type=str, required=True, help='Pretrained R-PPEM weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder')
    parser.add_argument('--label_file', type=str, required=True, help='Dataset txt file')
    parser.add_argument('--output_dir', type=str, default='./output_category')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_seed', type=int, default=43)
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

    # 最终汇总
    if results:
        # results format: [(seed, best_avg_srcc, detail_dict), ...]
        
        print("\n=================================================================================")
        print("FINAL CATEGORY SENSITIVITY RESULTS (Average of 3 runs)")
        print("=================================================================================")
        
        # 1. 打印每次的总分
        print("Per-seed Best Average SRCC:")
        avg_srccs = []
        for r in results:
            print(f"  Seed {r[0]}: {r[1]:.4f}")
            avg_srccs.append(r[1])
        print(f"  --> Global Average: {np.mean(avg_srccs):.4f} +/- {np.std(avg_srccs):.4f}")
        print("-" * 80)
        
        # 2. 统计每一类的 SRCC 和 PLCC 均值
        # 结构初始化
        final_stats = {cat: {'srcc': [], 'plcc': []} for cat in CATEGORIES}
        
        for r in results:
            details = r[2] # detail_metrics dict
            for cat in CATEGORIES:
                final_stats[cat]['srcc'].append(details[cat]['srcc'])
                final_stats[cat]['plcc'].append(details[cat]['plcc'])
        
        print(f"{'Category':<12} | {'SRCC (Mean)':<12} | {'SRCC (Std)':<12} | {'PLCC (Mean)':<12} | {'PLCC (Std)':<12}")
        print("-" * 80)
        
        # 写入汇总文件
        summary_path = os.path.join(args.output_dir, "final_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Global Best Epoch Average SRCC: {np.mean(avg_srccs):.4f} +/- {np.std(avg_srccs):.4f}\n\n")
            f.write(f"{'Category':<12} | {'SRCC_Mean':<10} | {'SRCC_Std':<10} | {'PLCC_Mean':<10} | {'PLCC_Std':<10}\n")
            f.write("-" * 65 + "\n")
            
            for cat in CATEGORIES:
                s_mean = np.mean(final_stats[cat]['srcc'])
                s_std = np.std(final_stats[cat]['srcc'])
                p_mean = np.mean(final_stats[cat]['plcc'])
                p_std = np.std(final_stats[cat]['plcc'])
                
                # 打印到控制台
                print(f"{cat:<12} | {s_mean:<12.4f} | {s_std:<12.4f} | {p_mean:<12.4f} | {p_std:<12.4f}")
                # 写入文件
                f.write(f"{cat:<12} | {s_mean:<10.4f} | {s_std:<10.4f} | {p_mean:<10.4f} | {p_std:<10.4f}\n")

if __name__ == "__main__":
    main()
