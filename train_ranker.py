import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from model import EPCFQA
from scipy.stats import spearmanr, kendalltau
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

logger = logging.getLogger('Ranker-Training')

class RankListDataset(Dataset):
    """Listwise ranking dataset, each sample contains n_rank images"""
    def __init__(self, image_dir, list_file, n_rank=10, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.n_rank = n_rank
        with open(list_file, 'r') as f:
            self.image_lists = [line.strip() for line in f if line.strip()]
        assert len(self.image_lists) % n_rank == 0, f"Number of lines in list file ({len(self.image_lists)}) is not a multiple of n_rank ({n_rank})"
        self.n_groups = len(self.image_lists) // n_rank

    def __len__(self):
        return self.n_groups

    def __getitem__(self, idx):
        names = self.image_lists[idx * self.n_rank: (idx + 1) * self.n_rank]
        images = []
        for img_name in names:
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images, dim=0)
        return {
            'images': images,
            'names': names
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_scores(metrics_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(metrics_history.keys())
    srcc_values = [metrics_history[e]['srcc'] for e in epochs]
    krcc_values = [metrics_history[e]['krcc'] for e in epochs]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, srcc_values, 'r-', label='SRCC')
    plt.plot(epochs, krcc_values, 'g-', label='KRCC')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    plt.title(f'SRCC and KRCC vs. Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'srcc_krcc_curve.png'))
    plt.close()

def compute_ranking_metrics(pred_scores, n_rank):
    """Given [n_sample*n_rank] predictions, compute SRCC/KRCC for each group compared to reference"""
    pred_scores = pred_scores.reshape(-1, n_rank)
    ref = np.arange(n_rank, 0, -1)
    srccs, krccs = [], []
    for row in pred_scores:
        s = spearmanr(row, ref)[0]
        k = kendalltau(row, ref)[0]
        srccs.append(s)
        krccs.append(k)
    return np.mean(srccs), np.mean(krccs)

def spearmanr_torch(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    pred_rank = pred.argsort().argsort().float()
    target_rank = target.argsort().argsort().float()
    n = pred.shape[0]
    num = 6 * ((pred_rank - target_rank)**2).sum()
    denom = n*(n**2 - 1)
    srcc = 1 - num/denom
    return srcc

def train_epoch_ranker(model, train_loader, optimizer, device, epoch, n_rank):
    model.train()
    all_srcc, all_krcc = [], []
    total_loss = 0.0
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for batch in train_progress:
        images = batch['images'].to(device)
        B, n_rank, C, H, W = images.shape
        images_2d = images.view(B * n_rank, C, H, W)
        optimizer.zero_grad()
        outputs = model(images_2d)
        pred_scores = outputs['score'].view(B, n_rank)
        loss = 0
        for group_pred in pred_scores:
            # gold ranking (n_rank, n_rank-1, ..., 1)
            ref = torch.arange(n_rank, 0, -1, dtype=group_pred.dtype, device=group_pred.device)
            group_loss = 0.
            # Pairwise within group, higher score should be ranked higher
            for i in range(n_rank):
                for j in range(i+1, n_rank):
                    label = 1. if ref[i] > ref[j] else -1.
                    group_loss += F.margin_ranking_loss(
                        group_pred[i].unsqueeze(0), group_pred[j].unsqueeze(0),
                        torch.tensor([label], device=group_pred.device),
                        margin=0.5
                    )
            group_loss /= (n_rank * (n_rank-1) / 2)
            loss += group_loss
        loss = loss / B
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_scores_np = pred_scores.detach().cpu().numpy()
        mean_srcc, mean_krcc = compute_ranking_metrics(pred_scores_np, n_rank)
        all_srcc.append(mean_srcc)
        all_krcc.append(mean_krcc)
        train_progress.set_postfix({'loss': loss.item(), 'SRCC': mean_srcc, 'KRCC': mean_krcc})
    avg_loss = total_loss / len(train_loader)
    avg_srcc = np.mean(all_srcc)
    avg_krcc = np.mean(all_krcc)
    return avg_loss, {'srcc': avg_srcc, 'krcc': avg_krcc}

def validate_ranker(model, val_loader, device, epoch, n_rank):
    model.eval()
    all_srcc, all_krcc = [], []
    with torch.no_grad():
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in val_progress:
            images = batch['images'].to(device)
            B, n_rank, C, H, W = images.shape
            images_2d = images.view(B * n_rank, C, H, W)
            outputs = model(images_2d)
            pred_scores = outputs['score'].view(B, n_rank)
            pred_scores_np = pred_scores.detach().cpu().numpy()
            mean_srcc, mean_krcc = compute_ranking_metrics(pred_scores_np, n_rank)
            all_srcc.append(mean_srcc)
            all_krcc.append(mean_krcc)
            val_progress.set_postfix({'SRCC': mean_srcc, 'KRCC': mean_krcc})
    avg_srcc = np.mean(all_srcc)
    avg_krcc = np.mean(all_krcc)
    return {'srcc': avg_srcc, 'krcc': avg_krcc}

def load_pretrained_epcfqa(model, pretrained_path, logger=None):
    if not os.path.exists(pretrained_path): return
    state_dict = torch.load(pretrained_path, map_location='cpu')
    # Support both checkpoint and pure state_dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and (k.startswith('rppem.') or not k.startswith('rppem.'))}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    if logger:
        logger.info(f"Loaded pretrained parameters: {pretrained_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint['metrics']
    return model, optimizer, scheduler, start_epoch, metrics

def main():
    parser = argparse.ArgumentParser(description='Train Listwise Ranking Model with Ablation Support')
    parser.add_argument('--image_dir', type=str, required=True, help='Image folder')
    parser.add_argument('--train_list', type=str, required=True, help='Training list file')
    parser.add_argument('--val_list', type=str, required=True, help='Validation list file')
    parser.add_argument('--output_dir', type=str, default='./ranker_output', help='Output directory')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--n_rank', type=int, default=10, help='Number of images per group')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=4)
    # Ablation parameters
    parser.add_argument('--use_beta_D',      action='store_true',  default=True)
    parser.add_argument('--no_beta_D',       action='store_false', dest='use_beta_D')
    parser.add_argument('--use_beta_B',      action='store_true',  default=True)
    parser.add_argument('--no_beta_B',       action='store_false', dest='use_beta_B')
    parser.add_argument('--use_g',           action='store_true',  default=True)
    parser.add_argument('--no_g',            action='store_false', dest='use_g')
    parser.add_argument('--use_L',           action='store_true',  default=True)
    parser.add_argument('--no_L',            action='store_false', dest='use_L')
    parser.add_argument('--use_A_CDF',       action='store_true',  default=False)
    parser.add_argument('--no_A_CDF',        action='store_false', dest='use_A_CDF')
    parser.add_argument('--use_L_DDF',       action='store_true',  default=False)
    parser.add_argument('--no_L_DDF',        action='store_false', dest='use_L_DDF')
    parser.add_argument('--dataset_type', type=str, default='dataset1', help='Dataset type (reserved for model interface compatibility)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = RankListDataset(args.image_dir, args.train_list, n_rank=args.n_rank, transform=transform)
    val_dataset = RankListDataset(args.image_dir, args.val_list, n_rank=args.n_rank, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ablation_config = {
        'use_beta_D': args.use_beta_D,
        'use_beta_B': args.use_beta_B,
        'use_g': args.use_g,
        'use_L': args.use_L,
        'use_A_CDF': args.use_A_CDF,
        'use_L_DDF': args.use_L_DDF
    }
    logger.info(f"Ablation config: {ablation_config}")

    model = EPCFQA(image_size=256, ablation_config=ablation_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    if args.pretrained and os.path.exists(args.pretrained):
        load_pretrained_epcfqa(model, args.pretrained, logger)

    metrics_history = {}
    best_srcc = -1
    best_krcc = -1
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss, train_metrics = train_epoch_ranker(model, train_loader, optimizer, device, epoch, args.n_rank)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Train SRCC: {train_metrics['srcc']:.4f} | Train KRCC: {train_metrics['krcc']:.4f}")

        # Validate one epoch
        val_metrics = validate_ranker(model, val_loader, device, epoch, args.n_rank)
        metrics_history[epoch+1] = val_metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Val SRCC: {val_metrics['srcc']:.4f} | Val KRCC: {val_metrics['krcc']:.4f}")

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        logger.info(f"Learning Rate: {current_lr:.8f} -> {optimizer.param_groups[0]['lr']:.8f}")

        # Save best checkpoint
        if val_metrics['srcc'] > best_srcc:
            best_srcc = val_metrics['srcc']
            best_krcc = val_metrics['krcc']
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'srcc': best_srcc,
                'krcc': best_krcc,
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"Saved best model: SRCC={best_srcc:.4f}, KRCC={best_krcc:.4f}")
        plot_scores(metrics_history, args.output_dir)
    logger.info(f"Training finished! Best SRCC={best_srcc:.4f}, Best KRCC={best_krcc:.4f}")

if __name__ == "__main__":
    main()
