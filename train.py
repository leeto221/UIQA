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
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import sys
from datetime import datetime


# Configure logger
logger = logging.getLogger('EPCFQA-Training')

class UnderwaterQualityDataset(Dataset):
    """Underwater image quality assessment dataset"""
    def __init__(self, image_dir, image_files, scores, dataset_type, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.scores = scores
        self.dataset_type = dataset_type
        self.transform = transform
        
        logger.info(f"Loaded {len(self.image_files)} valid images and their MOS scores, dataset type: {dataset_type}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image filename and score
        img_name = self.image_files[idx]
        score = self.scores[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Convert score to tensor
        score_tensor = torch.tensor([score], dtype=torch.float32)
        
        return {
            'image': image,
            'score': score_tensor,
            'name': img_name,
            'dataset_type': self.dataset_type
        }


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, is_ddp=False):
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
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path, is_ddp=False):

    checkpoint = torch.load(path, map_location='cpu')
    
    if is_ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    metrics = checkpoint['metrics']
    return model, optimizer, scheduler, start_epoch, metrics

def calculate_metrics(pred_scores, gt_scores):
    pred_scores = pred_scores.squeeze().cpu().numpy()
    gt_scores = gt_scores.squeeze().cpu().numpy()
    plcc, _ = pearsonr(pred_scores, gt_scores)
    srcc, _ = spearmanr(pred_scores, gt_scores)
    krcc, _ = kendalltau(pred_scores, gt_scores)
    rmse = np.sqrt(np.mean((pred_scores - gt_scores) ** 2))
    
    return {
        'plcc': plcc,
        'srcc': srcc,
        'krcc': krcc,
        'rmse': rmse
    }

def plot_losses_and_metrics(train_losses, val_losses, metrics_history, save_dir):
    """Plot loss curves and evaluation metric curves"""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()
    
    # Extract metric history
    epochs = list(metrics_history.keys())
    plcc_values = [metrics_history[e]['plcc'] for e in epochs]
    srcc_values = [metrics_history[e]['srcc'] for e in epochs]
    krcc_values = [metrics_history[e]['krcc'] for e in epochs]
    rmse_values = [metrics_history[e]['rmse'] for e in epochs]
    
    # Plot PLCC and SRCC curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, plcc_values, 'b-', label='PLCC')
    plt.plot(epochs, srcc_values, 'r-', label='SRCC')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    plt.title('PLCC and SRCC vs. Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'plcc_srcc.png'))
    plt.close()
    
    # Plot KRCC curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, krcc_values, 'g-', label='KRCC')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    plt.title('KRCC vs. Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'krcc.png'))
    plt.close()
    
    # Plot RMSE curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmse_values, 'm-', label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE vs. Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'rmse.png'))
    plt.close()

def train_epoch(model, train_loader, optimizer, device, epoch, dataset_type):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    all_pred_scores = []
    all_gt_scores = []
    
    # Progress bar
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in train_progress:
        # Move data to device
        images = batch['image'].to(device)
        scores = batch['score'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images, dataset_type=dataset_type)
        pred_scores = outputs['score']
        
        # Loss (Huber loss)
        loss = F.huber_loss(pred_scores, scores, delta=0.5)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Update progress and loss
        batch_loss = loss.item()
        train_loss += batch_loss
        train_progress.set_postfix({'loss': batch_loss})
        
        # Collect predictions and ground truth for metrics
        all_pred_scores.append(pred_scores.detach())
        all_gt_scores.append(scores)
    
    # Average training loss
    train_loss /= len(train_loader)
    
    # Concatenate all predictions and ground truth
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_gt_scores = torch.cat(all_gt_scores, dim=0)
    
    # Calculate metrics on training set
    metrics = calculate_metrics(all_pred_scores, all_gt_scores)
    
    return train_loss, metrics

def validate(model, val_loader, device, epoch, dataset_type):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    all_pred_scores = []
    all_gt_scores = []
    
    # Progress bar
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
    
    with torch.no_grad():
        for batch in val_progress:
            # Move data to device
            images = batch['image'].to(device)
            scores = batch['score'].to(device)
            
            # Forward
            outputs = model(images, dataset_type=dataset_type)
            pred_scores = outputs['score']
            
            # Loss (Huber loss)
            loss = F.huber_loss(pred_scores, scores, delta=0.5)

            
            # Update loss
            batch_loss = loss.item()
            val_loss += batch_loss
            val_progress.set_postfix({'loss': batch_loss})
            
            # Collect predictions and ground truth for metrics
            all_pred_scores.append(pred_scores)
            all_gt_scores.append(scores)
    
    # Average validation loss
    val_loss /= len(val_loader)
    
    # Concatenate all predictions and ground truth
    all_pred_scores = torch.cat(all_pred_scores, dim=0)
    all_gt_scores = torch.cat(all_gt_scores, dim=0)
    
    # Calculate metrics on validation set
    metrics = calculate_metrics(all_pred_scores, all_gt_scores)
    
    return val_loss, metrics

def main(run_idx=None, base_seed=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train E-PCFQA model end-to-end')
    # Data paths
    parser.add_argument('--image_dir', type=str, 
                        default=None, 
                        help='Image directory')
    parser.add_argument('--mos_file', type=str, 
                        default=None, 
                        help='MOS file')
    parser.add_argument('--output_dir', type=str, 
                        default=None, 
                        help='Output directory')
    

    parser.add_argument('--pretrained', type=str, 
                        default='prepth/best_model.pth', 
                        help='Pretrained model path')

    parser.add_argument('--test_file', type=str, 
                        default=None, 
                        help='File containing validation image names')
    
    # Add random split option
    parser.add_argument('--random_split', action='store_true', 
                        help='Use random split for train/val set instead of fixed test file')
    
    # Add repeat experiment parameter
    parser.add_argument('--repeat', type=int, default=1, 
                        help='Number of repeated experiments (with different random seeds)')

    # Add ablation experiment parameters
    # ... [Ablation parameters unchanged] ...
    parser.add_argument('--use_beta_D', action='store_true', default=True, 
                        help='Use beta_D feature')
    parser.add_argument('--no_beta_D', action='store_false', dest='use_beta_D',
                        help='Do not use beta_D feature')
    parser.add_argument('--use_beta_B', action='store_true', default=True,
                        help='Use beta_B feature')
    parser.add_argument('--no_beta_B', action='store_false', dest='use_beta_B',
                        help='Do not use beta_B feature')
    parser.add_argument('--use_g', action='store_true', default=True,
                        help='Use g feature')
    parser.add_argument('--no_g', action='store_false', dest='use_g',
                        help='Do not use g feature')
    parser.add_argument('--use_L', action='store_true', default=True,
                        help='Use L feature')
    parser.add_argument('--no_L', action='store_false', dest='use_L',
                        help='Do not use L feature')
    parser.add_argument('--use_A_CDF', action='store_true', default=False,
                        help='Use A-CDF feature')
    parser.add_argument('--no_A_CDF', action='store_false', dest='use_A_CDF',
                        help='Do not use A-CDF feature')
    parser.add_argument('--use_L_DDF', action='store_true', default=False,
                        help='Use L-DDF feature')
    parser.add_argument('--no_L_DDF', action='store_false', dest='use_L_DDF',
                        help='Do not use L-DDF feature')
    
    # Dataset parameters
    parser.add_argument('--dataset_type', type=str, default='dataset1', help='Dataset type identifier')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--seed', type=int, default=43, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume training')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (epochs)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')

    # GPU parameters
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID to use')
    
    args = parser.parse_args()
    
    # If this is a repeated run, use run_idx and corresponding seed
    if run_idx is not None:
        # Create subdirectory for each run
        args.output_dir = os.path.join(args.output_dir, str(run_idx))
        # Adjust seed
        if base_seed is not None:
            args.seed = base_seed + run_idx
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logger
    log_file = os.path.join(args.output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log repeat experiment info
    if run_idx is not None:
        logger.info(f"Current repeat experiment: {run_idx+1} / {args.repeat}")
    
    # Load MOS data
    image_files = []
    scores = []
    
    with open(args.mos_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    score = float(parts[1].strip())
                    
                    # Check if image exists
                    img_path = os.path.join(args.image_dir, img_name)
                    if os.path.exists(img_path):
                        image_files.append(img_name)
                        scores.append(score)
    
    logger.info(f"Loaded {len(image_files)} valid images from MOS file")

    # Determine train and validation sets
    if args.random_split or not os.path.exists(args.test_file):
        # Use random split
        logger.info(f"Using random split for train/val set (val ratio: {args.val_ratio})")
        train_files, val_files, train_scores, val_scores = train_test_split(
            image_files, scores, test_size=args.val_ratio, random_state=args.seed
        )
        logger.info(f"Random split: train set: {len(train_files)} images, val set: {len(val_files)} images")
    else:
        # Use specified test file
        logger.info(f"Using specified test file: {args.test_file}")
        val_files = []
        
        # Read test file
        with open(args.test_file, 'r') as f:
            for line in f:
                img_name = line.strip()
                if img_name in image_files:  # Ensure image is in valid list
                    val_files.append(img_name)
        
        logger.info(f"Loaded {len(val_files)} validation images from {args.test_file}")
        
        # Remove validation images from all images, rest are train set
        train_files = [img for img in image_files if img not in val_files]
        
        # Get scores for train and val sets
        train_scores = [scores[image_files.index(img)] for img in train_files]
        val_scores = [scores[image_files.index(img)] for img in val_files]
        
        logger.info(f"Using specified split: train set {len(train_files)} images, val set {len(val_files)} images")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = UnderwaterQualityDataset(
        image_dir=args.image_dir,
        image_files=train_files,
        scores=train_scores,
        dataset_type=args.dataset_type,
        transform=transform
    )
    
    val_dataset = UnderwaterQualityDataset(
        image_dir=args.image_dir,
        image_files=val_files,
        scores=val_scores,
        dataset_type=args.dataset_type,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create ablation config
    ablation_config = {
        'use_beta_D': args.use_beta_D,
        'use_beta_B': args.use_beta_B,
        'use_g': args.use_g,
        'use_L': args.use_L,
        'use_A_CDF': args.use_A_CDF,
        'use_L_DDF': args.use_L_DDF
    }
    
    # Create model and pass ablation config
    model = EPCFQA(image_size=256, ablation_config=ablation_config).to(device)
    
    # Log ablation config
    logger.info(f"Ablation config: {ablation_config}")
    
    # Load pretrained weights (if any)
    if args.pretrained and os.path.exists(args.pretrained):
        logger.info(f"Loading pretrained weights: {args.pretrained}")
        pretrained_dict = torch.load(args.pretrained, map_location='cpu', weights_only=True)
        
        # If checkpoint format
        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']
            
        # Load R-PPEM module weights
        model_dict = model.state_dict()
        # Filter out unmatched keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.startswith('rppem.')}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Successfully loaded pretrained R-PPEM module weights")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Initialize variables
    start_epoch = 0
    max_plcc = float('-inf')  # Initialize as negative infinity
    best_metrics = {
        'plcc': -1.0,
        'srcc': -1.0,
        'krcc': -1.0,
        'rmse': float('inf')
    }
    train_losses = []
    val_losses = []
    metrics_history = {}
    
    # If resume checkpoint provided
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model, optimizer, scheduler, start_epoch, best_metrics = load_checkpoint(
            model, optimizer, scheduler, args.resume
        )
        logger.info(f"Start epoch: {start_epoch}, best metrics: PLCC={best_metrics['plcc']:.4f}, SRCC={best_metrics['srcc']:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, 
            dataset_type=args.dataset_type
        )
        train_losses.append(train_loss)
        
        # Log training info
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {train_loss:.6f}")
        logger.info(f"Train Metrics: PLCC={train_metrics['plcc']:.4f}, SRCC={train_metrics['srcc']:.4f}, KRCC={train_metrics['krcc']:.4f}, RMSE={train_metrics['rmse']:.4f}")
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        logger.info(f"Learning Rate: {current_lr:.8f} -> {optimizer.param_groups[0]['lr']:.8f}")
        
        # Validation phase (every val_interval epochs)
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_loss, val_metrics = validate(
                model, val_loader, device, epoch,
                dataset_type=args.dataset_type
            )
            val_losses.append(val_loss)
            metrics_history[epoch+1] = val_metrics
            
            # Log validation info
            logger.info(f"Val Loss: {val_loss:.6f}")
            logger.info(f"Val Metrics: PLCC={val_metrics['plcc']:.4f}, SRCC={val_metrics['srcc']:.4f}, KRCC={val_metrics['krcc']:.4f}, RMSE={val_metrics['rmse']:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)
            
            # Update best model (based on PLCC)
            if val_metrics['plcc'] > max_plcc:
                max_plcc = val_metrics['plcc']
                max_srcc = val_metrics['srcc']
                max_krcc = val_metrics['krcc']
                min_rmse = val_metrics['rmse']
                
                # Save best model
                best_model_path = os.path.join(args.output_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
                logger.info(f"Saved best model, PLCC: {max_plcc:.4f}, SRCC: {max_srcc:.4f}, KRCC: {max_krcc:.4f}, RMSE: {min_rmse:.4f}")
            
            # Output current best metrics
            logger.info(f"Current best metrics: PLCC={max_plcc:.4f}, SRCC={max_srcc:.4f}, KRCC={max_krcc:.4f}, RMSE={min_rmse:.4f}")
            
            # Plot loss and metric curves
            plot_losses_and_metrics(train_losses, val_losses, metrics_history, args.output_dir)
    
    # Training finished, log final results
    logger.info("Training finished!")
    logger.info(f"Final best metrics: PLCC={max_plcc:.4f}, SRCC={max_srcc:.4f}, KRCC={max_krcc:.4f}, RMSE={min_rmse:.4f}")
    
    # Return best metrics (for repeat experiment summary)
    return {
        'plcc': max_plcc,
        'srcc': max_srcc,
        'krcc': max_krcc,
        'rmse': min_rmse
    }

def run_multiple_experiments():
    """Run multiple repeated experiments and summarize results"""
    parser = argparse.ArgumentParser(description='Run multiple E-PCFQA model experiments')
    parser.add_argument('--repeat', type=int, default=10, help='Number of repeated experiments')
    parser.add_argument('--output_dir', type=str, 
                       default=None, 
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    # Add other needed parameters if necessary
    
    args, unknown_args = parser.parse_known_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure summary logger
    summary_log_file = os.path.join(args.output_dir, 'experiments_summary.log')
    summary_logger = logging.getLogger('Experiments-Summary')
    handler = logging.FileHandler(summary_log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    summary_logger.addHandler(handler)
    summary_logger.setLevel(logging.INFO)
    
    # Record start time
    start_time = datetime.now()
    summary_logger.info(f"Start running {args.repeat} repeated experiments, base seed: {args.seed}")
    summary_logger.info(f"Command line args: {sys.argv}")
    
    # Collect results
    results = []
    
    # Run multiple experiments
    for i in range(args.repeat):
        summary_logger.info(f"Start experiment {i+1}/{args.repeat}")
        try:
            # Run single experiment
            metrics = main(run_idx=i, base_seed=args.seed)
            results.append(metrics)
            summary_logger.info(f"Experiment {i+1} finished: PLCC={metrics['plcc']:.4f}, SRCC={metrics['srcc']:.4f}, KRCC={metrics['krcc']:.4f}, RMSE={metrics['rmse']:.4f}")
        except Exception as e:
            summary_logger.error(f"Experiment {i+1} failed: {str(e)}")
    
    # Calculate mean and std
    if results:
        plcc_values = [r['plcc'] for r in results]
        srcc_values = [r['srcc'] for r in results]
        krcc_values = [r['krcc'] for r in results]
        rmse_values = [r['rmse'] for r in results]
        
        plcc_mean, plcc_std = np.mean(plcc_values), np.std(plcc_values)
        srcc_mean, srcc_std = np.mean(srcc_values), np.std(srcc_values)
        krcc_mean, krcc_std = np.mean(krcc_values), np.std(krcc_values)
        rmse_mean, rmse_std = np.mean(rmse_values), np.std(rmse_values)
        
        # Summarize results
        summary_logger.info(f"======== Experiment Results Summary ({len(results)}/{args.repeat} succeeded) ========")
        summary_logger.info(f"PLCC: {plcc_mean:.4f} ± {plcc_std:.4f}")
        summary_logger.info(f"SRCC: {srcc_mean:.4f} ± {srcc_std:.4f}")
        summary_logger.info(f"KRCC: {krcc_mean:.4f} ± {krcc_std:.4f}")
        summary_logger.info(f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
        
        # Save detailed results
        with open(os.path.join(args.output_dir, 'detailed_results.txt'), 'w') as f:
            f.write("Run\tPLCC\tSRCC\tKRCC\tRMSE\n")
            for i, r in enumerate(results):
                f.write(f"{i}\t{r['plcc']:.6f}\t{r['srcc']:.6f}\t{r['krcc']:.6f}\t{r['rmse']:.6f}\n")
            
            f.write("\nSummary:\n")
            f.write(f"PLCC: {plcc_mean:.6f} ± {plcc_std:.6f}\n")
            f.write(f"SRCC: {srcc_mean:.6f} ± {srcc_std:.6f}\n")
            f.write(f"KRCC: {krcc_mean:.6f} ± {krcc_std:.6f}\n")
            f.write(f"RMSE: {rmse_mean:.6f} ± {rmse_std:.6f}\n")
    else:
        summary_logger.warning("All experiments failed, cannot generate summary statistics")
    
    # Record end time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    summary_logger.info(f"All experiments finished, total time: {elapsed_time}")

if __name__ == "__main__":
    # Check if running repeated experiments
    if '--repeat' in sys.argv and len(sys.argv) > sys.argv.index('--repeat') + 1:
        repeat_idx = sys.argv.index('--repeat')
        try:
            repeat_count = int(sys.argv[repeat_idx + 1])
            if repeat_count > 1:
                run_multiple_experiments()
                exit(0)
        except:
            pass
        
    # Default: run single experiment
    main()