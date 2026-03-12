import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import math
from PIL import Image
from torchvision import transforms

# =========================================================
# Part 1: Model Definition (From your previous code)
# =========================================================

class DoubleConv(nn.Module):
    """Double convolution block for feature extraction and upsampling"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class R_PPEM(nn.Module):
    """Robust Physical Parameter Estimation Module"""
    def __init__(self):
        super(R_PPEM, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the last fully connected layer and avgpool
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Shared encoder features
        self.shared_features = DoubleConv(512, 256)
        
        # Depth estimation path
        self.depth_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 2, 1)
        )
        # beta_D estimation path (channel-specific, output H×W×3)
        self.beta_D_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 3, 1)
        )
        # beta_B estimation path (global parameter, output 3)
        self.beta_B_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 128, 1), nn.ReLU(True), nn.Conv2d(128, 3, 1)
        )
        # g estimation path (spatially varying, output H×W×1)
        self.g_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1)
        )
        # Lighting estimation path L (spatially varying, output H×W×1)
        self.light_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1)
        )
        
        # Upsample to original resolution
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Shared features
        shared = self.shared_features(features)
        
        # Depth estimation
        depth_out = self.depth_path(shared)
        mu_d = torch.sigmoid(depth_out[:, 0:1]) * 12.0  # Depth range [0, 12]
        sigma_d = F.softplus(depth_out[:, 1:2]) + 1e-6  # Ensure variance is positive
        
        # Parameter estimation - with constraints
        # beta_D: log space mapping to [0.001, 4.0]
        log_min_d, log_max_d = math.log(0.001), math.log(4.0)
        beta_D = torch.exp(torch.sigmoid(self.beta_D_path(shared)) * (log_max_d - log_min_d) + log_min_d)

        # beta_B: log space mapping to [0.001, 5.0]
        log_min_b, log_max_b = math.log(0.001), math.log(5.0)
        beta_B = torch.exp(torch.sigmoid(self.beta_B_path(shared)) * (log_max_b - log_min_b) + log_min_b)

        g = torch.sigmoid(self.g_path(shared)) * 0.45 + 0.5  # Range [0.5, 0.95]
        L = torch.sigmoid(self.light_path(shared)) + 0.5  # Range [0.5, 1.5]
        
        # Uncertainty-weighted parameter adjustment
        uncertainty_weight = 1.0 / (1.0 + sigma_d)
        
        # Upsample before or after weighting? The original code weighted then upsampled for beta_D, g, L.
        # sigma_d was upsampled directly, mu_d was upsampled directly.
        # Let's align with the second version of R_PPEM forward pass for upsampling logic
        beta_D_up = self.upsample(beta_D * uncertainty_weight)
        g_up = self.upsample(g * uncertainty_weight)
        L_up = self.upsample(L * uncertainty_weight)
        mu_d_up = self.upsample(mu_d)
        sigma_d_up = self.upsample(sigma_d)
        
        # Note: beta_B is a global parameter, so it doesn't get upsampled. 
        # The original R_PPEM returned beta_B (global) and beta_B_broadcast (upsampled).
        # The new PhysicsGuidedIQA uses beta_B (global) for phys_globals.
        # So we return the global beta_B.
        
        return mu_d_up, sigma_d_up, beta_D_up, beta_B, g_up, L_up


class PhysicsAttention(nn.Module):
    def __init__(self, visual_channels=512, spatial_phys_channels=7, global_phys_channels=3):
        super(PhysicsAttention, self).__init__()
        
        # 1. Spatial Gate (High-Res)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(spatial_phys_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.InstanceNorm2d(1, affine=True), 
            nn.Sigmoid()
        )
        
        # 2. Channel Gate
        total_global_input = global_phys_channels + spatial_phys_channels
        self.channel_gate = nn.Sequential(
            nn.Linear(total_global_input, visual_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(visual_channels // 4, visual_channels),
            nn.Sigmoid() 
        )
        
        # 3. 放大系数 (初始化为 1.0 或更大，让 Attention 一开始就很有存在感)
        self.alpha = nn.Parameter(torch.tensor(2.0)) 

    def forward(self, visual_feat, phys_maps, phys_globals):
        # --- Spatial Mask ---
        # 1. High Res Calculation
        spatial_mask_high = self.spatial_gate(phys_maps) # [B, 1, 256, 256]
        
        # 2. Downsample
        target_h, target_w = visual_feat.shape[2], visual_feat.shape[3]
        spatial_mask = F.adaptive_avg_pool2d(spatial_mask_high, (target_h, target_w))
        
        # --- Channel Mask ---
        phys_maps_avg = torch.mean(phys_maps, dim=(2, 3))
        phys_globals_flat = phys_globals.view(phys_globals.shape[0], -1) 
        global_context = torch.cat([phys_globals_flat, phys_maps_avg], dim=1)
        channel_mask = self.channel_gate(global_context).unsqueeze(2).unsqueeze(3)
        
        # --- Residual Fusion with Boost ---
        # F_out = F + alpha * (F * M_s * M_c)
        attention_term = visual_feat * spatial_mask * channel_mask
        refined_feat = visual_feat + (self.alpha * attention_term)
        
        return refined_feat


class PhysicsGuidedIQA(nn.Module):
    def __init__(self, pretrained_rppem_path=None):
        super(PhysicsGuidedIQA, self).__init__()
        
        # Stream A: Frozen Physics
        self.rppem = R_PPEM()
        if pretrained_rppem_path:
            self._load_and_freeze_rppem(pretrained_rppem_path)

        # Stream B: Trainable Visual
        self.visual_net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.visual_backbone = nn.Sequential(*list(self.visual_net.children())[:-2])
        
        # Fusion
        self.phys_attn = PhysicsAttention(visual_channels=512, spatial_phys_channels=7, global_phys_channels=3)
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def _load_and_freeze_rppem(self, path):
        print(f"Loading R-PPEM from {path}...")
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('rppem.'):
                clean_state_dict[k.replace('rppem.', '')] = v
            # If the checkpoint is for the full EPCFQA model, it might contain 'acdf', 'lddf', 'nldcim' weights
            # which are not part of R_PPEM. We should only load R_PPEM related weights.
            # However, the user provided 'if not any(x in k for x in ['acdf', 'lddf', 'nldcim'])'
            # which implies that if key is NOT rppem. and NOT acdf/lddf/nldcim, it's considered.
            # This logic needs to be careful. The most robust way for freezing R_PPEM from a full model checkpoint
            # is to load the full state_dict and then only apply the relevant R_PPEM keys.
            # For simplicity, assuming the provided clean_state_dict logic is intended.
            elif not any(x in k for x in ['acdf', 'lddf', 'nldcim']):
                 # This part seems like it might accidentally include non-rppem weights if the checkpoint is a full model.
                 # For a pre-trained R_PPEM only, the first if condition `k.startswith('rppem.')` is usually sufficient.
                 # Given the context of the user's provided code, this branch might be for loading a general checkpoint
                 # and extracting R_PPEM weights if they are not prefixed.
                 # Let's stick to the user's provided logic for clean_state_dict for now.
                 clean_state_dict[k] = v # This line looks like it would try to load non-rppem parts if present in the model's direct state_dict
                                         # The more robust way would be to filter only R_PPEM keys explicitly.
        
        # Filter for keys that actually exist in self.rppem
        rppem_state_dict = self.rppem.state_dict()
        filtered_clean_state_dict = {
            k: v for k, v in clean_state_dict.items() if k in rppem_state_dict and v.shape == rppem_state_dict[k].shape
        }

        # Load the filtered state dict
        missing_keys, unexpected_keys = self.rppem.load_state_dict(filtered_clean_state_dict, strict=False)
        if missing_keys:
            print(f"R-PPEM missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"R-PPEM unexpected keys: {unexpected_keys}")
        
        for param in self.rppem.parameters():
            param.requires_grad = False
        self.rppem.eval() 

    def forward(self, x):
        with torch.no_grad(): # Ensure R_PPEM forward pass doesn't compute gradients
            mu_d, sigma_d, beta_D, beta_B, g, L = self.rppem(x)
            # Concat spatial physical maps: mu_d (1), sigma_d (1), beta_D (3), g (1), L (1) = 7 channels
            phys_maps = torch.cat([mu_d, sigma_d, beta_D, g, L], dim=1)
            # Global physical parameter: beta_B (3 channels)
            phys_globals = beta_B 

        vis_feat = self.visual_backbone(x)
        refined_feat = self.phys_attn(vis_feat, phys_maps, phys_globals)
        score = self.head(refined_feat)
        return score

# =========================================================
# Part 2: Utility Functions
# =========================================================

def get_image_paths_from_dir(image_dir):
    """
    遍历指定目录，并返回所有图片文件的绝对路径列表。
    """
    image_paths = []
    # 支持的图片扩展名
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    for root, _, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

# =========================================================
# Part 3: Batch Prediction Logic
# =========================================================

def batch_predict_scores(image_paths, model_path, output_path, gpu_id=0, rppem_weights_for_init=None):
    """
    批量预测图片分数。
    Args:
        image_paths (list): 包含所有图片绝对路径的列表。
        model_path (str): 完整模型 (PhysicsGuidedIQA) 的权重文件路径。
        output_path (str): 结果输出文件路径。
        gpu_id (int): 使用的 GPU ID。
        rppem_weights_for_init (str, optional): 如果PhysicsGuidedIQA的RPPEM部分
                                              需要单独加载预训练权重，这里提供路径。
                                              否则，PhysicsGuidedIQA会使用默认的ImageNet权重。
    """
    # 设备设置
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    # 实例化 PhysicsGuidedIQA 模型
    model = PhysicsGuidedIQA(pretrained_rppem_path=rppem_weights_for_init).to(device) 
    
    checkpoint = torch.load(model_path, map_location=device)
    # 兼容两种保存格式，通常是 'model_state_dict' 键下，或直接就是 state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval() # 设置模型为评估模式

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)), # 调整图像大小，与模型输入匹配
        transforms.ToTensor(),         # 转换为 Tensor
    ])

    if not image_paths:
        print(f"No images found for prediction.")
        return

    print(f"Total images to infer: {len(image_paths)}")

    results = []

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad(): # 在推理过程中禁用梯度计算，节省内存和计算
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            try:
                image = Image.open(img_path).convert('RGB') # 确保图像是 RGB 格式
            except Exception as e:
                print(f"Cannot read image: {img_path}, {str(e)}")
                continue
            
            # 准备模型输入
            input_tensor = transform(image).unsqueeze(0).to(device) # 增加 batch 维度并移动到设备

            # 模型前向传播，直接得到分数
            outputs = model(input_tensor)

            # 提取分数
            score = outputs.item() 
            results.append((img_name, score))
            print(f"{img_name}: {score:.6f}")

    # 保存分数结果
    with open(output_path, 'w') as f:
        for (img_name, score) in results:
            f.write(f"{img_name}\t{score:.6f}\n")
    print(f"Scores saved to: {output_path}")

# =========================================================
# Part 4: Main Execution
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring of images using PhysicsGuidedIQA model.")
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images to be scored.')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Single image path to be scored.')
    parser.add_argument('--model_path', type=str,
                        default="/mnt/SSD8T/home/xwz/lt/UnderwaterIQA/UIQA-main/output/dual_experiment/SOTA_2/best_model_seed_45.pth",
                        help='Path to the trained PhysicsGuidedIQA model weights.')
    parser.add_argument('--output_txt', type=str,
                        default='/mnt/SSD8T/home/xwz/lt/UnderwaterIQA/UIQA-main/predict/score_results.txt',
                        help='Output file to save image scores.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use, set to -1 for CPU.')
    parser.add_argument('--rppem_weights_for_init', type=str, default=None,
                        help='Optional: Path to pre-trained R-PPEM weights.')

    args = parser.parse_args()

    all_image_paths = []

    if args.image_path is not None:
        if os.path.isfile(args.image_path):
            all_image_paths = [args.image_path]
            print(f"Scoring single image: {args.image_path}")
        else:
            print(f"Image file not found: {args.image_path}")
            exit(1)

    elif args.image_dir is not None:
        print(f"Scanning image directory: {args.image_dir}")
        all_image_paths = get_image_paths_from_dir(args.image_dir)
        if not all_image_paths:
            print("No images found in the specified directory. Exiting.")
            exit(1)
    else:
        print("Please provide either --image_path or --image_dir")
        exit(1)

    print("Starting prediction...")
    batch_predict_scores(
        image_paths=all_image_paths,
        model_path=args.model_path,
        output_path=args.output_txt,
        gpu_id=args.gpu_id,
        rppem_weights_for_init=args.rppem_weights_for_init
    )
    print("Prediction completed.")

