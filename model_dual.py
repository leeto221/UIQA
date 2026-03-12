import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import math

# ---------------------------------------------------------
# Part 1: Basic Blocks & R_PPEM (Keep unchanged for weight loading)
# ---------------------------------------------------------

class DoubleConv(nn.Module):
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
    def __init__(self):
        super(R_PPEM, self).__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.shared_features = DoubleConv(512, 256)
        
        self.depth_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 2, 1)
        )
        self.beta_D_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 3, 1)
        )
        self.beta_B_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(256, 128, 1), nn.ReLU(True), nn.Conv2d(128, 3, 1)
        )
        self.g_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1)
        )
        self.light_path = nn.Sequential(
            DoubleConv(256, 128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1)
        )
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        features = self.features(x)
        shared = self.shared_features(features)
        
        depth_out = self.depth_path(shared)
        mu_d = torch.sigmoid(depth_out[:, 0:1]) * 12.0
        sigma_d = F.softplus(depth_out[:, 1:2]) + 1e-6
        
        log_min, log_max = math.log(0.001), math.log(4.0)
        beta_D = torch.exp(torch.sigmoid(self.beta_D_path(shared)) * (log_max - log_min) + log_min)
        
        log_min_b, log_max_b = math.log(0.001), math.log(5.0)
        beta_B = torch.exp(torch.sigmoid(self.beta_B_path(shared)) * (log_max_b - log_min_b) + log_min_b)

        g = torch.sigmoid(self.g_path(shared)) * 0.45 + 0.5
        L = torch.sigmoid(self.light_path(shared)) + 0.5
        
        uncertainty_weight = 1.0 / (1.0 + sigma_d)
        beta_D_up = self.upsample(beta_D * uncertainty_weight)
        g_up = self.upsample(g * uncertainty_weight)
        L_up = self.upsample(L * uncertainty_weight)
        mu_d_up = self.upsample(mu_d)
        sigma_d_up = self.upsample(sigma_d)
        
        return mu_d_up, sigma_d_up, beta_D_up, beta_B, g_up, L_up

# ---------------------------------------------------------
# Part 2: Physics-Guided Attention & Main Model
# ---------------------------------------------------------

class PhysicsAttention(nn.Module):
    def __init__(self, visual_channels=512, spatial_phys_channels=7, global_phys_channels=3):
        super(PhysicsAttention, self).__init__()
        
        # 1. Spatial Gate (High-Res)
        # 重点修改：加入了 InstanceNorm2d
        self.spatial_gate = nn.Sequential(
            # 第一层：提取物理特征
            nn.Conv2d(spatial_phys_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            
            # 第二层：压缩通道
            nn.Conv2d(64, 1, kernel_size=1),
            
            # 【核心修改】InstanceNorm2d
            # 强制让每张图的输出分布拉开，不让它缩在 0 附近
            # affine=True 允许网络学习缩放和平移，防止强行归一化破坏语义
            nn.InstanceNorm2d(1, affine=True), 
            
            # 激活函数
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
            elif not any(x in k for x in ['acdf', 'lddf', 'nldcim']):
                 clean_state_dict[k] = v
        
        self.rppem.load_state_dict(clean_state_dict, strict=False)
        for param in self.rppem.parameters():
            param.requires_grad = False
        self.rppem.eval() 

    def forward(self, x):
        with torch.no_grad():
            mu_d, sigma_d, beta_D, beta_B, g, L = self.rppem(x)
            phys_maps = torch.cat([mu_d, sigma_d, beta_D, g, L], dim=1)
            phys_globals = beta_B 

        vis_feat = self.visual_backbone(x)
        refined_feat = self.phys_attn(vis_feat, phys_maps, phys_globals)
        score = self.head(refined_feat)
        return score
