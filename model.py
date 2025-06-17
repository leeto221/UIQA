import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torchvision.models import ResNet18_Weights


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
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)  # Output depth mean and uncertainty
        )
        
        # beta_D estimation path (channel-specific, output H×W×3)
        self.beta_D_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)  # RGB three channels
        )
        
        # beta_B estimation path (global parameter, output 3)
        self.beta_B_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1)  # RGB three channels
        )
        
        # g estimation path (spatially varying, output H×W×1)
        self.g_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # Lighting estimation path L (spatially varying, output H×W×1)
        self.light_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
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
        mu_d = torch.sigmoid(depth_out[:, 0:1, :, :]) * 12.0  # Depth range [0, 12]
        sigma_d = F.softplus(depth_out[:, 1:2, :, :]) + 1e-6  # Ensure variance is positive
        
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
        beta_D_final = beta_D * uncertainty_weight
        beta_B_final = beta_B  # beta_B is global, no adjustment
        g_final = g * uncertainty_weight
        L_final = L * uncertainty_weight
        
        # Upsample to original resolution
        mu_d_up = self.upsample(mu_d)
        sigma_d_up = self.upsample(sigma_d)
        beta_D_up = self.upsample(beta_D_final)
        g_up = self.upsample(g_final)
        L_up = self.upsample(L_final)
        
        # beta_B is global, no upsampling, but needs to be broadcast to the whole image
        batch_size = x.shape[0]
        beta_B_broadcast = beta_B_final.expand(batch_size, 3, mu_d_up.shape[2], mu_d_up.shape[3])
        
        return mu_d_up, sigma_d_up, beta_D_up, beta_B_final, beta_B_broadcast, g_up, L_up

"""not used"""
class A_CDF(nn.Module):
    def __init__(self):
        super(A_CDF, self).__init__()

    def forward(self, I, d, beta_D):
        epsilon = 1e-6
        I_R = I[:, 0:1, :, :]
        I_G = I[:, 1:2, :, :]
        I_B = I[:, 2:3, :, :]
        beta_D_R = beta_D[:, 0:1, :, :]
        beta_D_G = beta_D[:, 1:2, :, :]
        beta_D_B = beta_D[:, 2:3, :, :]

        I_R = torch.clamp(I_R, min=0.0) + epsilon
        I_G = torch.clamp(I_G, min=0.0) + epsilon
        I_B = torch.clamp(I_B, min=0.0) + epsilon

        ratio_RG = I_R / I_G
        ratio_RB = I_R / I_B
        observed_log_RG = torch.log(torch.clamp(ratio_RG, min=1e-6, max=1e6))
        observed_log_RB = torch.log(torch.clamp(ratio_RB, min=1e-6, max=1e6))

        expected_log_RG = - (beta_D_R - beta_D_G) * d
        expected_log_RB = - (beta_D_R - beta_D_B) * d
        distortion_RG = torch.abs(observed_log_RG - expected_log_RG)
        distortion_RB = torch.abs(observed_log_RB - expected_log_RB)
        A_CDF = distortion_RG + distortion_RB
        return A_CDF

"""not used"""
class L_DDF(nn.Module):
    def __init__(self, f_rep=0.1):
        super(L_DDF, self).__init__()
        self.f_rep = f_rep  

    def forward(self, I, d, beta_B, g):
        kernel_x = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 3).repeat(3, 1, 1, 1).to(I.device)
        kernel_y = torch.tensor([[-1], [0], [1]], dtype=torch.float32).view(1, 1, 3, 1).repeat(3, 1, 1, 1).to(I.device)
        grad_x = F.conv2d(I, kernel_x, padding=(0,1), groups=3)
        grad_y = F.conv2d(I, kernel_y, padding=(1,0), groups=3)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        grad_mag = torch.mean(grad_mag, dim=1, keepdim=True)
        
        beta_B_mean = torch.mean(beta_B, dim=1, keepdim=True)
        k = beta_B_mean * (1 - g)
        exp_input = k * (self.f_rep**2) * torch.clamp(d, min=0.0, max=12.0)  
        exp_input = torch.clamp(exp_input, min=-50, max=50)
        W = torch.exp(exp_input)
        L_DDF = grad_mag * W
        return L_DDF


class MLPHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.mlp(x)




class PhysicalConsistencyLoss(nn.Module):
    """Physical Consistency Loss"""
    def __init__(self, kernel_size=33):
        super(PhysicalConsistencyLoss, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        
    def generate_psf_kernel(self, depth, beta, g):
        """Generate PSF kernel"""
        # Create grid
        center = self.kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-center, center + 1, device=depth.device),
            torch.arange(-center, center + 1, device=depth.device),
            indexing='ij'
        )
        
        # Compute radius
        r = torch.sqrt(x**2 + y**2 + 1e-6)
        
        # Compute cos_theta
        cos_theta = depth / torch.sqrt(r**2 + depth**2 + 1e-6)
        
        # Compute g^2
        g_squared = g**2
        
        # Compute phase term
        phase = (1 - g_squared) / (4 * math.pi * (1 + g_squared - 2 * g * cos_theta)**1.5)
        
        # Compute transmission
        transmission = torch.exp(-beta * depth)
        
        # Compute PSF
        kernel = phase * transmission / (r**2 + 1e-6)
        
        # Normalize
        kernel = kernel / (kernel.sum() + 1e-6)
        
        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]
    
    def forward(self, I, J, mu_d, beta_D, beta_B, g, L, B_inf=None):
        """Compute physical consistency loss
        
        Args:
            I: Degraded image
            J: Original clear image (use the original synthetic image)
            mu_d: Depth
            beta_D: Absorption coefficient
            beta_B: Background light absorption coefficient
            g: Anisotropy coefficient
            L: Lighting
            
        Returns:
            loss: Physical consistency loss
            I_recon: Reconstructed image
        """
        batch_size, channels, height, width = I.shape
        
        # Extract B_inf (use mean of I's border as approximation)
        if B_inf is None:
            print("Warning: B_inf not provided, using zeros as replacement.")
            B_inf = torch.zeros(batch_size, channels, 1, 1, device=I.device)
        
        # Initialize reconstructed image
        I_recon = torch.zeros_like(I)
        
        # Process each sample and channel separately
        for b in range(batch_size):
            for c in range(channels):
                # Center depth, beta, and g
                center_d = mu_d[b, 0, height//2, width//2]
                center_beta = beta_D[b, c, height//2, width//2]
                center_g = g[b, 0, height//2, width//2]
                
                # Generate PSF kernel
                kernel = self.generate_psf_kernel(center_d, center_beta, center_g)
                
                # Convolution
                direct = F.conv2d(
                    F.pad(J[b:b+1, c:c+1], (self.pad, self.pad, self.pad, self.pad), mode='replicate'),
                    kernel,
                    padding=0
                )
                
                # Transmission
                t = torch.exp(-beta_D[b:b+1, c:c+1] * mu_d[b:b+1, 0:1])
                
                # Background term
                B_term = B_inf[b:b+1, c:c+1] * (1 - torch.exp(-beta_B[b, c, 0, 0] * mu_d[b:b+1, 0:1]))
                
                # Reconstruction
                I_recon[b:b+1, c:c+1] = direct * t * L[b:b+1, 0:1] + B_term
        
        # L1 loss
        loss = torch.mean(torch.abs(I - I_recon))
        
        return loss, I_recon

class EPCFQA(nn.Module):
    """E-PCFQA full model"""
    def __init__(self, image_size=256, ablation_config=None):
        super(EPCFQA, self).__init__()
        
        # Default config: use all features
        self.ablation_config = {
            'use_beta_D': True,
            'use_beta_B': True,
            'use_g': True,
            'use_L': True,
            'use_A_CDF': True,
            'use_L_DDF': True
        }
        
        # Update if config provided
        if ablation_config is not None:
            self.ablation_config.update(ablation_config)
        
        # Modules
        self.rppem = R_PPEM()
        self.acdf = A_CDF()
        self.lddf = L_DDF()

        #self.nldcim = NL_DCIM(image_size=image_size)

        combined_feature_dim = 10
        self.nldcim = MLPHead(in_channels=combined_feature_dim, hidden_dim=512)


        # Physical consistency loss
        self.phys_loss = PhysicalConsistencyLoss()
        
    def forward(self, I, J=None, B_inf=None, dataset_type=None):
        # Physical parameter estimation
        mu_d, sigma_d, beta_D, beta_B_global, beta_B, g, L = self.rppem(I)
        
        # Compute all features (even if not used, for saving original features)
        A_CDF_feature = self.acdf(I, mu_d, beta_D)
        L_DDF_feature = self.lddf(I, mu_d, beta_B, g)

        if torch.isnan(A_CDF_feature).any() or torch.isinf(A_CDF_feature).any():
            print("A_CDF_feature contains NaN or inf")
        if torch.isnan(L_DDF_feature).any() or torch.isinf(L_DDF_feature).any():
            print("L_DDF_feature contains NaN or inf")

        
        # Selectively replace features with zeros according to ablation config
        # Key: use detach() to break gradient
        if not self.ablation_config['use_beta_D']:
            beta_D = torch.zeros_like(beta_D).detach()
            
        if not self.ablation_config['use_beta_B']:
            beta_B = torch.zeros_like(beta_B).detach()
            
        if not self.ablation_config['use_g']:
            g = torch.zeros_like(g).detach()
            
        if not self.ablation_config['use_L']:
            L = torch.zeros_like(L).detach()
            
        if not self.ablation_config['use_A_CDF']:
            A_CDF_feature = torch.zeros_like(A_CDF_feature).detach()
            
        if not self.ablation_config['use_L_DDF']:
            L_DDF_feature = torch.zeros_like(L_DDF_feature).detach()
        
        # Combine features as input to NL-DCIM
        combined_features = torch.cat([
            beta_D,                # 3 channels
            beta_B,                # 3 channels
            g,                     # 1 channel
            A_CDF_feature,         # 1 channel
            L_DDF_feature,         # 1 channel
            L                      # 1 channel
        ], dim=1)
        
        # Quality score prediction
        raw_quality_score = self.nldcim(combined_features)
        quality_score = raw_quality_score
        
        # If original image J is provided, compute physical consistency loss
        # Note: physical consistency loss always uses original features
        phys_loss = None
        I_recon = None
        if J is not None:
            phys_loss, I_recon = self.phys_loss(I, J, mu_d, beta_D, beta_B_global, g, L, B_inf)
        
        return {
            'score': quality_score,
            'raw_score': raw_quality_score,
            'mu_d': mu_d,
            'sigma_d': sigma_d,
            'beta_D': beta_D,
            'beta_B': beta_B_global,
            'g': g,
            'L': L,
            'A_CDF': A_CDF_feature,
            'L_DDF': L_DDF_feature,
            'phys_loss': phys_loss,
            'I_recon': I_recon
        }

    
    def compute_loss(self, I, J, gt_scores, dataset_type=None, B_inf=None, lambda_phys=1.0, lambda_reg=0.01):
        """Compute total loss
        
        Args:
            I: Degraded image
            J: Original clear image (use the original synthetic image)
            gt_scores: Ground truth quality scores
            dataset_type: Dataset type identifier
            B_inf: Preloaded background light value
            lambda_phys: Physical consistency loss weight
            lambda_reg: Regularization loss weight
            
        Returns:
            total_loss: Total loss
            losses_dict: Dictionary of each loss part
        """
        # Forward pass
        outputs = self.forward(I, J, B_inf, dataset_type)
        
        # Get predicted scores
        pred_scores = outputs['score']
        
        # MOS loss (Huber)
        mos_loss = F.huber_loss(pred_scores, gt_scores, delta=0.5)
        
        # Physical consistency loss
        phys_loss = outputs['phys_loss']
        
        # Regularization loss
        reg_loss = sum(torch.sum(p**2) for p in self.parameters())
        
        # Total loss
        total_loss = mos_loss + lambda_phys * phys_loss + lambda_reg * reg_loss
        
        # Return loss dictionary
        losses_dict = {
            'total_loss': total_loss,
            'mos_loss': mos_loss,
            'phys_loss': phys_loss,
            'reg_loss': reg_loss
        }
        
        return total_loss, losses_dict


    
    def pretrain_rppem(self, I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf=None,
                   lambda_d=1.0, lambda_beta=0.6, lambda_g=0.4, lambda_L=0.4, lambda_phys=1.0):
        """Pretrain R-PPEM module
        
        Args:
            I: Degraded image
            J: Original clear image
            d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt: Ground truth physical parameters
            B_inf: Preloaded background light value
            lambda_*: Loss weights
            
        Returns:
            total_loss: Total loss
            losses_dict: Dictionary of each loss part
        """
        # Only forward R-PPEM
        mu_d, sigma_d, beta_D, beta_B_global, beta_B, g, L = self.rppem(I)
        
        # Depth loss
        depth_loss = F.l1_loss(mu_d, d_gt)
        
        # Parameter losses
        beta_D_loss = F.mse_loss(beta_D, beta_D_gt)
        beta_B_loss = F.mse_loss(beta_B_global, beta_B_gt)
        g_loss = F.mse_loss(g, g_gt)
        L_loss = F.mse_loss(L, L_gt)
        
        # Physical consistency loss
        phys_loss, _ = self.phys_loss(I, J, mu_d, beta_D, beta_B_global, g, L, B_inf)
        
        # Total loss
        total_loss = (lambda_d * depth_loss + 
                    lambda_beta * (beta_D_loss + beta_B_loss) + 
                    lambda_g * g_loss + 
                    lambda_L * L_loss + 
                    lambda_phys * phys_loss)
        
        # Return loss dictionary
        losses_dict = {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'beta_D_loss': beta_D_loss,
            'beta_B_loss': beta_B_loss,
            'g_loss': g_loss,
            'L_loss': L_loss,
            'phys_loss': phys_loss
        }
        
        return total_loss, losses_dict



def test_model():
    """Test if the model works properly"""
    # Create a random small batch of images
    batch_size = 2
    I = torch.rand(batch_size, 3, 256, 256)
    J = torch.rand(batch_size, 3, 256, 256)
    
    # Create random B_inf
    B_inf = torch.rand(batch_size, 3, 1, 1)
    
    # Create model
    model = EPCFQA(image_size=256)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(I)
    
    # Print output shapes
    print("Quality score shape:", outputs['score'].shape)
    print("Depth map shape:", outputs['mu_d'].shape)
    print("Beta_D shape:", outputs['beta_D'].shape)
    print("Beta_B shape:", outputs['beta_B'].shape)
    print("g shape:", outputs['g'].shape)
    print("L shape:", outputs['L'].shape)
    print("A_CDF shape:", outputs['A_CDF'].shape)
    print("L_DDF shape:", outputs['L_DDF'].shape)
    
    # Test physical consistency loss (with B_inf)
    with torch.no_grad():
        outputs_with_J = model(I, J, B_inf)
    
    print("Physical consistency loss:", outputs_with_J['phys_loss'].item())
    print("Reconstructed image shape:", outputs_with_J['I_recon'].shape)
    
    # Test overall loss computation
    gt_scores = torch.rand(batch_size, 1)
    total_loss, losses = model.compute_loss(I, J, outputs['score'], gt_scores, B_inf)
    
    print("Total loss:", total_loss.item())
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item()}")
    
    return model



if __name__ == "__main__":
    # Test the model
    model = test_model()