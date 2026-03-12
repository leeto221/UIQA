import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
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

        # Backbone
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Shared feature extractor
        self.shared_features = DoubleConv(512, 256)

        # Depth branch: output mean and uncertainty
        self.depth_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # beta_D branch
        self.beta_D_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        # beta_B branch (global)
        self.beta_B_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=1)
        )

        # g branch
        self.g_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # lighting branch
        self.light_path = nn.Sequential(
            DoubleConv(256, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.features(x)
        shared = self.shared_features(features)

        # Depth
        depth_out = self.depth_path(shared)
        mu_d = torch.sigmoid(depth_out[:, 0:1, :, :]) * 12.0
        sigma_d = F.softplus(depth_out[:, 1:2, :, :]) + 1e-6

        # beta_D in [0.001, 4.0]
        log_min_d, log_max_d = math.log(0.001), math.log(4.0)
        beta_D_raw = torch.exp(
            torch.sigmoid(self.beta_D_path(shared)) * (log_max_d - log_min_d) + log_min_d
        )

        # beta_B in [0.001, 5.0]
        log_min_b, log_max_b = math.log(0.001), math.log(5.0)
        beta_B_global = torch.exp(
            torch.sigmoid(self.beta_B_path(shared)) * (log_max_b - log_min_b) + log_min_b
        )

        # g in [0.5, 0.95]
        g_raw = torch.sigmoid(self.g_path(shared)) * 0.45 + 0.5

        # L in [0.5, 1.5]
        L_raw = torch.sigmoid(self.light_path(shared)) + 0.5

        # Uncertainty weighting (used only for feature modulation, not regression supervision)
        uncertainty_weight = 1.0 / (1.0 + sigma_d)
        beta_D_mod = beta_D_raw * uncertainty_weight
        g_mod = g_raw * uncertainty_weight
        L_mod = L_raw * uncertainty_weight

        # Upsample to input resolution
        mu_d_up = self.upsample(mu_d)
        sigma_d_up = self.upsample(sigma_d)

        beta_D_raw_up = self.upsample(beta_D_raw)
        g_raw_up = self.upsample(g_raw)
        L_raw_up = self.upsample(L_raw)

        beta_D_mod_up = self.upsample(beta_D_mod)
        g_mod_up = self.upsample(g_mod)
        L_mod_up = self.upsample(L_mod)

        batch_size = x.shape[0]
        beta_B_broadcast = beta_B_global.expand(
            batch_size, 3, mu_d_up.shape[2], mu_d_up.shape[3]
        )

        return {
            'mu_d': mu_d_up,
            'sigma_d': sigma_d_up,
            'beta_D_raw': beta_D_raw_up,
            'beta_B_global': beta_B_global,
            'beta_B_broadcast': beta_B_broadcast,
            'g_raw': g_raw_up,
            'L_raw': L_raw_up,
            'beta_D_mod': beta_D_mod_up,
            'g_mod': g_mod_up,
            'L_mod': L_mod_up,
            'W_unc': 1.0 / (1.0 + sigma_d_up)
        }


class PhysicalConsistencyLoss(nn.Module):
    """Physical reconstruction loss used in pretraining"""
    def __init__(self, kernel_size=33):
        super(PhysicalConsistencyLoss, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def generate_psf_kernel(self, depth, beta, g):
        center = self.kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-center, center + 1, device=depth.device),
            torch.arange(-center, center + 1, device=depth.device),
            indexing='ij'
        )

        r = torch.sqrt(x ** 2 + y ** 2 + 1e-6)
        cos_theta = depth / torch.sqrt(r ** 2 + depth ** 2 + 1e-6)
        g_squared = g ** 2

        phase = (1 - g_squared) / (
            4 * math.pi * (1 + g_squared - 2 * g * cos_theta) ** 1.5
        )
        transmission = torch.exp(-beta * depth)
        kernel = phase * transmission / (r ** 2 + 1e-6)
        kernel = kernel / (kernel.sum() + 1e-6)

        return kernel.unsqueeze(0).unsqueeze(0)

    def forward(self, I, J, mu_d, beta_D, beta_B, g, L, B_inf=None):
        batch_size, channels, height, width = I.shape

        if B_inf is None:
            B_inf = torch.zeros(batch_size, channels, 1, 1, device=I.device)

        I_recon = torch.zeros_like(I)

        for b in range(batch_size):
            for c in range(channels):
                center_d = mu_d[b, 0, height // 2, width // 2]
                center_beta = beta_D[b, c, height // 2, width // 2]
                center_g = g[b, 0, height // 2, width // 2]

                kernel = self.generate_psf_kernel(center_d, center_beta, center_g)

                direct = F.conv2d(
                    F.pad(J[b:b+1, c:c+1], (self.pad, self.pad, self.pad, self.pad), mode='replicate'),
                    kernel,
                    padding=0
                )

                t = torch.exp(-beta_D[b:b+1, c:c+1] * mu_d[b:b+1, 0:1])

                B_term = B_inf[b:b+1, c:c+1] * (
                    1 - torch.exp(-beta_B[b, c, 0, 0] * mu_d[b:b+1, 0:1])
                )

                I_recon[b:b+1, c:c+1] = direct * t * L[b:b+1, 0:1] + B_term

        loss = torch.mean(torch.abs(I - I_recon))
        return loss, I_recon


class EPCFQA(nn.Module):
    """
    Pretraining-only model.
    Only keeps R-PPEM and physical reconstruction loss.
    """
    def __init__(self, image_size=256):
        super(EPCFQA, self).__init__()
        self.rppem = R_PPEM()
        self.phys_loss = PhysicalConsistencyLoss()

    def forward(self, I):
        """
        Forward for parameter prediction only.
        """
        return self.rppem(I)

    def pretrain_rppem(self, I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf=None):
        outputs = self.rppem(I)

        mu_d = outputs['mu_d']
        sigma_d = outputs['sigma_d']
        beta_D_raw = outputs['beta_D_raw']
        beta_B_global = outputs['beta_B_global']
        g_raw = outputs['g_raw']
        L_raw = outputs['L_raw']

        # 1) Parameter regression loss
        depth_loss = (
            ((mu_d - d_gt) ** 2) / (sigma_d ** 2 + 1e-8) +
            torch.log(sigma_d ** 2 + 1e-8)
        ).mean()

        beta_D_loss = ((beta_D_raw - beta_D_gt) ** 2).mean()
        beta_B_loss = ((beta_B_global - beta_B_gt) ** 2).mean()
        g_loss = ((g_raw - g_gt) ** 2).mean()
        L_loss = ((L_raw - L_gt) ** 2).mean()

        L_rec = depth_loss + beta_D_loss + beta_B_loss + g_loss + L_loss

        # 2) Reconstruction loss
        L_recon, I_recon = self.phys_loss(
            I, J, mu_d, beta_D_raw, beta_B_global, g_raw, L_raw, B_inf
        )

        # 3) Total loss
        total_loss = L_rec + L_recon

        losses_dict = {
            'total_loss': total_loss,
            'L_rec': L_rec,
            'L_recon': L_recon,
            'depth_loss': depth_loss,
            'beta_D_loss': beta_D_loss,
            'beta_B_loss': beta_B_loss,
            'g_loss': g_loss,
            'L_loss': L_loss
        }

        return total_loss, losses_dict


def test_model():
    batch_size = 2
    I = torch.rand(batch_size, 3, 256, 256)
    J = torch.rand(batch_size, 3, 256, 256)

    d_gt = torch.rand(batch_size, 1, 256, 256) * 12.0
    beta_D_gt = torch.rand(batch_size, 3, 256, 256)
    beta_B_gt = torch.rand(batch_size, 3, 1, 1)
    g_gt = torch.rand(batch_size, 1, 256, 256) * 0.45 + 0.5
    L_gt = torch.rand(batch_size, 1, 256, 256) + 0.5
    B_inf = torch.rand(batch_size, 3, 1, 1)

    model = EPCFQA(image_size=256)

    with torch.no_grad():
        outputs = model(I)

    print("mu_d:", outputs['mu_d'].shape)
    print("sigma_d:", outputs['sigma_d'].shape)
    print("beta_D_raw:", outputs['beta_D_raw'].shape)
    print("beta_B_global:", outputs['beta_B_global'].shape)
    print("g_raw:", outputs['g_raw'].shape)
    print("L_raw:", outputs['L_raw'].shape)

    total_loss, losses_dict = model.pretrain_rppem(
        I, J, d_gt, beta_D_gt, beta_B_gt, g_gt, L_gt, B_inf
    )

    print("total_loss:", total_loss.item())
    for k, v in losses_dict.items():
        print(f"{k}: {v.item()}")


if __name__ == "__main__":
    test_model()
