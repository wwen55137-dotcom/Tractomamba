"""
Tractomamba streamline classification model.

This module defines the model architecture only. Trained parameters are loaded
from an external checkpoint by the inference script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba



# ==================== STN ====================
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self._init_weights()

    def _init_weights(self):
        # Initialize the predicted residual transform as zero, so the
        # initial transformation is exactly the identity matrix.
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, N)

        Returns:
            trans: Tensor of shape (B, 3, 3)
        """
        B = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)

        x = x.max(dim=2)[0]  # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x).view(B, 3, 3)

        identity = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        return x + identity


class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self._init_weights()

    def _init_weights(self):
        # Initialize the predicted residual transform as zero, so the
        # initial feature transformation is exactly the identity matrix.
        nn.init.constant_(self.fc3.weight, 0.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, K, N)

        Returns:
            trans: Tensor of shape (B, K, K)
        """
        B = x.shape[0]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)

        x = x.max(dim=2)[0]  # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x).view(B, self.k, self.k)

        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B, 1, 1)
        return x + identity


# ==================== Mamba Stack ====================
class MambaStack(nn.Module):
    """
    Weight-tied bidirectional Mamba stack.

    The same Mamba layer is applied to the forward and reversed point sequence,
    which encourages direction-robust streamline modeling and keeps the parameter
    count lower than using two independent directional branches.

    Args:
        d_model: Feature dimension D.
        n_layers: Number of stacked bidirectional Mamba blocks.
        dropout: Dropout probability applied only to the residual branch.
    """
    def __init__(self, d_model, n_layers=2, dropout=0.0):
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, but got {n_layers}.")

        self.layers = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(n_layers)
        ])

        self.fuse = nn.ModuleList([
            nn.Linear(2 * d_model, d_model) for _ in range(n_layers)
        ])

        self.norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, D)

        Returns:
            Tensor of shape (B, N, D)
        """
        x = x.contiguous()

        for i, layer in enumerate(self.layers):
            # Forward sequence modeling.
            x_fwd = layer(x)

            # Backward sequence modeling with weight sharing.
            x_rev = torch.flip(x, dims=[1]).contiguous()
            x_bwd = layer(x_rev)
            x_bwd = torch.flip(x_bwd, dims=[1]).contiguous()

            # Directional fusion followed by residual connection and LayerNorm.
            h = torch.cat([x_fwd, x_bwd], dim=-1)
            h = self.fuse[i](h)
            h = self.dropout(h)

            x = self.norm[i](x + h).contiguous()

        return x


# ==================== Tractomamba Feature Extractor ====================
class TractomambaFeatureExtractor(nn.Module):
    def __init__(
        self,
        k=0,
        k_global=0,
        global_feat=True,
        feature_transform=False,
        first_feature_transform=False,
        mamba_layers=2,
        mamba_dropout=0.1,
        dropout=0.3,
    ):
        super().__init__()

        self.k = k
        self.k_global = k_global
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.first_feature_transform = first_feature_transform

        # Optional input and feature transformations.
        if self.first_feature_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        # First feature extraction layer.
        if self.k + self.k_global == 0:
            self.conv1 = nn.Conv1d(3, 64, 1)
            self.bn1 = nn.BatchNorm1d(64)
        else:
            self.info_conv = nn.Conv2d(3 * 2, 64, 1)
            self.info_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # Gated max-mean pooling.
        self.gate = nn.Linear(1024, 1024)

        # Sequence modeling along streamline points.
        self.mamba1 = MambaStack(
            d_model=64,
            n_layers=mamba_layers,
            dropout=mamba_dropout,
        )
        self.mamba2 = MambaStack(
            d_model=128,
            n_layers=mamba_layers,
            dropout=mamba_dropout,
        )

    @staticmethod
    def _transform_info_point_set(info_point_set, trans):
        """
        Apply the same input transformation to info_point_set.

        Args:
            info_point_set: Tensor of shape (B, 3, N, K)
            trans: Tensor of shape (B, 3, 3)

        Returns:
            Transformed info_point_set with shape (B, 3, N, K)
        """
        B, C, N, K = info_point_set.shape
        if C != 3:
            raise ValueError(f"Expected info_point_set channel dimension to be 3, but got {C}.")

        info = info_point_set.permute(0, 2, 3, 1).contiguous()  # (B, N, K, 3)
        info = info.view(B, N * K, 3)
        info = torch.bmm(info, trans)
        info = info.view(B, N, K, 3)
        info = info.permute(0, 3, 1, 2).contiguous()  # (B, 3, N, K)

        return info

    def forward(self, xyz, info_point_set):
        """
        Args:
            xyz: Tensor of shape (B, 3, N)
            info_point_set: Tensor of shape (B, 3, N, K), where
                K = k + k_global when neighborhood information is used.

        Returns:
            global feature: (B, 1024) if global_feat=True
            trans: input transform or None
            trans_feat: feature transform or None
        """
        B, C, N = xyz.shape
        if C != 3:
            raise ValueError(f"Expected xyz channel dimension to be 3, but got {C}.")

        total_k = self.k + self.k_global

        # Optional input transform.
        if self.first_feature_transform:
            trans = self.stn(xyz)
            xyz = torch.bmm(xyz.transpose(1, 2).contiguous(), trans).transpose(1, 2).contiguous()

            # Keep xyz and info_point_set in the same transformed coordinate system.
            if total_k > 0 and info_point_set is not None:
                info_point_set = self._transform_info_point_set(info_point_set, trans)
        else:
            trans = None

        # First feature extraction.
        if total_k == 0:
            x = F.relu(self.bn1(self.conv1(xyz)), inplace=True)  # (B, 64, N)
        else:
            if info_point_set is None:
                raise ValueError("info_point_set must be provided when k + k_global > 0.")

            if info_point_set.shape[0] != B or info_point_set.shape[1] != 3 or info_point_set.shape[2] != N:
                raise ValueError(
                    "info_point_set must have shape (B, 3, N, K) with the same B and N as xyz."
                )
            if info_point_set.shape[3] != total_k:
                raise ValueError(
                    f"Expected info_point_set K dimension to be k + k_global = {total_k}, "
                    f"but got {info_point_set.shape[3]}."
                )

            xyz_exp = xyz.unsqueeze(-1).expand(-1, -1, -1, total_k)
            edge_feat = torch.cat([info_point_set - xyz_exp, xyz_exp], dim=1)  # (B, 6, N, K)

            x = F.relu(self.info_bn(self.info_conv(edge_feat)), inplace=True)
            x = x.max(dim=-1)[0].contiguous()  # (B, 64, N)

        # Mamba block at 64-D point features.
        x_seq = x.transpose(1, 2).contiguous()       # (B, N, 64)
        x_seq = self.mamba1(x_seq)
        x = x_seq.transpose(1, 2).contiguous()       # (B, 64, N)

        # Optional feature transform.
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(1, 2).contiguous(), trans_feat).transpose(1, 2).contiguous()
        else:
            trans_feat = None

        point_feat = x

        # Higher-level point features.
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)  # (B, 128, N)

        # Mamba block at 128-D point features.
        x_seq = x.transpose(1, 2).contiguous()       # (B, N, 128)
        x_seq = self.mamba2(x_seq)
        x = x_seq.transpose(1, 2).contiguous()       # (B, 128, N)

        # Final streamline feature projection.
        x = self.bn3(self.conv3(x))                  # (B, 1024, N)

        # Max + gated mean pooling.
        x_max = x.max(dim=2)[0]                      # (B, 1024)
        x_mean = x.mean(dim=2)                       # (B, 1024)
        gate = torch.sigmoid(self.gate(x_max))       # (B, 1024)
        x_global = x_max + gate * x_mean             # (B, 1024)

        if self.global_feat:
            return x_global, trans, trans_feat

        x_global = x_global.unsqueeze(-1).expand(-1, -1, N)
        return torch.cat([x_global, point_feat], dim=1), trans, trans_feat


# ==================== Classifier ====================
class TractomambaClassifier(nn.Module):
    def __init__(
        self,
        k=0,
        k_global=0,
        num_classes=2,
        feature_transform=False,
        first_feature_transform=False,
        mamba_layers=2,
        mamba_dropout=0.1,
        dropout=0.3,
    ):
        super().__init__()

        self.feat = TractomambaFeatureExtractor(
            k=k,
            k_global=k_global,
            global_feat=True,
            feature_transform=feature_transform,
            first_feature_transform=first_feature_transform,
            mamba_layers=mamba_layers,
            mamba_dropout=mamba_dropout,
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, xyz, info_point_set):
        """
        Args:
            xyz: Tensor of shape (B, 3, N)
            info_point_set: Tensor of shape (B, 3, N, K)

        Returns:
            log_probs: Tensor of shape (B, num_classes)
            trans: input transform or None
            trans_feat: feature transform or None
        """
        x, trans, trans_feat = self.feat(xyz, info_point_set)

        x = F.relu(self.bn1(self.fc1(x)), inplace=True)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)

        x = self.fc3(x)

        return F.log_softmax(x, dim=1), trans, trans_feat

# ==================== Regularization ====================
def feature_transform_regularizer(trans):
    """
    Orthogonality regularization for STN feature transformation.

    Args:
        trans: Tensor of shape (B, D, D), or None.

    Returns:
        A scalar regularization loss. If trans is None, returns 0.0.
    """
    if trans is None:
        return 0.0

    B, D, _ = trans.shape
    I = torch.eye(D, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss
