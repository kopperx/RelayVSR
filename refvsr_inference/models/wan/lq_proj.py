import torch
import torch.nn as nn
from einops import rearrange


class RMSNorm2d(nn.Module):
    """
    RMSNorm over the channel dimension.

    Input:
        [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]

        dtype = x.dtype
        x = x.float()

        x = x * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + self.eps)

        # Follow PyTorch RMSNorm numerical order:
        # normalize -> affine -> cast back
        x = x * self.weight

        return x.to(dtype)


class PixelUnshuffle2d(nn.Module):
    def __init__(
        self,
        patch_h: int,
        patch_w: int,
    ):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        return rearrange(
            x,
            "b c (h ph) (w pw) -> b (c ph pw) h w",
            ph=self.patch_h,
            pw=self.patch_w,
        )


class LQ4xProj(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 3072,
        hidden_dim1: int = 2048,
        hidden_dim2: int = 3072,
    ):
        super().__init__()

        self.patch_size = 32

        self.pixel_unshuffle = PixelUnshuffle2d(
            self.patch_size // 2,
            self.patch_size // 2,
        )

        patch_dim = in_dim * self.patch_size // 2 * self.patch_size // 2

        self.conv1 = nn.Conv2d(
            patch_dim,
            hidden_dim1,
            kernel_size=1,
        )

        self.norm1 = RMSNorm2d(hidden_dim1)

        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            hidden_dim1,
            hidden_dim2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.norm2 = RMSNorm2d(hidden_dim2)

        self.act2 = nn.SiLU()

        self.out_proj = nn.Linear(
            hidden_dim2,
            out_dim,
        )

        self._zero_init()

    def _zero_init(self):
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, video):
        # video:
        # [B, C, F, H, W]

        b, c, f, h, w = video.shape

        # 每帧独立进行 spatial encoding
        x = rearrange(
            video,
            "b c f h w -> (b f) c h w",
        )

        x = self.pixel_unshuffle(x)

        x = self.act1(self.norm1(self.conv1(x)))

        x = self.act2(self.norm2(self.conv2(x)))

        # 恢复 Wan token 顺序
        x = rearrange(
            x,
            "(b f) c h w -> b (f h w) c",
            b=b,
            f=f,
        )

        return self.out_proj(x)
