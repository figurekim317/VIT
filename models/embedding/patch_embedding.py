import copy
import logging
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """
    Patch Embedding module for VIT
    Splits the image into patches, flattens them, and projects them into embedding space.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Args:
            img_size (int): Size of the input image (assumes square image, e.g., 224x224).
            patch_size (int): Size of each patch (assumes square patches, e.g., 16x16).
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            embed_dim (int): Dimension of the embedding space.
        """
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d layer for patch embedding
        self.proj = nn.Conv2d(in_channels=in_channels,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size
                              )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_channels, img_size, img_size)
        
        """
        assert x.shape[2] == self.img_size and x.shape[3] == self.img_size, \
            f"Input image size must be {self.img_size}x{self.img_size}, but got {x.shape[2]}x{x.shape[3]}"

        # Project patches and flatten
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x