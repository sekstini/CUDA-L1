import torch
import torch.nn as nn
import torch.nn.functional as F

class FastPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pos_embedding):
        ctx.save_for_backward(pos_embedding)
        return x + pos_embedding

    @staticmethod
    def backward(ctx, grad_output):
        pos_embedding, = ctx.saved_tensors
        return grad_output, grad_output.sum(dim=0, keepdim=True)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model with optimized implementation.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(ModelNew, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # Use standard PyTorch transformer with JIT compilation if available
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        try:
            if torch.cuda.is_available():
                self.transformer = torch.jit.script(self.transformer)
        except:
            pass
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        # Cache for expanded class tokens
        self.cls_token_cache = {}
        
        # Pre-compute constants
        self.num_patches = num_patches
        self.unfold_size = (patch_size, patch_size)
        self.unfold_stride = (patch_size, patch_size)
    
    def forward(self, img):
        """
        Forward pass of the Vision Transformer with optimized implementation.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        batch_size = img.shape[0]
        
        # Optimized patch extraction using F.unfold
        x = F.unfold(img, kernel_size=self.unfold_size, stride=self.unfold_stride)
        x = x.transpose(1, 2).contiguous()  # [B, num_patches, C*p*p]
        
        # Apply linear transformation
        x = self.patch_to_embedding(x)
        
        # Get or create expanded class token
        if batch_size in self.cls_token_cache:
            cls_tokens = self.cls_token_cache[batch_size]
        else:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            if batch_size <= 32:  # Only cache for common batch sizes
                self.cls_token_cache[batch_size] = cls_tokens
        
        # Concatenate class token
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding and apply dropout using custom CUDA function
        try:
            x = FastPositionEmbedding.apply(x, self.pos_embedding)
        except:
            x = x + self.pos_embedding
            
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Extract class token and apply MLP head
        x = x[:, 0]  # Faster than using self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.randn(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]