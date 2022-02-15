from turtle import forward
import paddle
import paddle.nn as nn

paddle.set_device('cpu')

class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x [n, c, h, w]
        x = self.patch_embed(x) # [n, embed_dim,h', w']
        x = x.flatten(2) # [n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1]) # [n, h'*w', embed_dim]
        x = self.norm(x)
        return x
    
class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)
    def forward(self, x):
        # x [4, 3136, 96]
        h, w = self.resolution # [56, 56]
        b, _, c = x.shape
        
        x = x.reshape([b, h, w, c])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = paddle.concat([x0, x1, x2, x3], axis=-1) # [B, h, w, 4*c]
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x) # [4, 784， 384]
        x = self.reduction(x) # [4, 784， 192]
        return x
class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))        
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C]) # [4, 8, 7, 8, 7, 96]
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B, h//ws, w//ws, ws, ws, c]
    print(x.shape) # [4, 8, 8, 7, 7, 96]
    x = x.reshape([-1, window_size, window_size, C]) # [256, 7, 7, 96]
    # [B * num_patches, ws, ws, c]
    return x
def windows_reverse(windows, windows_size, H, W):
    B = int(windows.shape[0] // (H / windows_size * W / windows_size))
    x = windows.reshape([B, H//windows_size, W//windows_size, windows_size, windows_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x

class WindowAttenion(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def tranpose_multi_head(self, x):
        print(x.shape) # [4, 8, 8, 7, 7, 96]
        print(x.shape[:-1]) # [256, 49]
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        # new_shape 是一个list [256, 49, 4, 24]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3]) # [B, num_heads, num_patches, dim_head]
        return x
    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        B, N, d = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.tranpose_multi_head, qkv)
        
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        
        out = paddle.matmul(attn, v) # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3])
        # [B, num_patches, num_heads, dim_head]
        out = out.reshape([B, N, d])
        out = self.proj(out)
        return out
    
class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttenion(dim, window_size, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
    def forward(self, x):
        # x [4, 3136, 96] [batch_size, token_num, embed_dim]
        H, W =self.resolution # [56, 56]
        B, N, C = x.shape
        
        h = x
        x = self.attn_norm(x)
        
        x = x.reshape([B, H, W, C]) # [4, 56, 56, 96]
        x_windows = windows_partition(x, self.window_size) 
        # [B * num_patches, ws, ws, c] [256, 7, 7, 96]
        x_windows = x_windows.reshape([-1, self.window_size*self.window_size, C])
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        x = windows_reverse(attn_windows, self.window_size, H, W)
        # [B,H,W,C]
        x = x.reshape([B, H*W, C])
        x = h + x
        
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x

def main():
    t = paddle.randn([4, 3, 224, 224])
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    
    swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)
    
    out = patch_embedding(t) # [4, 3136, 96]
    print('patch_embedding shape= ', out.shape)
    out = swin_block(out) # [4, 3136, 96]
    print('swin_block shape= ', out.shape)
    out = patch_merging(out) # [4, 784, 192]
    print('patch_merging shape= ', out.shape)
    
if __name__ == "__main__":
    main()