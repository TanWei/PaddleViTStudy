import paddle
import paddle.nn as nn

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
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

def generate_mask(window_size=4, shift_size=2, input_relolution=(8, 8)):
    H, W = input_relolution
    img_mask = paddle.zeros([1, H, W, 1]) #[1, 8, 8, 1]
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)] #
    
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
           img_mask[:, h_slice, w_slice, :] = cnt
           cnt+=1
    # [1, 8, 8, 1]
    window_mask = windows_partition(img_mask, window_size = window_size)
    # [4, 4, 4, 1]
    window_mask = window_mask.reshape([-1, window_size * window_size]) # 
    # [4, 16]
    attn_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)
    # [n, 1, ws*ws] - [n, ws*ws, 1]  Broadcasting 行向量减列向量：把两个矩阵补全为h*w的矩阵，然后按照元素相减
    attn_mask = paddle.where(attn_mask != 0, 
                             paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
    return attn_mask

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
    def forward(self, x, mask = None):
        # x: [B, num_patches, embed_dim]
        B, N, d = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.tranpose_multi_head, qkv)
        
        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        
        #### BEGIN CLASS 8
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows, num_patches, num_patches]
            # attn: [B*num_windows, num_heads, num_patches, num_patches] [256, 4, 49, 49]
            attn = attn.reshape([x.shape[0]//mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]]) 
            # attn: [B, num_windows, num_heads, num_patches, num_patches] [4, 64, 4, 49, 49]
            # temp = mask.unsqueeze(1).unsqueeze(0) # [1, 64, 1, 49, 49]
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            # [4, 64, 4, 49, 49]
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
            attn = self.softmax(attn)
        #### END CLASS 8
        out = paddle.matmul(attn, v) # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3])
        # [B, num_patches, num_heads, dim_head]
        out = out.reshape([B, N, d])
        out = self.proj(out)
        return out
    
class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        
        # CLASS 7
        if min(self.resolution) <= self.window_size:
            self.shift_size = 0
            self.windows_size = min(self.resolution)
        
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttenion(dim, window_size, num_heads)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)
        # BEGIN CLASS 8
        # generate mask and register buffer
        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size, self.resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)
        
    def forward(self, x):
        # x [4, 3136, 96] [batch_size, token_num, embed_dim]
        H, W =self.resolution # [56, 56]
        B, N, C = x.shape
        
        h = x
        x = self.attn_norm(x)
        
        x = x.reshape([B, H, W, C]) # [4, 56, 56, 96]
        
        ####BEGIN CLASS 8
        # shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1,2))
        else:
            shifted_x = x
        # compute window attn
        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask = self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)
        # shift back
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1,2))
        else:
            x = shifted_x
        ####END CLASS 8
        
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
    
class SwinStage(nn.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, windows_size, patch_merging=None):
        super().__init__()
        self.Blocks = nn.LayerList()
        for i in range(depth):
            self.Blocks.append(
                SwinBlock(dim=dim,
                          input_resolution=input_resolution,
                          num_heads=num_heads,
                          window_size=windows_size,
                          shift_size=0 if (i%2 == 0) else windows_size // 2)
            )
        if patch_merging is None:
            self.patch_merging = Identity()
        else:
            self.patch_merging = patch_merging(input_resolution, dim=dim)
    def forward(self, x):
        for Block in self.Blocks:
            x = Block(x)
        x = self.patch_merging(x)
        return x

class Swin(nn.Layer):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_channels=3,
                 embed_dim=96,
                 window_size=7,
                 num_heads=[3, 6, 12, 24],
                 depths=[2, 2, 6, 2],
                 num_classes=1000
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.num_features = int(self.embed_dim * 2**(self.num_stages-1))
        
        self.patch_embedding = PatchEmbedding(patch_size=patch_size,embed_dim=embed_dim)
        self.patch_resolution = [image_size // patch_size, image_size // patch_size]
        self.stages = nn.LayerList()
        for idx, (depth, n_heads) in enumerate(zip(self.depths, self.num_heads)):
            stage = SwinStage(
                dim=int(self.embed_dim * 2**idx),
                input_resolution=(self.patch_resolution[0] // (2**idx), self.patch_resolution[1] // (2**idx)),
                depth=depth,
                num_heads=n_heads,
                windows_size=window_size,
                patch_merging=PatchMerging if (idx < self.num_stages -1) else None)
            self.stages.append(stage)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.fc = nn.Linear(self.num_features, num_classes)
    def forward(self, x):
        x = self.patch_embedding(x)
        # [4, 3136, 96] [batch_size, token_num, embed_dim]
        for stage in self.stages:
            print('stage')
            x = stage(x)
        print(x.shape)
        # [4, 49, 768] [batch_size, (56/2^3)*(56/2^3), 8*96]
        x = self.norm(x) 
        x = x.transpose([0, 2, 1]) #[B, embed_dim, num_windows] [4, 768, 49]
        x = self.avgpool(x)
        # [4, 768, 1]
        print(x.shape)
        # [B, embed_dim, 1] [4, 768, 1]
        x = x.flatten(1)
        print(x.shape)
        # [4, 768]
        x = self.fc(x)
        print(x.shape)
        # [4, 1000]
        return x
    
def main():
    t = paddle.randn([4, 3, 224, 224])
    # patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    
    # swin_block_w_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    # swin_block_sw_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    
    # patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)
    
    # out = patch_embedding(t) # [4, 3136, 96]
    # print('patch_embedding shape= ', out.shape)
    # out = swin_block_w_msa(out) # [4, 3136, 96]
    # out = swin_block_sw_msa(out)
    # print('swin_block shape= ', out.shape)
    # out = patch_merging(out) # [4, 784, 192]
    # print('patch_merging shape= ', out.shape)
    model = Swin()
    print(model)
    out = model(t)
    print('out shape= ', out.shape)
    
if __name__ == "__main__":
    main()