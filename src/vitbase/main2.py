from re import X
from turtle import forward
import paddle
import paddle.nn as nn

class Identify(nn.Layer):
    def __init__(self):
        super().__init__()
    
    def forward(x):
        return x
    
class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0):
        super().__init__()
        self.patch_embed = nn.Conv2D(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     bias_attr=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x:[1,1,28,28]
        x = self.patch_embed(x)
        # x:[n, embed_dim, h', w']
        x = x.flatten(2) # [n, emded_dim, h'*w']
        x = x.transpose([0, 2, 1]) #[n , h'*w', embed_dim]
        x = self.dropout(x)
        return x