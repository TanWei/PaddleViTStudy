from re import X
from turtle import forward
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image


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
    
def main():
    img = Image.open('D:/githubSpace/mine/PaddleViTStudy/src/vitbase./724.jpg')
    img = np.array(img)
    for i in range(28):
        for j in range(28):
            print(f'{img[i,j]:03} ', end='')
        print()
    sample = paddle.to_tensor(img, dtype='float32')
    sample = sample.reshape([1,1,28,28])
    #2.patch embedding
    patch_embed = PatchEmbedding(image_size=128, patch_size=7, in_channels=1, embed_dim=96)
    out = patch_embed(sample)
    #3.MLP
    
if __name__ == "__main__":
    main()