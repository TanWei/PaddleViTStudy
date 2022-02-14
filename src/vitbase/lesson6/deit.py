# ViT Online Class
# Author: Dr. Zhu
# Project: PaddleViT (https://github.com/BR-IDL/PaddleViT)
# 2021.11
from turtle import forward
import paddle
import paddle.nn as nn
import numpy as np
from PIL import Image
from torch import dropout

paddle.set_device('cpu')

class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
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


class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        n_patches = (image_size // patch_size) * (image_size // patch_size)
        # image_size=224, patch_size=7, in_channels=3, embed_dim=16 
        self.patch_embedding = nn.Conv2D(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                         bias_attr=False)
        self.dropout = nn.Dropout(dropout)
        # 第四节课新加的内容
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.)
        )
        
        self.position_embeding = paddle.create_parameter(
            shape=[1, n_patches+2, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02)
        )

        self.distill_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02)
        )
    def forward(self, x):    
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        distill_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        
        # [n, c, h, w] 
        x = self.patch_embedding(x) # [n, c', h', w'] c'=embed_dim
        x = x.flatten(2) # [n, c', h'*w']  [4, 16, 32*32] h'*w' = num_patches
        x = x.transpose([0, 2, 1]) # [n, h'*w', c'] [4, 1024, 16]
        
        x = paddle.concat([cls_tokens, distill_tokens, x], axis=1)

        # [1, 1025, 16]
        x = x + self.position_embeding
        # lesson 4 end
        
        
        x = self.dropout(x)
        return x


# class Attention(nn.Layer):
#     def  __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x
class Attention(nn.Layer):
    def __init__(self, embed_dim, 
                 num_heads, 
                 qkv_bias=False, 
                 qk_scale=None, 
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads # 4
        self.embed_dim = embed_dim # 96
        self.head_dim = int(embed_dim / num_heads) #24
        self.all_head_dim = self.head_dim * num_heads #96
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None)
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)
    def transpose_multi_head(self, x):
        print(x.shape) # [8, 16, 96]
        # x: [B, N, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        print(x.shape) # [8, 16, 4, 24]
        # x: [B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x: [B, num_heads, num_patches(N), head_dim] [8, ,4, 16， 24]
        return x
    
    def forward(self, x):
        B, N, _ = x.shape
        qkvtmp = self.qkv(x)
        print(qkvtmp.shape) # [8, 16, 96x3]
        qkv = qkvtmp.chunk(3,-1)
        print(type(qkv))
        # [B, N, all_head_dim] * 3  
        q, k, v = map(self.transpose_multi_head, qkv)
        print(q.shape)
        # [8, 4, 16, 24]
        # q,k,v: [B, num_heads, num_patches, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True) # q * k'
        attn = self.scale * attn
        attn = self.softmax(attn)
        attn_weights = attn
        print(attn_weights.shape)
        #dropout
        # attn: [B, num_heads, num_patches, num_patches] 每个值对其他值的相似度矩阵

        out = paddle.matmul(attn, v) # softmax(scale*(q*k')) * v
        print(out.shape)
        out = out.transpose([0,2,1,3])
        # attn: [B, num_patches, num_heads, head_dim]
        out = out.reshape([B, N, -1])
        print(out.shape)
        out = self.proj(out)
        return out


class EncoderLayer(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        # self.attn = Attention()
        self.attn = Attention(embed_dim, num_heads=4, qkv_bias=False, qk_scale=None)
         
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)

    def forward(self, x):
        h = x 
        x = self.attn_norm(x)
        print(x.shape)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x


class Encoder(nn.Layer):
    def __init__(self, embed_dim, depth):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim)
            layer_list.append(encoder_layer)
        
        self.layers = nn.LayerList(layer_list)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

class DeiT(nn.Layer):
    def __init__(self,
                image_size=224,
                patch_size=16,
                in_channels=3,
                num_classes=1000,
                embed_dim=768,
                depth=3,
                num_headers=8,
                mlp_ratio=4,
                qkv_bias=True,
                dropout=0.,
                attention_dropout=0.,
                droppath=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(224, 16, 3, 768)
        self.encoder = Encoder(embed_dim, depth)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_distill = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x, x_distill = self.encoder(x)
        x = self.head(x)
        x_distill = self.head_distill(x_distill)
        if self.training:
            return x, x_distill
        else:
            return (x + x_distill) / 2 #  软蒸馏
        
def main():
    model = DeiT()
    print(model)
    paddle.summary(model, (4, 3, 224, 224))

if __name__ == "__main__":
    main()
