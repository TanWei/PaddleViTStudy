from math import fabs
from cv2 import transpose
import paddle
import paddle.nn as nn

paddle.set_device('cpu')

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
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale # 降低序列波动
        self.softmax = nn.Softmax(-1) # 降低序列波动
        self.proj = nn.Linear(self.all_head_dim, embed_dim)  # 多头统一意见
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
        # x: [8, 16, 96]
        B, N, _ = x.shape
        qkvtmp = self.qkv(x)
        print(qkvtmp.shape) # [8, 16, 96x3]
        qkv = qkvtmp.chunk(3,-1)
        # [B, N, all_head_dim] *3
        print(type(qkv))
        # [B, N, all_head_dim] * 3  
        q, k, v = map(self.transpose_multi_head, qkv) # qkv分别应用到transpose_multi_head函数
        print(q.shape)
        # [8, 4, 16, 24]
        # q,k,v: [B, num_heads, num_patches, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True) # q * k' [8, 4, 16, 16]
# 降低序列波动
        attn = self.scale * attn
        attn = self.softmax(attn)
# 降低序列波动
        attn_weights = attn
        print(attn_weights.shape)
        #dropout
        # attn: [B, num_heads, num_patches, num_patches] 每个值对其他值的相似度矩阵
        
        out = paddle.matmul(attn, v) # softmax(scale*(q*k')) * v # [8, 4, 16, 24]
        print(out.shape)
        out = out.transpose([0,2,1,3])
        # attn: [B, num_patches, num_heads, head_dim]
        out = out.reshape([B, N, -1])
        print(out.shape) # [8, 16, 96] [B, num_patches, all_head_dim]
        out = self.proj(out) # 多头统一意见
        # [8, 16, 96]
        return out, attn_weights
def main(): 
    t = paddle.randn([8, 16, 96])
    print(t)
    # [4, 1024, 16] vit中的输入
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    out, w = model(t)
    print(out.shape)
    print(w.shape)

if __name__ == "__main__":
    main()