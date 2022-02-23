### MobileViT
![图片](./how_conv_is_computed.png)
![图片](./整体结构.png)
###重点MV2 block 和 MViT block
分组卷积：https://www.jianshu.com/p/a936b7bc54e3/ (group convolution (分组卷积)详解)
![图片](./MV2_block.png)
![图片](./MobileViT_block.png)
<b>论文主要部分如何将conv后的tensor输入到transformer里面</b>
![图片](./如何将conv后的tensor输入到transformer里面.png)

### 数据加载
python for循环原理
![图片](./for原理.png)