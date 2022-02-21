### 目标检测回顾
![图片](./Anchor4.png)
![图片](./Anchor5.png)
![图片](./Anchor5-1.png)
![图片](./NMS.png)
### DETR
![图片](./DETR整体结构.png)
![图片](./backbone.png)
![图片](./encoder.png)
![图片](./decoder.png)
![图片](./cross_attention.png)
![图片](./heads.png)
![图片](./DETR的decoder输出.png)
### 多GPU训练
原理：<br/>
每个node多个process(最多8GPU)<br/>
主process启动多个Process<br/>
每个Process独立运行训练<br/>
模型权重统一更新(最简单的一种模式)<br/>