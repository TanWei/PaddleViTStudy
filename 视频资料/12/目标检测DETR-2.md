预测结果的box set是无序的，如何对应ground_truth的box？<br/>
![图片](./DETR_loss.png)
![图片](./DETR_matcher.png)
L不是loss，是matching cost<br/>
$L_{match}$中的p是class的概率，$-1_{\{c_i≠φ\}}$表示如果不是是背景的时候值为-1，否则是0.<br/>
![图片](./DETR_matcher_2.png)
![图片](./DETR_loss_2.png)
loss_giou重叠度，loss_bbox比较像box的位置（中心点和b的四个值都要算）<br/>