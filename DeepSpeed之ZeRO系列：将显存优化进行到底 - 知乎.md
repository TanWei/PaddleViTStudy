前言
--

目前训练超大规模语言模型主要有两条技术路线：TPU + XLA + TensorFlow/JAX 和 GPU + PyTorch + Megatron-LM + DeepSpeed。前者由Google主导，由于TPU和自家云平台GCP深度绑定，对于非Googler来说， 只可远观而不可把玩，后者背后则有NVIDIA、Meta、MS大厂加持，社区氛围活跃，也更受到群众欢迎。

上面提到的DeepSpeed的核心是ZeRO(Zero Redundancy Optimizer)，简单来说，它是一种显存优化的数据并行(data parallelism, DP)方案。而“优化“这个话题又永无止境，在过去两年DeepSpeed团队发表了三篇ZeRO相关的论文，提出了去除冗余参数、引入CPU和内存、引入NVMe等方法，从始至终都围绕着一个目标：将显存优化进行到底。

ZeRO: 一种去除冗余的数据并行方案
-------------------

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://link.zhihu.com/?target=https%3A//sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap379.html) 发表在SC 20，DeepSpeed项目最初就是论文中ZeRO方法的官方实现。

### 背景

如今训练大模型离不开各种分布式并行策略，常用的并行策略包括：

*   数据并行（data parallelism, DP）：假设有 NN 张卡，每张卡都保存一个模型，每一次迭代（iteration/step）都将batch数据分割成 NN 个等大小的micro-batch，每张卡根据拿到的micro-batch数据独立计算梯度，然后调用AllReduce计算梯度均值，每张卡再独立进行参数更新。

```python3
# https://huggingface.co/docs/transformers/parallelism#model-parallelism
# 假设模型有三层：L0, L1, L2
# 每层有两个神经元
# 两张卡

GPU0: 
L0 | L1 | L2
---|----|---
a0 | b0 | c0
a1 | b1 | c1

GPU1:
L0 | L1 | L2
---|----|---
a0 | b0 | c0
a1 | b1 | c1
```

*   模型并行（model parallelism/tensor parallelism, MP/TP）：有的tensor/layer很大，一张卡放不下，将tensor分割成多块，一张卡存一块。

```python3
# https://huggingface.co/docs/transformers/parallelism#model-parallelism
# 假设模型有三层：L0, L1, L2
# 每层有两个神经元
# 两张卡

GPU0:
L0 | L1 | L2
---|----|---
a0 | b0 | c0

GPU1:
L0 | L1 | L2
---|----|---
a1 | b1 | c1
```

*   流水并行（pipeline parallelism, PP）：将网络按层切分，划分成多组，一张卡存一组。

```python3
# https://huggingface.co/docs/transformers/parallelism#model-parallelism
# 假设模型有8层
# 两张卡

======================  =====================
|  L0 | L1 | L2 | L3 |  | L4 | L5 | L6 | L7 |
======================  =====================
        GPU0                 GPU1

# 设想一下，当GPU0在进行（前向/后向）计算时，GPU1在干嘛？闲着
# 当GPU1在进行（前向/后向)计算时，GPU0在干嘛？闲着
# 为了防止”一卡工作，众卡围观“，实践中PP也会把batch数据分割成
# 多个micro-batch，流水线执行
```

其中数据并行由于简单易实现，应用最为广泛，当然这不表示它没有”缺点“，每张卡都存储一个模型，此时显存就成了模型规模的天花板。如果我们能减少模型训练过程中的显存占用，那不就可以训练更大的模型了？一个简单的观察是，如果有2张卡，那么系统中就存在2份模型参数，如果有4张卡，那么系统中就存在4份模型参数，如果有N张卡，系统中就存在N份模型参数，其中N-1份都是冗余的，我们有必要让每张卡都存一个完整的模型吗？系统中能否只有一个完整模型，每张卡都存 1N\\frac{1}{N} 参数，卡数越多，每张卡的显存占用越少，这样越能训练更大规模的模型。

下面就让我们看一下ZeRO是如何去除数据并行中的冗余参数。

> 注：对于LLMs动辄几百上千亿参数，实践中往往是3种并行策略混用，也就是论文中经常提到的3D parallelism，不过Google家的TPU Pod可以堆积几千张芯片，带宽也夸张，甚至不需要PP就可以训练LLMs。

### 显存去哪了

[混合精度训练](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1710.03740)（mixed precision training）和[Adam](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1412.6980)优化器基本上已经是训练语言模型的标配，我们先来简单回顾下相关概念。

Adam在SGD基础上，为每个参数梯度增加了一阶动量（momentum）和二阶动量（variance）[\[1\]](#ref_1)。

混合精度训练，字如其名，同时存在fp16和fp32两种格式的数值，其中模型参数、模型梯度都是fp16，此外还有fp32的模型参数，如果优化器是Adam，则还有fp32的momentum和variance。

![](https://pic3.zhimg.com/v2-afc7cc22a192144fcd5c6081aad24aca_b.jpg)

ZeRO将模型训练阶段，每张卡中显存内容分为两类：

1.  **模型状态**（model states）: 模型参数（fp16）、模型梯度（fp16）和Adam状态（fp32的模型参数备份，fp32的momentum和fp32的variance）。假设模型参数量 Φ\\Phi ，则共需要 2Φ+2Φ+(4Φ+4Φ+4Φ)\=4Φ+12Φ\=16Φ2\\Phi + 2\\Phi + (4\\Phi + 4\\Phi + 4\\Phi) = 4\\Phi + 12\\Phi = 16\\Phi 字节存储，可以看到，Adam状态占比 75%75\\% 。
2.  **剩余状态**（residual states）: 除了模型状态之外的显存占用，包括激活值（activation）、各种临时缓冲区（buffer）以及无法使用的显存碎片（fragmentation）。

来看一个例子，GPT-2含有1.5B个参数，如果用fp16格式，只需要3GB显存，但是模型状态实际上需要耗费24GB！相比之下，激活值可以用[activation checkpointing](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1604.06174.pdf)来大大减少，所以模型状态就成了头号显存杀手，它也是ZeRO的重点优化对象。而其中Adam状态又是第一个要被优化的。

针对模型状态的存储优化（去除冗余），ZeRO使用的方法是分片（partition），即每张卡只存 1N\\frac{1}{N} 的模型状态量，这样系统内只维护一份模型状态。

*   首先进行分片操作的是模型状态中的Adam，也就是下图中的 PosP\_{os} ，这里os指的是optimizer states。模型参数（parameters）和梯度（gradients）仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 4Φ+12ΦN4\\Phi + \\frac{12\\Phi}{N} 字节，当 NN 比较大时，趋向于 4ΦB4\\Phi B ，也就是原来 16ΦB16\\Phi B 的 14\\frac{1}{4} 。
*   如果继续对模型梯度进行分片，也就是下图中的 Pos+gP\_{os+g} ，模型参数仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 2Φ+2Φ+12ΦN2\\Phi + \\frac{2\\Phi + 12\\Phi}{N} 字节，当 NN 比较大时，趋向于 2ΦB2\\Phi B ，也即是原来 16ΦB16\\Phi B 的 18\\frac{1}{8} 。
*   如果继续对模型参数进行分片，也就是下图中的 Pos+g+pP\_{os+g+p} ，此时每张卡的模型状态所需显存是 16ΦN\\frac{16\\Phi}{N} 字节，当 NN 比较大时，趋向于 00 。

下图中Memory Consumption 第二列给出了一个示例： K\=12,Φ\=7.5B,N\=64K=12, \\Phi=7.5B, N=64 ，可以看到显存优化相当明显。

在DeepSpeed中， PosP\_{os} 对应ZeRO-1， Pos+gP\_{os+g} 对应ZeRO-2， Pos+g+pP\_{os+g+p} 对应ZeRO-3，一般使用ZeRO-1就足够了。

![](https://pic1.zhimg.com/v2-51a6660e548e05b0f06c56d5fab9f270_b.jpg)

模型状态分区

解决了模型状态，再来看剩余状态，也就是激活值（activation）、临时缓冲区（buffer）以及显存碎片（fragmentation）。

*   激活值同样使用分片方法，并且配合checkpointing
*   模型训练过程中经常会创建一些大小不等的临时缓冲区，比如对梯度进行AllReduce啥的，解决办法就是预先创建一个固定的缓冲区，训练过程中不再动态创建，如果要传输的数据较小，则多组数据bucket后再一次性传输，提高效率
*   显存出现碎片的一大原因是时候gradient checkpointing后，不断地创建和销毁那些不保存的激活值，解决方法是预先分配一块连续的显存，将常驻显存的模型状态和checkpointed activation存在里面，剩余显存用于动态创建和销毁discarded activation

上面的方案对于显存优化看起来很有效，但是还有一个疑问，相比于传统的数据并行，ZeRO是否会带来额外的通信（communication）成本？特别是在大规模训练场景下，通信本来就容易成为瓶颈，如果ZeRO舍本逐末，我想大家是不能接受的。

![](https://pic3.zhimg.com/v2-613fb8a502f2c8149cfd313b2f5743a2_b.jpg)

### 通信数据量分析

在分析之前，我们先回顾下常用的集合通信（collective communication）函数[Collective Operations](https://link.zhihu.com/?target=https%3A//docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

*   Reduce

![](https://pic4.zhimg.com/v2-51ff40433d988c7696624e44b3a8f75f_b.jpg)

*   Broadcast

![](https://pic2.zhimg.com/v2-e231dae32e242c83684fdeda0d235dc9_b.jpg)

*   AllReduce，这个操作是数据并行的通信基础，建议大家读一下袁老师写的

![](https://pic1.zhimg.com/v2-5a624583d1381b0e66a18a6f0d0ed9b4_b.jpg)

*   AllGather

![](https://pic3.zhimg.com/v2-6cbc23fd16b666f2ae3c9bc295722b52_b.jpg)

*   ReduceScatter

![](https://pic3.zhimg.com/v2-3cd09e10c4487e34dd653b0cc889978a_b.jpg)

下面我们就分析下通信数据量，先说结论， P\_{os} 和 P\_{os+g} 的通信量和传统数据并行相同，P\_{os+g+p} 会增加通信量。

传统数据数据并行在每一步（step/iteration）计算梯度后，需要进行一次AllReduce操作来计算梯度均值，目前常用的是Ring AllReduce，分为ReduceScatter和AllGather两步，每张卡的通信数据量（发送+接受）近似为 2\\Phi [\[2\]](#ref_2)。

我们直接分析 P\_{os+g} ，每张卡只存储 \\frac{1}{N} 的优化器状态和梯度，对于 gpu\_{0} 来说，为了计算它这 \\frac{1}{N} 梯度的均值，需要进行一次Reduce操作，通信数据量是 \\frac{1}{N} \\Phi \\cdot N=\\Phi ，然后其余显卡则不需要保存这部分梯度值了。实现中使用了bucket策略，保证 \\frac{1}{N} 的梯度每张卡只发送一次。

> 这里还要注意一点，假如模型最后两层的梯度落在 gpu\_0 ，为了节省显存，其他卡将这两层梯度删除，怎么计算倒数第三层的梯度呢？还是因为用了bucket，其他卡可以将梯度发送和计算倒数第三层梯度同时进行，当二者都结束，就可以放心将后两层梯度删除了。

当 gpu\_{0} 计算好梯度均值后，就可以更新局部的优化器状态（包括 \\frac{1}{N}\\Phi 的参数），当反向传播过程结束，进行一次Gather操作，更新 (1-\\frac{1}{N}) \\Phi 的模型参数，通信数据量是 \\frac{1}{N} \\Phi \\cdot N=\\Phi 。

从全局来看，相当于用Reduce-Scatter和AllGather两步，和数据并行一致。

![](https://pic2.zhimg.com/v2-c9bb5da2df6a6e14139d1be1372a7825_b.jpg)

P\_{os+g+p} 使得每张卡只存了 \\frac{1}{N} 的参数，不管是在前向计算还是反向传播，都涉及一次Broadcast操作。

实验方面，“仅”使用400张V100就能训练170B的模型，是Megatron-LM的8倍（因为只用了 P\_{os+g} ）。

ZeRO-Offload: 让人人都能训练得起大模型
--------------------------

[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/atc21/presentation/ren-jie)发表在ATC 21，一作是来自UC Merced的[Jie Ren](https://link.zhihu.com/?target=https%3A//jren73.github.io/)，博士期间的研究方向是 Memory Management on Heterogeneous Memory Systems for Machine Learning and HPC. 所以看到这个题目也就不奇怪了。

### 背景

ZeRO说到底是一种数据并行方案，可是很多人只有几张甚至一张卡，难道我们就没有梦想，我们就不想训练大模型吗:(

![](https://pic2.zhimg.com/v2-ea97e48640fcb70860fb40f5f45e1321_b.jpg)

一张卡训不了大模型，根因是显存不足，ZeRO-Offload的想法很简单：显存不足，内存来补。

直接看下效果，在单张V100的情况下，用PyTorch能训练1.4B的模型，吞吐量是30TFLOPS，有了ZeRO-Offload加持，可以训练10B的模型，并且吞吐量40TFLOPS。这么好的效果能不能扩展到多卡上面呢，能啊，比如只用一台[DGX-2](https://link.zhihu.com/?target=https%3A//www.nvidia.com/en-us/data-center/dgx-2/)服务器，可以训练70B的模型，是原来只用模型并行的4.5倍，在128张显卡的实验上基本也是线性加速，此外还可以与模型并行配合，快乐加倍:)

![](https://pic3.zhimg.com/v2-d0af9d03ba9149b360426b06f56a481a_b.jpg)

相比于昂贵的显存，内存廉价多了，能不能在模型训练过程中结合内存呢？其实已经有很多工作了，但是他们几乎只聚焦在内存上面，没有用到CPU计算，更没有考虑多卡的场景。ZeRO-Offload则将训练阶段的某些模型状态下放（offload）到内存以及CPU计算。

> 注：ZeRO-Offload没有涉及剩余状态（比如激活值）的下放，因为在Transformer LM场景中，他比模型状态占用的显存小。

ZeRO-Offload要做的事情我们清楚了，那么如何设计高效的offload策略呢？

### Offload策略

ZeRO-Offload并不希望为了最小化显存占用而让系统的计算效率下降，否则的话，我们只用CPU和内存不就得了。但是将部分GPU的计算和存储下放到CPU和内存，必然涉及CPU和GPU之间的通信增加，**不能让通信成为瓶颈**，此外GPU的计算效率相比于CPU也是数量级上的优势，也**不能让CPU参与过多计算**，避免成为系统瓶颈，只有前两条满足的前提下，再考虑**最小化显存的占用**。

为了找到最优的offload策略，作者将模型训练过程看作数据流图（data-flow graph）。

*   圆形节点表示模型状态，比如参数、梯度和优化器状态
*   矩形节点表示计算操作，比如前向计算、后向计算和参数更新
*   边表示数据流向

下图是某一层的一次迭代过程（iteration/step），使用了混合精读训练，前向计算（FWD）需要用到上一次的激活值（activation）和本层的参数（parameter），反向传播（BWD）也需要用到激活值和参数计算梯度，

![](https://pic2.zhimg.com/v2-80215e2003776eea0a82831e24aabb05_b.jpg)

如果用Adam优化器进行参数更新（Param update），流程如下：

![](https://pic1.zhimg.com/v2-b7052a7c180395b94d7be6a23b00df1c_b.jpg)

下面我们为边添加权重，物理含义是数据量大小（单位是字节），假设模型参数量是 M ，在混合精度训练的前提下，边的权重要么是2M（fp16），要么是4M（fp32），

![](https://pic2.zhimg.com/v2-3d3a9ce68a740dfd5f0210ab2f8aa03d_b.jpg)

**我们现在要做的就是沿着边把数据流图切分为两部分，分布对应GPU和CPU，**计算节点（矩形节点）落在哪个设备，哪个设备就执行计算，数据节点（圆形）落在哪个设备，哪个设备就负责存储，将被切分的边权重加起来，就是CPU和GPU的通信数据量。

ZeRO-Offload的切分思路是：

图中有四个计算类节点：FWD、BWD、Param update和float2half，前两个计算复杂度大致是 O(MB) ， B 是batch size，后两个计算复杂度是 O(M) 。为了不降低计算效率，将前两个节点放在GPU，后两个节点不但计算量小还需要和Adam状态打交道，所以放在CPU上，Adam状态自然也放在内存中，为了简化数据图，将前两个节点融合成一个节点FWD-BWD Super Node，将后两个节点融合成一个节点Update Super Node。如下图右边所示，沿着gradient 16和parameter 16两条边切分。

![](https://pic4.zhimg.com/v2-b9f59b045c1629983fb3e94e692bc457_b.jpg)

现在的计算流程是，在GPU上面进行前向和后向计算，将梯度传给CPU，进行参数更新，再将更新后的参数传给GPU。为了提高效率，可以将计算和通信并行起来，GPU在反向传播阶段，可以待梯度值填满bucket后，一遍计算新的梯度一遍将bucket传输给CPU，当反向传播结束，CPU基本上已经有最新的梯度值了，同样的，CPU在参数更新时也同步将已经计算好的参数传给GPU，如下图所示。

![](https://pic3.zhimg.com/v2-57ab3768637af499bda329bbecb1ce1a_b.jpg)

到目前为止，说的都是单卡场景，卡多的人表示。。。

![](https://pic3.zhimg.com/v2-4fccd9abae8cc8b197435efab9ae8dce_b.jpg)

### 扩展性

在多卡场景，ZeRO-Offload利用了ZeRO-2，回忆下ZeRO-2是将Adam状态和梯度进行了分片，每张卡只保存 \\frac{1}{N} ，而ZeRO-Offload做的同样是将这 \\frac{1}{N} 的Adam状态和梯度都offload到内存，在CPU上进行参数更新。

> 注意：在多卡场景，利用CPU多核并行计算，每张卡至少对应一个CPU进程，由这个进程负责进行局部参数更新。

并且CPU和GPU的通信量和 N 无关，因为传输的是fp16 gradient和fp16 parameter，总的传输量是固定的，由于利用多核并行计算，每个CPU进程只负责 \\frac{1}{N} 的计算，反而随着卡数增加节省了CPU计算时间。

![](https://pic4.zhimg.com/v2-120a8123fc745cf64ecd69cd745d5017_b.jpg)

直接看下效果吧，

![](https://pic1.zhimg.com/v2-fe50a06119ca19954e6fee767cafb028_b.jpg)

但是有一个问题，当batch size很小时，GPU上每个micro-batch计算很快，此时CPU计算时长会成为训练瓶颈，一种方法是让CPU在某个节点更新参数时延迟一步，后面就可以让GPU和CPU并行起来。

前N-1步，不进行延迟，避免早期训练不稳定，模型无法收敛，在第N步，CPU拿到GPU计算的梯度后，不更新参数，相当于GPU空算了一步，到N+1步，CPU开始根据刚才拿到的第N步的梯度计算，此时GPU开始算N+1步的梯度。

![](https://pic1.zhimg.com/v2-d9aa4eba2b96960e13364a7f004462e0_b.jpg)

当然这样会有一个问题，用来更新参数的梯度并不是根据当前模型状态计算得到的，论文的实验结果表明暂未发现对收敛和效果产生影响。

![](https://pic2.zhimg.com/v2-892a908b165cb7af1466565530fe00a5_b.jpg)

ZeRO-Infinity: 利用NVMe打破GPU显存墙
-----------------------------

[ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.07857.pdf) 发表在SC 21，同样是进行offload，ZeRO-Offload更侧重单卡场景，而ZeRO-Infinity则是典型的工业界风格，奔着极大规模训练去了。

### 背景

从GPT-1到GPT-3，两年时间内模型参数0.1B增加到175B，而同期，NVIDIA交出的成绩单是从V100的32GB显存增加A100的80GB，显然，显寸的提升速度远远赶不上模型模型增长的速度，这就是内存墙问题

![](https://pic1.zhimg.com/v2-81a75cad34d2deaf7662f0acac18acf0_b.jpg)

参考
--

1.  [^](#ref_1_0)An overview of gradient descent optimization algorithms [https://ruder.io/optimizing-gradient-descent/index.html#adam](https://ruder.io/optimizing-gradient-descent/index.html#adam)
2.  [^](#ref_2_0)手把手推导Ring All-reduce的数学性质，OneFlow [https://zhuanlan.zhihu.com/p/504957661](https://zhuanlan.zhihu.com/p/504957661)