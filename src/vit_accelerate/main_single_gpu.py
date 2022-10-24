import paddle
import paddle.nn as nn
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter
import requests
from vit import ViT2
import paddle.profiler as profiler
import time
import numpy as np
import random
from vit111 import build_vit
from config import get_config


train_time = 0 # 记录总训练时长

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=10, profiler=None):
    print(f'----- Training Epoch [{epoch}/{total_epoch}]:')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    # TODO AMP 提速
    amp = False
    if amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        start_time = time.time() # 记录开始训练时刻
        if not amp:
            # image dimension requrements ([4, 3, 224, 224])
            print("image shape", paddle.to_tensor(image).shape)
            out = model(image)
            print("out shape")
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
        else:
            with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                out = model(image) # 前向计算（9层Linear网络，每层由matmul、add算子组成）
                loss = criterion(out, label) # loss计算
            scaled = scaler.scale(loss) # loss缩放，乘以系数loss_scaling
            scaled.backward()           # 反向传播
            
            scaler.minimize(optimizer, scaled) #等同于下面连个函数一起用
            # scaler.step(optimizer)      # 更新参数（参数梯度先除系数loss_scaling再更新参数）
            # scaler.update()             # 基于动态loss_scaling策略更新loss_scaling系数
            optimizer.clear_grad(set_to_zero=False)

        # train_time += time.time() - start_time
        if profiler != None:
            profiler.step()
            if batch_id == 19:
                profiler.stop()
                exit()

        pred = nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)
        if batch_id > 0 and batch_id % report_freq == 0:
            print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')

    print(f'----- Epoch[{epoch}/{total_epoch}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')


def validate(model, dataloader, critertion):
    print(f'----- Validation')
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()
    for batch_id, data in enumerate(dataloader):
        image = data[0]
        label = data[1]

        out = model(image)
        loss = criterion(out, label)

        pred = nn.functional.softmax(out, axis=1)
        acc1 = paddle.metric.accuracy(pred, label.unsqueeze(-1))

        batch_size = image.shape[0]
        loss_meter.update(loss.cpu().numpy()[0], batch_size)
        acc_meter.update(acc1.cpu().numpy()[0], batch_size)
        if batch_id > 0 and batch_id % report_freq == 0:
            print(f'----- Batch[{batch_id}/{len(dataloader)}], Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')
    print(f'----- Validation Loss: {loss_meter.avg:.4}, Acc@1: {acc_meter.avg:.2}')

def train():
    device = paddle.device.get_device()
    print("============aaaaaaaaa==============")
    total_epoch = 200
    batch_size = 16
    # 1
    model  = ViT2()
    # config = get_config()
    # model = build_vit(config)
    # 2
    train_dataset = get_dataset(mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size, mode='train')
    val_dataset = get_dataset(mode='test')
    val_dataloader = get_dataloader(val_dataset, batch_size, mode='test')
    # 3
    criterion = nn.CrossEntropyLoss()
    # 4 学习速度衰减
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)
    # 5 优化器 简单使用Momentum
    optimizer = paddle.optimizer.Momentum(learning_rate=scheduler,
                                          parameters=model.parameters(),
                                          momentum=0.9,
                                          weight_decay=5e-4)
    # 6 ===统计
    def my_on_trace_ready(prof): # 定义回调函数，性能分析器结束采集数据时会被调用
      callback = profiler.export_chrome_tracing('./profiler_demo') # 创建导出性能数据到profiler_demo文件夹的回调函数
      callback(prof)  # 执行该导出函数
      prof.summary(sorted_by=profiler.SortedKeys.GPUTotal) # 打印表单，按GPUTotal排序表单项

    # 记录3到14个batch
    p = profiler.Profiler(scheduler = [3,14], on_trace_ready=my_on_trace_ready, timer_only=True) # 初始化Profiler对象

    p.start() # 性能分析器进入第0个step
    #===
    # 7 load pretrained model or resume model TODO
    
    # 8 开始训练
    print("start training...")
    for epoch in range(1, total_epoch+1):
        # model, dataloader, criterion, optimizer, epoch, total_epoch, report_freq=10, profiler=None
        train_one_epoch(model,
                        train_dataloader,
                        criterion,
                        optimizer,
                        epoch,
                        total_epoch,
                        10,
                        p)
        scheduler.step()
        #validate(model, val_dataloader, criterion) #统计精确度

def main():
    paddle.device.set_device('gpu')
    paddle.distributed.init_parallel_env()

    world_size = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    # seed = config.SEED + local_rank
    seed = local_rank
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    train()


if __name__ == "__main__":
    #r = requests.get('https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz', stream=True)
    main()
