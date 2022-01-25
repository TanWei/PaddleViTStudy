from sched import scheduler
import paddle
import paddle.nn as nn
from resnet18 import ResNet18
from dataset import get_dataset
from dataset import get_dataloader
from utils import AverageMeter

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, total_cpoch, report_freq=20):
    pass
def main():
    total_epoch = 200
    batch_size = 16

    model = ResNet18(num_classes=10)
    train_dataset = get_dataset(mode = 'train')
    train_dataloader = get_dataloader(dataset_train, mode='train', batch_size)
    val_dataset = get_dataset(model='test')
    val_dataloader = get_dataloader(dataset_val, mode='test', batch_size)
    criterion = nn.CrossEntropyLoss()

    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(0.02, total_epoch)
    for epoch in range(1, total_epoch):
        train_one_epoch(model, dataloader, criterion, )
    
if __name__=="__main__":
    main()