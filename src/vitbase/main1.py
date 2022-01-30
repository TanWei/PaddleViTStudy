import paddle
import paddle.nn as nn
from PIL import Image
import numpy as np

paddle.set_device('cpu')
def main():
    # 1. Create a tensor
    # t1 = paddle.zeros([3,3])
    # print(t1)
    # 2. Create a random tensor
    # t2 = paddle.randn([5,3])
    # print(t2)
    # 3. Create a tensor from Image ./724/jpg 28x28
    img =   np.array(Image.open('./724.jpg'))
    for i in range(28):
        for j in range(28):
            print(f'{img[i,j]:03} ', end='')
        print()
    # 4. print tensor tpye and dtype of tensor
    
    # 5. transpose imagetensor
    # 6.Reshape a random int rwnsoe from  5x5 to 25
    # 7.Unsqueeze a random int tensor from 5x5 to 5x5x1
    # 8.chunk a random int tensor from 5x15 to 5x5, 5x5 and 5x5

if __name__ == "__main__":
    main()