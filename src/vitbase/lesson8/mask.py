import paddle
import paddle.nn as nn
from PIL import Image
paddle.set_device('cpu')

def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H//window_size, window_size, W//window_size, window_size, C]) # [4, 8, 7, 8, 7, 96]
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B, h//ws, w//ws, ws, ws, c]
    print(x.shape) # [4, 8, 8, 7, 7, 96]
    x = x.reshape([-1, window_size, window_size, C]) # [256, 7, 7, 96]
    # [B * num_patches, ws, ws, c]
    return x

def generate_mask(window_size=4, shift_size=2, input_relolution=(8, 8)):
    H, W = input_relolution
    img_mask = paddle.zeros([1, H, W, 1]) #[1, 8, 8, 1]
    h_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)] #
    
    w_slices = [slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)]
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
           img_mask[:, h_slice, w_slice, :] = cnt
           cnt+=1
    # [1, 8, 8, 1]
    window_mask = windows_partition(img_mask, window_size = window_size)
    # [4, 4, 4, 1]
    window_mask = window_mask.reshape([-1, window_size * window_size]) # 
    # [4, 16]
    attn_mask = window_mask.unsqueeze(1) - window_mask.unsqueeze(2)
    # [n, 1, ws*ws] - [n, ws*ws, 1]  Broadcasting 行向量减列向量：把两个矩阵补全为h*w的矩阵，然后按照元素相减
    attn_mask = paddle.where(attn_mask != 0, 
                             paddle.ones_like(attn_mask) * 255,
                             paddle.zeros_like(attn_mask))
    return attn_mask
def main():
    mask = generate_mask()
    print(mask.shape)
    mask = mask.cpu().numpy().astype('uint8')
    for i in range(4):
        for j in range(16):
            for k in range(16):
                print(mask[i, j, k], end='\t')
            print()
        im = Image.fromarray(mask[i, :, :])
        im.save(f'{i}.png')
        print()
        print()
    print()

if __name__ == "__main__":
    main()