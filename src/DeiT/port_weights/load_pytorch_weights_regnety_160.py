#   Copyright (c) 2021 PPViT Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
import paddle
import torch
import timm
from regnet import *

def print_model_named_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)

def print_model_named_buffers(model):
    for name, buff in model.named_buffers():
        print(name, buff.shape)

def torch_to_paddle_mapping():
    mapping = [
        ('stem.conv.weight', 'stem.0.weight'),
        ('stem.bn', 'stem.1'),
    ]

    depths = [2, 4, 11, 1]
    for idx in range(len(depths)):
        for block_idx in range(depths[idx]):
            th_prefix = f's{idx+1}.b{block_idx+1}'
            pp_prefix = f'stages.{idx}.blocks.{block_idx}'

            layer_mapping = [
                (f'{th_prefix}.conv1.conv', f'{pp_prefix}.conv1'),
                (f'{th_prefix}.conv1.bn', f'{pp_prefix}.bn1'),
                (f'{th_prefix}.conv2.conv', f'{pp_prefix}.conv2'),
                (f'{th_prefix}.conv2.bn', f'{pp_prefix}.bn2'),
                (f'{th_prefix}.se.fc1', f'{pp_prefix}.se.conv1_1x1'),
                (f'{th_prefix}.se.fc2', f'{pp_prefix}.se.conv2_1x1'),
                (f'{th_prefix}.downsample.conv', f'{pp_prefix}.downsample.conv1x1'),
                (f'{th_prefix}.downsample.bn', f'{pp_prefix}.downsample.bn'),
                (f'{th_prefix}.conv3.conv', f'{pp_prefix}.conv3'),
                (f'{th_prefix}.conv3.bn', f'{pp_prefix}.bn3'),
            ]
            mapping.extend(layer_mapping)

    head_mapping = [
        ('head.fc', 'head.2'),
    ]
    mapping.extend(head_mapping)

    return mapping



def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
        if len(value.shape) == 2:
            value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)

    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in paddle_model.named_buffers():
        pd_params[name] = param

    for name, param in torch_model.named_parameters():
        th_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param

    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()
    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            _set_value(th_name, pd_name)
        else: # weight & bias
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_mean' in th_params.keys():
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model


def main():

    paddle.set_device('cpu')
    paddle_model = build_regnet()
    paddle_model.eval()

    print_model_named_params(paddle_model)
    print('--------------')
    print_model_named_buffers(paddle_model)
    print('----------------------------------')

    device = torch.device('cpu')
    torch_model = timm.create_model('regnety_160', pretrained=True)
    torch_model = torch_model.to(device)
    torch_model.eval()

    print_model_named_params(torch_model)
    print('--------------')
    print_model_named_buffers(torch_model)
    print('----------------------------------')

    #return

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn(2, 3, 288, 288).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)

    print(torch_model)
    out_torch = torch_model(x_torch)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:100])
    print(out_paddle[0, 0:100])
    assert np.allclose(out_torch, out_paddle, atol = 1e-5)
    
    # save weights for paddle model
    model_path = os.path.join('./regnety_160.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()
