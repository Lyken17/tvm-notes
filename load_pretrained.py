#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from torchvision import models

import numpy as np 

import tvm
from tvm import relay, te
from tvm.contrib import graph_executor


# In[4]:


SEMVER = '#[version = "0.0.5"]\n'

width = 1.0
bs = 1 
rs = 224
input_shape = [bs, 3, rs, rs]
dev = tvm.cpu(0)

target = "llvm"
os.makedirs("bin", exist_ok=True)


# In[5]:


code = '''
fn (%input0: Tensor[(1, 3, 224, 224), float32], %features_0_0_weight: Tensor[(32, 3, 3, 3), float32], %features_0_0_bias: Tensor[(32), float32], %features_1_conv_0_0_weight: Tensor[(32, 1, 3, 3), float32]) 
    -> (Tensor[(1, 32, 112, 112), float32], Tensor[(1, 32, 112, 112), float32])
{
  %0 = nn.conv2d(%input0, %features_0_0_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 = nn.bias_add(%0, %features_0_0_bias) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  (%0, %1)
}
'''

mod = tvm.parser.parse_expr(SEMVER + code)
new_mod = tvm.IRModule.from_expr(mod)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=None)

module = graph_executor.GraphModule(lib["default"](dev))
print(module.get_num_outputs())

fpath = "bin/test-456.so"
if osp.exists(fpath):
    print("removed")
    os.remove(fpath)

lib.export_library(fpath)
new_lib = tvm.runtime.load_module(fpath)
module = graph_executor.GraphModule(new_lib["default"](dev))
print(module.get_num_outputs())


# In[6]:


code = '''
fn (%input0: Tensor[(1, 3, 224, 224), float32], %features_0_0_weight: Tensor[(32, 3, 3, 3), float32], %features_0_0_bias: Tensor[(32), float32], %features_1_conv_0_0_weight: Tensor[(32, 1, 3, 3), float32]) 
    -> Tensor[(1, 32, 112, 112), float32]
{
  %0 = nn.conv2d(%input0, %features_0_0_weight, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 = nn.bias_add(%0, %features_0_0_bias) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 
}
'''

mod = tvm.parser.parse_expr(SEMVER + code)
new_mod = tvm.IRModule.from_expr(mod)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=None)

module = graph_executor.GraphModule(lib["default"](dev))
print(module.get_num_outputs())

fpath = "bin/test-123.so"
if osp.exists(fpath):
    print("removed")
    os.remove(fpath)

lib.export_library(fpath)
new_lib = tvm.runtime.load_module(fpath)
module = graph_executor.GraphModule(new_lib["default"](dev))
print(module.get_num_outputs())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




