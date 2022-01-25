#!/usr/bin/env python
# coding: utf-8

# In[73]:


import os, os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
from torchvision import models

import numpy as np 

import tvm
from tvm import relay
from tvm.contrib import graph_executor


# In[3]:


target = "llvm"
os.makedirs("bin", exist_ok=True)
lib_path = f"bin/tmp-{target}.so"


bs = 1
rs = 224 
input_shape = [bs, 3, rs, rs]
input_data = torch.randn(input_shape)
input_name = "input0"
shape_list = [(input_name, input_data.shape)]

model = models.mobilenet_v2(pretrained=True)
model = model.eval()


# In[4]:


print(f"not found library, export to {lib_path}")
scripted_model = torch.jit.trace(model, input_data).eval()
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, use_parser_friendly_name=True)
print("Pass 1")
# target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
os.makedirs("bin", exist_ok=True)
lib.export_library(lib_path)


# In[5]:


dev = tvm.cpu(0)
data = np.random.uniform(0, 1, size=input_data.shape).astype("float32")
th_data = torch.from_numpy(data)

th_out = model(th_data)

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, data)
module.run()
tvm_output = module.get_output(0)

mean_diff = th_out.mean().detach().numpy() - np.mean(tvm_output.numpy())
var_diff = th_out.var().detach().numpy() - np.var(tvm_output.numpy())
print(f"{mean_diff:.4f}, {var_diff:.4f}")


# In[52]:


bin_params = tvm.runtime.save_param_dict(params);
r_params = tvm.runtime.load_param_dict(bin_params)
for k, v in params.items():
    break
r_params["features_2_conv_0_1_bias"]


# In[77]:


model = models.mobilenet_v2(pretrained=True)
model = model.eval()
old_model = deepcopy(model)

from torchvision.models.mobilenetv2 import ConvBNActivation, ConvNormActivation, InvertedResidual
def disable_dropout_bn(module):
    module_output = module
    if isinstance(module, (nn.Dropout)):
        print("removing dropout")
        module_output = nn.Identity()
    if isinstance(module, (ConvNormActivation, )):
        print("[ConvNormActivation]\tfusing BN and dropout")
        idx = 0
        conv = module[idx]
        bn = module[idx+1]
        channels = bn.weight.shape[0]
        invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight *                 bn.weight[:, None, None, None] *                 invstd[:, None, None, None]
        if conv.bias is not None:
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
        module[idx+1] = nn.Identity()
    if isinstance(module, (InvertedResidual, )):
        print("[InvertedResidual]\tfusing BN and dropout")
        idx = -2
        conv = module.conv[idx]
        bn = module.conv[idx+1]
        channels = bn.weight.shape[0]
        invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight *                 bn.weight[:, None, None, None] *                 invstd[:, None, None, None]
        if conv.bias is not None:
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
        module.conv[idx+1] = nn.Identity()
    for name, child in module.named_children():
        module_output.add_module(name, disable_dropout_bn(child))
    del module
    return module_output

model = disable_dropout_bn(model)


# In[90]:


data = torch.randn(1, 3, 224, 224)


# In[94]:


out1 = model(data)
out2 = old_model(data)
(out1- out2).mean(), (out1- out2).std()


# In[ ]:




