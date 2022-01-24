import torch
import torch.nn as nn
import torchvision
from torchvision import models

import numpy as np 

import tvm
from tvm import relay
from tvm.contrib import graph_executor

model = models.mobilenet_v2(pretrained=True)
model = model.eval()

bs = 1 
rs = 224 

input_shape = [bs, 3, rs, rs]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

input_name = "input0"
shape_list = [(input_name, input_data.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, use_parser_friendly_name=True)
print("Pass 1")

# target = tvm.target.Target("llvm", host="llvm")
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.cpu(0)
# dev = tvm.cuda()
data = np.random.uniform(0, 1, size=input_data.shape).astype("float32")
th_data = torch.from_numpy(data)

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, data)
module.run()
tvm_output = module.get_output(0)

print(th_data.mean(), np.nanmean(tvm_output.numpy()))
