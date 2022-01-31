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
from copy import deepcopy

import tvm
from tvm import relay, te
from tvm.contrib import graph_executor

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


dev = tvm.cpu(0)
x = relay.var("x", shape=[1])
y = relay.var("y", shape=[1])
fn = relay.Function([x, y], x + y)
mod = tvm.IRModule.from_expr(fn)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params={"y": np.ones(1, dtype="float32")})
g = graph_executor.GraphModule(lib["default"](dev))


# In[4]:


g.set_input("x", np.ones(1) * 3)
g.run()
g.get_output(0)


# In[5]:


lib_param = lib.get_params()
lib_param["p0"] = tvm.nd.array(np.ones(1, dtype="float32") * 2, tvm.cpu(0))
g.load_params(tvm.runtime.save_param_dict(lib_param))

g.run()
g.get_output(0)


# In[6]:


lib_param = lib.get_params()
lib_param["p0"] = tvm.nd.array(np.ones(1, dtype="float32") * 3, tvm.cpu(0))
g.load_params(tvm.runtime.save_param_dict(lib_param))

g.run()
g.get_output(0)


# In[ ]:





# In[ ]:





# In[7]:


# TODO: fix the autograd example
# TODO: fix the example


# In[8]:


target = "llvm"

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
input_name = "input0"
shape_list = [(input_name, input_data.shape)]

model = models.resnet18(pretrained=True)
model = model.eval()

data = np.random.randn(*input_shape)
dev = tvm.cpu(0)


# In[9]:


scripted_model = torch.jit.trace(model, input_data).eval()

mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, use_parser_friendly_name=True)
target = "llvm"


# In[10]:


# with tvm.transform.PassContext(opt_level=3):
lib = relay.build(mod, target=target, params=params)


# In[11]:


g = graph_executor.GraphModule(lib["default"](dev))


# In[12]:


data = np.random.randn(1, 3, 224, 224)
g.set_input(input_name, data)


# In[13]:


g.run()
print(f"{np.mean(g.get_output(0).numpy()) * 1e9:.6f}")


# In[14]:


lib_params = lib.get_params()
v = lib_params["p60"]
v.shape


# In[15]:


lib_params = lib.get_params()
v = lib_params["p60"]
g.load_params(tvm.runtime.save_param_dict(lib_params))
g.run()
print(f"{np.mean(g.get_output(0).numpy()) * 1e9:.6f}")
# print(f"{np.mean(g.get_output(0).numpy()) * 1e9:.2f}")


# In[16]:


from copy import deepcopy 
new_lib_params = dict()
for k, v in lib_params.items():
    new_lib_params[k] = np.copy(lib_params[k])


# In[17]:


v = lib_params["p60"]
# g.load_params(tvm.runtime.save_param_dict(lib_params))
g.run()
print(f"{np.mean(g.get_output(0).numpy()) * 1e9:.6f}")


# In[18]:


v = lib_params["p60"].numpy()
lib_params["p60"] = tvm.nd.array(np.zeros_like(v), tvm.cpu(0))


v = lib_params["p61"].numpy()
lib_params["p61"] = tvm.nd.array(np.zeros_like(v) + 1, tvm.cpu(0))


g.load_params(tvm.runtime.save_param_dict(lib_params))
g.run()
g.get_output(0).numpy()


# In[ ]:





# In[ ]:





# In[171]:


with tvm.transform.PassContext(opt_level=3):
    graph, libs, params = relay.build(mod, target=target, params=params)
type(graph), type(libs), type(params)


# In[173]:


print(len(params.keys()))


# In[174]:


with tvm.transform.PassContext(opt_level=3):
    graph, libs, params = relay.build(mod, target=target, params=None)
type(graph), type(libs), type(params)


# In[175]:


print(len(params.keys()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[166]:


"p60"
for k in sorted(lib.get_params().keys(), key=lambda x: int(x[1:])):
    print(k, lib.get_params()[k].shape)


# In[93]:


for k, v in params.items():
    print(k, v.shape)


# In[ ]:




