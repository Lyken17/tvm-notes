import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay, te, TVMError, auto_scheduler
from tvm import topi
from tvm.contrib import graph_executor


net = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
)

data = torch.randn(1, 10)
ts = torch.jit.script(net, data)
shape_list = [("input0", data.shape)]
scripted_model = torch.jit.trace(net, data).eval()
fmod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

global extern_prim_fn
extern_prim_fn = None
@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    global extern_prim_fn
    print(f)
    extern_prim_fn = f

try:
    with tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": [(3, print_tir)]}
    ):
        lib = relay.build(fmod, target="llvm")
except TVMError:
    if extern_prim_fn is None:
        raise
        
        
rt_mod = tvm.build(extern_prim_fn, target="c")
print(rt_mod.get_source())
