import numpy as np

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import tvm
from tvm import relay, te, TVMError, auto_scheduler
from tvm import topi
from tvm.contrib import graph_executor

x = relay.var("x", shape=[1, 10])
w = relay.var("w", shape=[20, 10])
y = relay.nn.dense(x, w)
fn = relay.Function([x, w], y)
mod = tvm.IRModule.from_expr(fn)

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

target = tvm.target.Target("cuda")

tasks, task_weights = auto_scheduler.extract_tasks(fmod["main"], params, target)
for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
    
log_file = "tune-log.txt"
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=20,  # change this to 20000 to achieve the best performance
    runner=auto_scheduler.LocalRunner(repeat=1, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
tuner.tune(tune_option)


log_file = "tune-log.txt"
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=20,  # change this to 20000 to achieve the best performance
    runner=auto_scheduler.LocalRunner(repeat=1, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
tuner.tune(tune_option)

# https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.html#sphx-glr-how-to-tune-with-autoscheduler-tune-conv2d-layer-cuda-py

print("CUDA source code:")
print(task.print_best(log_file, print_mode="cuda"))
