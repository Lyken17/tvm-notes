import torch
import torch.nn as nn

def disable_dropout_bn(module):
    module_output = module
    if isinstance(module, (nn.Dropout)):
        print("removing dropout")
        module_output = nn.Identity()
    if isinstance(module, (BasicBlock, )):
        print("[BasicBlock]\tfusing BN and dropout")
        idx = 0
        conv = module.conv1
        bn = module.bn1
        channels = bn.weight.shape[0]
        invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight * \
                bn.weight[:, None, None, None] * \
                invstd[:, None, None, None]
        if conv.bias is not None:
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
        module.bn1 = nn.Identity()
        
        conv = module.conv2
        bn = module.bn2
        channels = bn.weight.shape[0]
        invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight * \
                bn.weight[:, None, None, None] * \
                invstd[:, None, None, None]
        if conv.bias is not None:
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
        module.bn2 = nn.Identity()
    if isinstance(module, (nn.Sequential, )):
        print("[nn.Sequential]\tfusing BN and dropout")
        idx = 0
        for idx in range(len(module) - 1):
            if not isinstance(module[idx], nn.Conv2d) or not isinstance(module[idx+1], nn.BatchNorm2d):
                continue 
            conv = module[idx]
            bn = module[idx+1]
            channels = bn.weight.shape[0]
            invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data = conv.weight * \
                    bn.weight[:, None, None, None] * \
                    invstd[:, None, None, None]
            if conv.bias is not None:
                conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
            module[idx+1] = nn.Identity()
    if isinstance(module, (ResNet, )):
        print("[ResNet]\tfusing BN and dropout")
        conv = module.conv1
        bn = module.bn1
        channels = bn.weight.shape[0]
        invstd = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv.weight.data = conv.weight * \
                bn.weight[:, None, None, None] * \
                invstd[:, None, None, None]
        if conv.bias is not None:
            conv.bias.data = (conv.bias - bn.running_mean) * bn.weight * invstd + bn.bias
        module.bn1 = nn.Identity()
        
    for name, child in module.named_children():
        module_output.add_module(name, disable_dropout_bn(child))
    del module
    return module_output
