"""Canonical ResNet-18 (2,2,2,2; 64,128,256,512), ten-class head.

The original 7x7/stride-2 stem and 3x3/stride-2 max-pool are retained.
Native 32x32 CIFAR inputs are used; this is not the 3x3-stem CIFAR variant.
Architecture follows He et al., 2016; implementation is written for this study.
"""
import torch
from torch import nn
from .optim import ns_torch


class BasicBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)))

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self.stage(64, 64, 1)
        self.layer2 = self.stage(64, 128, 2)
        self.layer3 = self.stage(128, 256, 2)
        self.layer4 = self.stage(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def stage(incoming, outgoing, stride):
        return nn.Sequential(BasicBlock(incoming, outgoing, stride), BasicBlock(outgoing, outgoing))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.fc(self.pool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1))


class ConvMuon(torch.optim.Optimizer):
    """Hidden convolutional weights flattened to (out_channels, rest).

    K=0 explicitly means Frobenius-normalized momentum with no polynomial step.
    It is not an identity/zero update and not unnormalized momentum SGD.
    """
    def __init__(self, params, lr=0.02, steps=5, momentum=0.95, weight_decay=0.0005):
        if steps not in {0, 1, 3, 5, 7}:
            raise ValueError('Unsupported ablation depth')
        super().__init__(params, dict(lr=lr, steps=steps, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 4:
                    raise ValueError('ConvMuon expects hidden convolution kernels')
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                state['momentum'].lerp_(p.grad, 1 - group['momentum'])
                matrix = p.grad.lerp(state['momentum'], group['momentum']).reshape(p.shape[0], -1)
                q = (matrix / (matrix.norm() + 1e-7) if group['steps'] == 0 else ns_torch(matrix, group['steps']))
                q = q * max(1.0, q.shape[0] / q.shape[1]) ** 0.5
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(q.reshape_as(p), alpha=-group['lr'])


def vision_optimizers(model, method, lr, auxiliary_lr=0.001, steps=5):
    decay, nodecay, hidden = [], [], []
    names = {'muon': [], 'auxiliary_decay': [], 'auxiliary_nodecay': []}
    for name, p in model.named_parameters():
        if method == 'muon' and p.ndim == 4 and name != 'conv1.weight':
            hidden.append(p)
            names['muon'].append(name)
        elif p.ndim >= 2:
            decay.append(p)
            names['auxiliary_decay'].append(name)
        else:
            nodecay.append(p)
            names['auxiliary_nodecay'].append(name)
    groups = [{'params': decay, 'weight_decay': 0.0005}, {'params': nodecay, 'weight_decay': 0.0}]
    if method == 'sgd':
        return [(torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=False, foreach=False), lr)], names
    aux = lr if method == 'adamw' else auxiliary_lr
    optimizers = [(torch.optim.AdamW(groups, lr=aux, betas=(0.9, 0.95), eps=1e-8, foreach=False), aux)]
    if hidden:
        optimizers.append((ConvMuon(hidden, lr=lr, steps=steps), lr))
    return optimizers, names
