import math

import torch
import torch.nn as nn

from torch._torchlite import compile, trace, run_passes
from torch._torchlite.passes import _graph_meta, precompile, decompose, fuse, triton_codegen


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 32
        n_heads = 2
        head_dim = 16

        self.w1 = nn.Parameter(torch.randn(10, hidden))
        self.b1 = nn.Parameter(torch.randn(hidden))
        self.rms_norm = nn.RMSNorm(hidden)
        self.wq = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wk = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wv = nn.Parameter(torch.randn(hidden, n_heads * head_dim))
        self.wo = nn.Parameter(torch.randn(n_heads * head_dim, hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, 10))
        self.b2 = nn.Parameter(torch.randn(10))

    def forward(self, x):
        B, S = x.shape[0], x.shape[1]
        n_heads, head_dim = 2, 16

        h = x @ self.w1
        h.add_(self.b1)
        h = self.rms_norm(h)
        h = torch.sin(h)

        q = (h @ self.wq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        k = (h @ self.wk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        v = (h @ self.wv).reshape(B, S, n_heads, head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, n_heads * head_dim)

        h = out @ self.wo
        out = h @ self.w2
        out.add_(self.b2)
        out = torch.cos(out)
        out = torch.dropout(out, 0.5, True)
        return out


class TrainStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, target):
        out = self.model(x)
        return ((out - target) ** 2).mean()


model = MyModel()
train_step = TrainStep(model)
x = torch.randn(2, 4, 10)
target = torch.randn(2, 4, 10)

# trace -> run_passes -> decompose -> fuse -> triton codegen -> precompile
gm = trace(train_step, [x, target])
gm = run_passes(gm, [x, target], lr=0.01)
gm = decompose(gm, [x, target]).gm
gm = fuse(gm, [x, target]).gm
gm = triton_codegen(gm, [x, target]).gm
gm = precompile(gm, [x, target]).gm

code = _graph_meta(gm.graph)["precompiled_code"]
print(code)
