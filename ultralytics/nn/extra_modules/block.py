import torch
import torch.nn as nn

from ..modules.block import C2f
from .EVA import EVA

__all__ = ['C2f_EVA']

######################################## ICIP2025 BEVANET start ########################################

class C2f_EVA(C2f):
    def __init__(self, c1, c2, n=1, stage=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EVA(self.c) for _ in range(n))

######################################## ICIP2025 BEVANET end ########################################