# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
import math
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmcv.runner.hooks import HOOKS, Hook

__all__ = ['UpdateMixDepth']


@HOOKS.register_module()
class UpdateMixDepth(Hook):
    """Load Weight Hook."""
    def __init__(self, mix_alpha, max_iter, x_range_width=10, linear_w=3, mode="sigmoid"):
        super().__init__()
        self.mix_alpha = mix_alpha
        self.max_iter = max_iter
        self.x_range_width = x_range_width
        assert mode in ["sigmoid", "linear"]
        self.mode = mode
        self.linear_w = linear_w
    def before_train_iter(self, runner: Runner) -> None:
        curr_step = runner.iter
        x = (curr_step / self.max_iter) * self.x_range_width - self.x_range_width//2
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if self.mode == "sigmoid":
            model.mix_coefficient = 1 / (1 + math.exp(-self.mix_alpha * x))
        else:
            if x <- self.linear_w:
                model.mix_coefficient = 0
            if x >=-self.linear_w and x<= self.linear_w:
                model.mix_coefficient = 1 / (2 * self.linear_w) * x + 1/2
            if x > self.linear_w:
                model.mix_coefficient = 1