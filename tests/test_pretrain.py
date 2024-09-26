#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

import pytest
import timm
import torch

from terratorch.cli_tools import build_lightning_cli


@pytest.mark.parametrize("model_name", ["prithvi_vit_100"])
def test_finetune_multiple_backbones(model_name):
    command_list = ["fit", "-c", f"tests/resources/configs/manufactured-pretrain_{model_name}_band_interval.yaml"]
    _ = build_lightning_cli(command_list)


