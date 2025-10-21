# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Converting legacy network pickle into the new format."""

import pickle
import torch

#----------------------------------------------------------------------------

def load_network_pkl(f):
    data = _LegacyUnpickler(f).load()

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))
    assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    return data

#----------------------------------------------------------------------------

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'training.networks_baseline':
            module = 'src.R3GAN.training.networks'
        if module[:12] == 'BaselineGAN.':
            module = 'src.R3GAN.' + module[12:]
        return super().find_class(module, name)
