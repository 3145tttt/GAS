import torch
from ml_collections import ConfigDict
from src.gas.base_model import BaseModel, EDMModel, LDMModel, SDModel
from src.gas.gs_wrapper import GSWrapper, GSWrapperLatent


def load_base_model(config: ConfigDict, device=torch.device('cuda')) -> BaseModel:
    if config.type == 'EDM':
        return EDMModel(config, device)

    if config.type == 'LDM':
        return LDMModel(config, device)

    if config.type == 'SD':
        return SDModel(config, device)

    raise NotImplementedError(f"unknown model type {config.type} was passed")


def get_gs_wrapper(model: BaseModel, solver_config: ConfigDict) -> GSWrapper:
    if model.config.type == 'EDM':
        gs_wrapper = GSWrapper(model, solver_config)

    elif model.config.type in ['LDM', 'SD']:
        gs_wrapper = GSWrapperLatent(model, solver_config)

    else:
        raise NotImplementedError(f"unknown model type {model.config.type} was passed")

    gs_wrapper.to(model.device)
    gs_wrapper.eval()    
    return gs_wrapper
