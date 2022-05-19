from .space.space import Space
from .vqspace.space import Space as VQSpace
from .qrspace.space import Space as QRSpace

# from .vqspace.space import Space as VQSpace

__all__ = ['get_model']

def get_model(cfg):
    """
    Also handles loading checkpoints, data parallel and so on
    :param cfg:
    :return:
    """
    
    model = None
    if cfg.model == 'SPACE':
        model = Space(cfg.arch)
    elif cfg.model == 'VQSPACE':
        model = VQSpace(cfg.vqarch)
    elif cfg.model == 'QRSPACE':
        model = QRSpace(cfg.arch)
    print('+++++++++++++++++++++++++++++++')
    print(f'Using Model {model.__class__}.')
    return model
