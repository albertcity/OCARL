import os
import argparse
import numpy as np
from argparse import ArgumentParser
# from space.config import cfg
import omegaconf as oc

def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='space/space_small.yaml',
        metavar='FILE',
        help='Path to config file'
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    cfg = oc.OmegaConf.load(args.config_file)
    cfg_cli = oc.OmegaConf.from_dotlist(args.opts)
    cfg = oc.OmegaConf.merge(cfg, cfg_cli)
    # Seed
    if cfg.seed <= 0:
      cfg.seed = np.random.randint(0, 100000000)
    import torch

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    return cfg, args.task


