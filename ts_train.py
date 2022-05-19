import argparse
import multiprocessing as mp
import os
import functools
from omegaconf import OmegaConf, DictConfig
import omegaconf as oc
import warnings
warnings.filterwarnings('ignore')
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import VectorReplayBuffer
from collector import Collector
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy
from onpolicy import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
from stable_baselines3.common import logger as L
from policies import *
from tianshou.trainer import gather_info, test_episode

from objenv_wrapper import *
from hunter_game import Env as Hunter
import crafter

def make_clean_env(env_name, cfg):
  if 'crafter' in env_name:
      env = crafter.Recorder(crafter.Env(), cfg.logdir,
               save_stats=True, save_video=False, save_episode=False)
  elif 'hunter' in env_name:
      env = Hunter(**cfg.env_kwargs)
  else:
      assert False
  return env

def make_policy(cfg, obs_space, act_space):
    pol_cls = PPO 
    if cfg.pol_type in ['ocarl', 'rrl']:
        pol_cls = ObjSpacePolicy
    elif cfg.pol_type == 'smorl':
        pol_cls = SMORL
    return pol_cls(obs_space, act_space,
              device = 'cuda' if torch.cuda.is_available() else 'cpu',
              pol_kwargs=cfg.pol_kwargs[cfg.pol_type],
              ppo_kwargs=cfg.ppo_kwargs)

def get_args(config_file = './config.yaml', test=False, cfg_cli=None):
    from omegaconf import OmegaConf
    args = OmegaConf.load(config_file) 
    if test:
        args = OmegaConf.merge(args, args.test)
    if cfg_cli:
      args = OmegaConf.merge(args, cfg_cli)
    args.use_test = test
    return args

class MultiCollector:
  def __init__(self, collectors):
    self.collectors = collectors
  def reset_stat(self):
    for c in self.collectors:
      c.reset_stat()
  @property
  def collect_time(self): 
    return sum([c.collect_time for c in self.collectors])
  @property
  def collect_step(self): 
    return sum([c.collect_step for c in self.collectors])
  @property
  def collect_episode(self): 
    return sum([c.collect_episode for c in self.collectors])
  def __iter__(self):
    return iter(self.collectors)
  def __getitem__(self, i):
    return self.collectors[i]
  def __len__(self):
    return len(self.collectors)

def custom_test_episode(policy, test_collector, test_fn, epoch, n_episode,
                        logger=None, global_step=None, reward_metric=None, test_prefix=[]):
    if isinstance(test_collector, MultiCollector):
      assert len(test_collector) == len(test_prefix)
      all_res = {}
      for tc, prefix in zip(test_collector, test_prefix):
        res = test_episode(policy, tc, test_fn, epoch, n_episode, logger, global_step, reward_metric) 
        all_res.update({f'SepTest/{prefix}_{k}': res[k] for k in ['rew', 'rew_std', 'n/ep', 'n/st', 'len']})
      all_res.update(res)
      return all_res
    else:
      return test_episode(policy, test_episode, test_fn, epoch, n_episode, logger, global_step, reward_metric) 

def test_ppo(args, policy=None, logger=None, configureL=True, start_infos=None):
    env_name = args.task
    if args.seed == -1:
      args.seed = np.random.randint(1, 10000000)
    log_path = os.path.join(args.logdir, args.task, f's{args.seed}')

    os.makedirs(log_path, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    obj_cat_num = args.obj_cat_nums.get(args.task, 0)
    # 1. Prepare Envs.
    get_env_args = lambda env_kwargs: oc.DictConfig(dict(G=8, obj_cat_num=obj_cat_num,
                                        logdir=log_path, reset_new=False,
                                        env_kwargs=env_kwargs))
    train_env_args = get_env_args(env_kwargs=args.env_kwargs.get('train', {}))
    test_env_args = args.env_kwargs.get('test', {})
    multi_test = args.get('multi_test', False)
    if not multi_test:
      test_env_args = [test_env_args]
    test_env_args = [get_env_args(argsi) for argsi in test_env_args]
    if args.pol_type == 'mlp':
      # cur_make_env = lambda env_name, args: gym.make(env_name, **args.env_kwargs)
      cur_make_env = make_clean_env
    else:
      cur_make_env = lambda env_name, args: ObjEnvWrapper(env_name, args)
    if 'crafter' not in env_name:
      env_cls = DummyVectorEnv
    else:
      env_cls = SubprocVectorEnv
    train_envs = env_cls(
        [lambda: cur_make_env(env_name, train_env_args) for i in range(args.training_num)]
    )
    test_envs_fn_i = lambda env_args: DummyVectorEnv(
        [lambda: cur_make_env(env_name, env_args) for i in range(args.test_num)]
    )
    test_envs_fn = lambda: [test_envs_fn_i(env_args) for env_args in test_env_args]
    test_prefix = [f'Test{i}' for i in range(len(test_env_args))]
    test_envs = test_envs_fn()
    env = cur_make_env(env_name, train_env_args)


    # 2. Prepare policy and Collector.
    if args.pol_type not in ['smorl', 'ocarl']:
      args.space.use_space = False
    if args.pol_type == 'smorl':
      objcat_preprocess = ObjCatPreprocessV2(env, args.space)
      args.pol_kwargs.smorl.input_shape = objcat_preprocess.output_shape 
    else:
      objcat_preprocess = ObjCatPreprocess(env, args.space)
    if policy is None:
      policy = make_policy(args, env.observation_space, env.action_space)
      print(policy)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f'Total Params: {total_params}')
    print(f'Obs Space: {env.observation_space}, Act space: {env.action_space}')
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs),
          ignore_obs_next=args.get('ignore_obs_next', True)),
        preprocess_fn = objcat_preprocess,
    )
    test_collector = MultiCollector([Collector(policy, test_env, env_reset_fn=None, preprocess_fn=objcat_preprocess) for test_env in test_envs])
    # 3. Configure logs.
    os.makedirs(log_path, exist_ok=True)
    print(OmegaConf.to_yaml(args))
    OmegaConf.save(args, os.path.join(log_path, 'config.yaml'))
    if logger is None:
      writer = SummaryWriter(log_path)
      logger = TensorboardLogger(writer)
    if configureL:
      L.configure(log_path, ['csv'])
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # 4. Traning.
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num * 2,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        save_fn=save_fn,
        save_checkpoint_fn = (lambda ep, env_step, gs: None),
        logger=logger,
        L=L,
        test_in_train=False,
        start_infos= start_infos,
        custom_test_episode = functools.partial(custom_test_episode, test_prefix=test_prefix)
    )
    return policy, result, logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_known_args()[0]
    cfg_cli = OmegaConf.from_dotlist(args.opts)
    args = get_args(args.config, args.test, cfg_cli)
    test_ppo(args)
