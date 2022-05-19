
import os
import platform
import time
import subprocess as cmd
import json
import gym

def inverse_permutation(p):
  q = []
  for i in range(len(p)):
    for j, pi in enumerate(p):
      if pi == i:
        q.append(j)
  assert len(q) == len(p)
  return q

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

def config_mujoco():
  cur_node  = platform.node()
  under_license = []
  check_license = True
  if check_license:
    cmd_str = f'ls ~/.mujoco/keys/*'
    _, file_names = cmd.getstatusoutput(cmd_str)
    for f in file_names.split('\n'):
      under_license.append(f.split('/')[-1].split('.')[0])
  lic = None
  for n in under_license:
    if n in cur_node:
      lic = n
      break
  if lic is None:
    print('node not in the lics list,sleep....')
    return False
  os.environ['MUJOCO_PY_MJKEY_PATH']=f'/home/S/yiqi/.mujoco/keys/{lic}.txt'
  os.environ['MJKEY_PATH']=f'/home/S/yiqi/.mujoco/keys/{lic}.txt'
  return True
  

class ConfDict(dict):
  def __getattr__(self, k):
    return super().__getitem__(k)
  def __setattr__(self, k, v):
    super().__setitem__(k, v)

def save_vars(local_vars, save_dir, save_name = 'args.json'):
  inputs = {}
  for k in local_vars:
    try:
      json.dumps(local_vars[k])
      inputs[k] = local_vars[k]
    except:
      inputs[k] = str(local_vars[k])
  print(save_dir, inputs)
  try:
    with open(os.path.join(save_dir, save_name), 'w') as f:
      json.dump(inputs, f)
  except:
    pass


import imageio
import os
import numpy as np
import cv2

class VideoRecorder(object):
    def __init__(self, dir_name, size = 80, fps=8):
        self.dir_name = dir_name
        self.fps = fps
        self.size = size
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def render(self, env):
        if self.enabled:
            frame = env.render('rgb_array')
            self.frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (self.size, self.size), interpolation=cv2.INTER_NEAREST))    
            # self.frames.append(frame)    
    def record(self, env, obs, ob_shape=(7,7,3), ob_rgb=False):
        if self.enabled:
          for ob in obs:
            if ob_rgb:
              ob = ob * 255
              frame = ob.transpose([1, 2, 0]).astype(np.uint8)
            else:
              frame = env.get_obs_render(ob.reshape(ob_shape))
            self.frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (self.size, self.size), interpolation=cv2.INTER_NEAREST))    

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            # imageio.mimsave(path, self.frames, fps=self.fps)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            videoWriter = cv2.VideoWriter(path, fourcc, self.fps, (self.size, self.size))
            for frame in self.frames:
                videoWriter.write(frame)
            videoWriter.release()
            cv2.destroyAllWindows()

class ImgNormalize(gym.core.ObservationWrapper):
    def __init__(self, env, room_id=True):
      super().__init__(env)
      obs_shape = env.observation_space.shape
      self.room_id = room_id
      assert len(obs_shape) == 3
      if not room_id:
        self.observation_space = gym.spaces.Box(shape=(obs_shape[2], obs_shape[0], obs_shape[1]), low=0, high=1)
        self.obs_spec = {'obs': obs_shape}
      else:
        self.observation_space = gym.spaces.Box(shape=(2 + np.product(obs_shape),), low=-10, high=10000)
        self.obs_spec = {'roomid': (2,), 'obs': (obs_shape[2], obs_shape[0], obs_shape[1])}
    def observation(self, obs):
      obs = obs.astype(np.float) / 255. 
      if not self.room_id:
        return obs.transpose((2,0,1))
      else:
        return np.concatenate([np.asarray(self.getRoom(False)),obs.transpose((2,0,1)).flatten()])
    def getRoom(self, uni=True):
      x, y = self.agent_pos   
      i, j = x // (self.room_size -1), y // (self.room_size -1)
      if uni:
        return j * self.num_rows + i
      else:
        return j, i
      # return i, j



