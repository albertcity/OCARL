import numpy as np
import gym
import torch
import os
import matplotlib.image as mpimg
from PIL import Image
import functools
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random

class Env:
  def __init__(self, area=(8,8), img_size=(64,64),
               spawn_args = 'Z4C4',
               max_len=64, **kwargs):
    self.area = area
    self.max_len = max_len
    self.cur_len = 0
    self.img_size = img_size

    random_sample_args = [(int(v[1]), int(v[3])) for v in spawn_args.split('/')]
    print(f'ENV Random Sample Args: {random_sample_args}')

    self.random_sample_args = random_sample_args
    self.state = np.zeros(area)
    actions = ['NONE', 'U', 'D', 'L', 'R', 'FU', 'FD', 'FL', 'FR']
    self.act_map = {k:v for k,v in zip(np.arange(len(actions)), actions)}
    self.object_map = {'zombie': [], 'cow': [], 'agent':[]}
    self.idx2obj_map = {k:v for k, v in zip(np.arange(5), ['none', 'zombie', 'agent', 'cow', 'wall'])}
    self.obj2idx_map = {v:k for k, v in zip(np.arange(5), ['none', 'zombie', 'agent', 'cow', 'wall'])}
    self.block_size = img_size[0] // self.area[0], img_size[1] // self.area[1]
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*img_size, 3))
    self.action_space = gym.spaces.Discrete(len(actions))
    self.textures = self.load_textures()

  def load_textures(self):
    assets_list = ['sand', 'zombie', 'player', 'cow', 'stone']
    assets = {k: os.path.join('assets', f'{v}.png')
                for k,v in zip(['none', 'zombie', 'agent', 'cow', 'wall'],
                               assets_list)}
    textures = {k:np.asarray(Image.open(v).resize(self.block_size, Image.ANTIALIAS)) for k,v in assets.items()}
    return textures

  def draw_item(self, canvas, pos, obj_type):
    sx, sy = pos
    bx, by = self.block_size
    obj_texture = self.textures[obj_type]
    if obj_texture.shape[-1] == 3:
      canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] = obj_texture
    elif obj_texture.shape[-1] == 4:
      cur_texture = canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] 
      alpha = obj_texture[...,-1:] / 255.
      obj_texture = obj_texture[...,:-1]
      canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] = cur_texture * (1-alpha) + alpha * obj_texture 
    else:
      assert False
    return canvas
    
  @functools.lru_cache(10)
  def background(self):
    canvas = np.zeros(self.observation_space.shape)
    for i in range(self.area[0]):
      for j in range(self.area[1]):
        canvas = self.draw_item(canvas, (i,j), 'none')
    return canvas
    
  def render(self, mode='rgb'):
    if mode == 'symbolic':
      array = []
      idx2symb_map = {k:v for k, v in zip(np.arange(5), [' ', 'Z', 'A', 'C', 'W'])}
      for a in self.state:
        array.append([])
        for j in a:
          array[-1].append(idx2symb_map[j])
      return np.asarray(array)
    elif mode == 'rgb':
      canvas = self.background().copy()
      for i in range(self.area[0]):
        for j in range(self.area[1]):
          obj_type = self.idx2obj_map[self.state[(i,j)]]
          if obj_type in ['zombie', 'agent', 'cow', 'wall']:
            canvas = self.draw_item(canvas, (i,j), obj_type)
      return (canvas).astype(np.uint8)
    else:
      assert False

  def get_gt_bbox(self):
    bbox = []
    bx, by = self.block_size
    for i in range(self.area[0]):
      for j in range(self.area[1]):
        obj = self.state[i, j]
        if obj > 0:
          bbox.append([j*by,i*bx,bx,by,obj]) 
    bbox = np.asarray(bbox)
    num_obj = len(bbox)
    if len(bbox) < 32:
      bbox = np.concatenate([bbox, np.ones((32 - bbox.shape[0], bbox.shape[1])) * -1], axis=0)
    return bbox, num_obj
      
  def random_spawn(self, obj_type):
    pos = np.random.randint(0, self.area[0]), np.random.randint(0, self.area[1])
    if self.state[pos] != 0:
      self.random_spawn(obj_type)
    else:
      self.state[pos] = self.obj2idx_map[obj_type]
      self.object_map[obj_type].append(pos)
  def random_wall(self):
    pos1 = np.random.randint(1, self.area[0]//2-1), np.random.randint(1, self.area[1]//2-1)          # (1,2,3)
    pos2 = np.random.randint(self.area[0]//2, self.area[0] - 1), np.random.randint(self.area[1] // 2, self.area[1] - 1) # (4,5,6)
    dir1 = np.random.randint(0,4)
    dir2 = np.random.randint(0,4)
    wall_idx = self.obj2idx_map['wall']
    for d, pos in zip([dir1, dir2],[pos1, pos2]):
      if d == 0:
        self.state[pos[0]:, pos[1]] = wall_idx
      elif d == 1:
        self.state[:pos[0]+1, pos[1]] = wall_idx
      elif d == 2:
        self.state[pos[0], pos[1]:] = wall_idx
      elif d == 3:
        self.state[pos[0], :pos[1]+1] = wall_idx
      else:
        assert False
    if dir1 == 0 and dir2 == 3:
      self.state[pos2[0],:pos1[1]] = 0
    elif dir1 == 2 and dir2 == 1:
      self.state[:pos1[0],pos2[1]] = 0
      
  def reset(self):
    self.cur_len = 0
    self.state = np.zeros(self.area)
    self.object_map = {'zombie': [], 'cow': [], 'agent':[]}
    self.random_wall()
    self.random_spawn('agent')
    num_zombie, num_cow = random.choice(self.random_sample_args)
    for i in range(num_zombie):
      self.random_spawn('zombie')
    for i in range(num_cow):
      self.random_spawn('cow')
    return self.render()
  def handle_collosion(self, pos, try_pos):
    valid = True
    die   = False
    if try_pos[0] < 0 or try_pos[0] >= self.area[0] or try_pos[1] < 0 or try_pos[1] >= self.area[1]:
      valid = False
    targ_type = self.idx2obj_map(self.state[try_pos])
  def check_bound(self, try_pos): 
    return not(try_pos[0] < 0 or try_pos[0] >= self.area[0] or try_pos[1] < 0 or try_pos[1] >= self.area[1])
  def remove(self, pos):
    obj = self.idx2obj_map[self.state[pos]]
    assert obj in ['agent', 'cow', 'zombie']
    assert pos in self.object_map[obj]
    self.state[pos] = 0
    self.object_map[obj] = list(set(self.object_map[obj]) - set([pos]))
    
  def move(self, pos, pos2):
    o1= self.state[pos]
    obj = self.idx2obj_map[o1]
    self.state[pos2] = o1
    if obj in self.object_map:
      self.remove(pos)
      self.object_map[obj].append(pos2) #= list(set(self.object_map[obj]) - set([pos]))

  def step(self, a):
    ag_pos = self.object_map['agent']
    assert len(ag_pos) == 1
    ag_pos = ag_pos[0]
    a_type = self.act_map[a]
    reward = 0
    done = False
    if a_type in ['U', 'D', 'L', 'R']:
      try_pos = ag_pos[0] - int(a_type == 'U') + int(a_type == 'D'), ag_pos[1] - int(a_type=='L') + int(a_type=='R') 
      if self.check_bound(try_pos):
        obj_type = self.idx2obj_map[self.state[try_pos]]
        if obj_type  == 'none':
          self.move(ag_pos, try_pos)
        elif obj_type == 'zombie':
          reward -= 1
          done = True
        elif obj_type == 'cow':
          self.remove(try_pos)
          self.move(ag_pos, try_pos)
          reward += 1
    elif 'F' in a_type:
      ax, ay = ag_pos
      for ex, ey in self.object_map['zombie']:
        assert (ex, ey) != (ax, ay)
        destroyed = False
        if ex == ax:
          destroyed = (ey > ay and a_type == 'FR') or (ey < ay  and a_type == 'FL')
        if ey == ay:
          destroyed = (ex > ax and a_type == 'FD') or (ex < ax  and a_type == 'FU')
        if destroyed:
          self.remove((ex, ey))
          reward += 1
    # randomly move enemies
    for ex, ey in self.object_map['zombie']: 
      a_type = np.random.choice(['U', 'D', 'L', 'R'])
      try_pos = ex - int(a_type == 'U') + int(a_type == 'D'), ey - int(a_type=='L') + int(a_type=='R') 
      if self.check_bound(try_pos):
        obj_type = self.idx2obj_map[self.state[try_pos]]
        if obj_type == 'agent':
          reward -= 1
          done = True
        elif obj_type == 'none':
          self.move((ex, ey), try_pos)
    clear = ((len(self.object_map['cow']) + len(self.object_map['zombie'])) == 0)
    if clear:
      reward += 5
    self.cur_len += 1
    if self.cur_len >= self.max_len or clear:
      done = True
    ag_pos = self.object_map['agent']
    assert len(ag_pos) == 1
    ag_pos = ag_pos[0]
    return self.render(), reward, done, dict(ag_pos=np.asarray(ag_pos),
              num_zombie=len(self.object_map['zombie']),
              state=self.state.copy(),
              metric_clear=int(clear),
              num_cow=len(self.object_map['cow']))

def run_gui():
  env = Env(num_zombie=2, num_cow=4)
  o = env.reset()
  keymap = {k:v for k, v in zip([' ',
                                'w','s','a','d',
                                'i','k', 'j', 'l',
                                ], np.arange(9))}
  running = True
  print(o)
  while running:
    event = input('Enter a action:')
    if event in keymap.keys():
      a = keymap[event]
      o, r, d, i = env.step(a)
      print(o)
      print(r, d)
      if d:
        print('RESET')
        o = env.reset()
        print(o)


if __name__ == '__main__':
  # play yourself
  run_gui()
