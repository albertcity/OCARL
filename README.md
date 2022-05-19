# Implementation for Object-Category Aware Reinforcement Learning.

We use a modified version of Tianshou and SPACE, which are included in this repository. 
You should install Tianshou first ().

Addional needed packages are listed as follows:

```
pytorch=1.8.1
stable-baselines3=1.1.0
matplotlib=3.0.2

```

We provides the pretrained SPACE model in the 'space_models' dictionary, as well as the dataset to train SPACE in the 'space_datasets'. The config file for SPACE is saved in the 'space_models' dictionary.
To reproduce the results in the 'Evaluation' section, you can just run the following scripts: 

```
python ts_train.py pol_type=ocarl task=hunter env_kwargs.train.spawn_args='Z1C0/Z0C1'
```
The 'pol_type' above can be set to 'ocarl', 'mlp', 'rrl', or 'smorl'; 'task'='hunter' or 'crafter'; env_kwargs.train.spawn_args='Z1C0/Z0C1', 'Z4C0/Z0C4', 'Z1C1' or 'Z4C4'.

