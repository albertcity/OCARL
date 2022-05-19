from space.engine.utils import get_config
from space.engine.train import train
from space.engine.eval import eval
from space.engine.show import show


if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval': eval,
        'show': show,
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)


