import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from configs import get_args
import trainer

trainer_names = sorted(name for name in trainer.__dict__ if 'Trainer' in name and callable(trainer.__dict__[name]))

def main():
    args = get_args()
    use_trainer = getattr(trainer, args.trainer)(args)
    use_trainer.run()

if __name__ == '__main__':
    main()
