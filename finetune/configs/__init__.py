import os
import argparse
from omegaconf import OmegaConf
import pathlib
import trainer
algorithm_names = sorted(name for name in trainer.__dict__ if 'Trainer' in name and callable(trainer.__dict__[name]))


def get_args():
    parser = argparse.ArgumentParser(description="DinoV2 finetune")
    
    parser.add_argument("--config-file", type=str, required=True, help="Model configuration file", default="/data/dataserver01/zhangruipeng/code/BoneFM/our_finetune/configs/default_configs.yaml")
    parser.add_argument("--output-dir", default="", type=str, help="Output directory to write results and logs")
    parser.add_argument("--log-interval", type=int, help="Log interval", default=10)
    parser.add_argument("--log_display", action="store_true", help="Whether to display log")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from checkpoint. Default: False (start from scratch). Use --resume to enable resuming.")

    args = parser.parse_args()

    default_config_path = pathlib.Path(__file__).parent.resolve() / "default_configs.yaml"
    default_cfg = OmegaConf.load(default_config_path)
    
    user_cfg = OmegaConf.load(args.config_file)

    merged_cfg = OmegaConf.merge(default_cfg, user_cfg)

    for key, value in merged_cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)
        else:
            if getattr(args, key) is None:
                setattr(args, key, value)

    args.output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args_dict = vars(args)
    sorted_args_dict = dict(sorted(args_dict.items()))

    config_yaml_path = os.path.join(args.output_dir, "config.yaml")
    with open(config_yaml_path, "w") as f:
        OmegaConf.save(config=OmegaConf.create(sorted_args_dict), f=f)

    return args


if __name__ == "__main__":
    args = get_args()
    print(args.batch_size)