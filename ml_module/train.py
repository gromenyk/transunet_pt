import os
import random
import numpy as np
import torch
import yaml
from types import SimpleNamespace
import torch.backends.cudnn as cudnn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from includes.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from includes.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from includes.TransUNet.trainer import trainer_synapse

def run_training_pipeline(train_cfg):
    args = SimpleNamespace(**train_cfg)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'num_classes': 1,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.is_pretrain = False

    snapshot_path = "includes/model/latest_model"
    os.makedirs(snapshot_path, exist_ok=True)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(args.pretrained_path))

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_training_pipeline(config["training"])
