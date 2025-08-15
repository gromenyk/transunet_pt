import os
import random
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import logging
import sys
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import io
from types import SimpleNamespace
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from includes.TransUNet.datasets.dataset_synapse import Synapse_dataset
from includes.TransUNet.utils import test_single_volume
from includes.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from includes.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def inference(args, model, test_save_path=None, save_debug=False, debug_dir="debug"):
    db_test = args.Dataset(base_dir=args.root_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations")

    model.eval()

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    predictions_in_memory = {}

    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        with torch.no_grad():
            output = model(image.to(torch.float32).cuda())
            prediction = torch.sigmoid(output).squeeze(0).cpu().detach().numpy()

        DEBUG_INDICES = [50, 100, 200]

        if save_debug and i_batch in DEBUG_INDICES and test_save_path:
            os.makedirs(test_save_path, exist_ok=True)
            np.save(os.path.join(test_save_path, f"{case_name}_prediction.npy"), prediction)
            prediction_image_path = os.path.join(test_save_path, f'{case_name}_prediction.png')
            plt.imsave(prediction_image_path, prediction.squeeze(), cmap='hot')

 
        image_np = image.squeeze().cpu().numpy()
        label_np = label.squeeze().cpu().numpy()

        original_image_np = sampled_batch["original_image"].numpy() \
            if torch.is_tensor(sampled_batch["original_image"]) else sampled_batch["original_image"]


        if image_np.ndim == 3:
            if image_np.shape[0] in [1, 3]:  
                image_np = image_np[0]
            elif image_np.shape[2] in [1, 3]:  
                image_np = image_np[..., 0]

        if label_np.ndim == 3:
            if label_np.shape[0] in [1, 3]:
                label_np = label_np[0]
            elif label_np.shape[2] in [1, 3]:
                label_np = label_np[..., 0]

        predictions_in_memory[case_name] = {
            "image": image_np,
            "label": label_np,
            "prediction": prediction.squeeze(),
            "original_image": original_image_np 
        }

        print(f"Testing {i_batch + 1}/{len(testloader)}")

        fig, ax = plt.subplots()
        ax.imshow(prediction.squeeze(), cmap='hot')
        ax.set_title(f'Prediction {case_name}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        prediction_image = Image.open(buf)

    return predictions_in_memory

def run_testing_pipeline(test_cfg):
    args = SimpleNamespace(**test_cfg)

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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'num_classes': args.num_classes,
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot_dir = "includes/model/latest_model"
    model_files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pth")), key=os.path.getmtime)
    if not model_files:
        raise FileNotFoundError(f"No model found in {snapshot_dir}")
    snapshot_file = model_files[-1]
    print(f"Using model: {snapshot_file}")

    checkpoint = torch.load(snapshot_file)
    missing_keys, unexpected_keys = net.load_state_dict(checkpoint, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    log_folder = f'./test_log/test_log_TU_{dataset_name}{args.img_size}'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, f"{dataset_name}_test.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    save_debug = test_cfg.get("save_debug", False)
    debug_dir = test_cfg.get("debug_dir", "debug/testing")
    test_save_path = None
    if save_debug:
        test_save_path = os.path.join(debug_dir, f"TU_{dataset_name}{args.img_size}")
        os.makedirs(test_save_path, exist_ok=True)

    return inference(args, net, test_save_path=test_save_path, save_debug=save_debug, debug_dir=debug_dir)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_testing_pipeline(config.get("testing", {}))
