import os
import random
import numpy as np
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from includes.TransUNet.datasets.dataset_synapse import Synapse_dataset
from includes.TransUNet.utils import test_single_volume
from includes.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from includes.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.root_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"{len(testloader)} test iterations")

    model.eval()

    predictions_dir = './predicted_images'
    os.makedirs(predictions_dir, exist_ok=True)

    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        with torch.no_grad():
            output = model(image.to(torch.float32).cuda())
            prediction = torch.sigmoid(output).squeeze(0).cpu().detach().numpy()

        if test_save_path:
            np.save(os.path.join(test_save_path, f"{case_name}_prediction.npy"), prediction)

        print(f"Testing {i_batch + 1}/{len(testloader)}")

        prediction_image_path = os.path.join(predictions_dir, f'{case_name}_prediction.png')
        plt.imsave(prediction_image_path, prediction.squeeze(), cmap='hot')

        fig, ax = plt.subplots()
        ax.imshow(prediction.squeeze(), cmap='hot')
        ax.set_title('Prediction')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        prediction_image = Image.open(buf)

        wandb.log({
            f'Image {case_name}': wandb.Image(image[0,0].cpu().numpy(), caption='Original Image'),
            f'Prediction {case_name}': wandb.Image(prediction_image, caption='Prediction')            
        })

    return "Testing Finished!"

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
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
    }

    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    # Modelo y configuración
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    

    # Checkpoint loading
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = f"./model/{args.exp}/TU"

    if getattr(args, "is_pretrain", False):
        snapshot_path += '_pretrain'

    snapshot_path += f"_{args.vit_name}"
    snapshot_path += f"_skip{args.n_skip}"
    if args.vit_patches_size != 16:
        snapshot_path += f"_vitpatch{args.vit_patches_size}"
    if args.max_iterations != 30000:
        snapshot_path += f"_{str(args.max_iterations)[:2]}k"
    if args.max_epochs != 30:
        snapshot_path += f"_epo{args.max_epochs}"
    snapshot_path += f"_bs{args.batch_size}"
    if args.base_lr != 0.01:
        snapshot_path += f"_lr{args.base_lr}"
    snapshot_path += f"_{args.img_size}"
    if args.seed != 1234:
        snapshot_path += f"_s{args.seed}"

    os.makedirs(snapshot_path, exist_ok=True)

    snapshot_file = getattr(args, "snapshot_path", None)
    if not snapshot_file:
        snapshot_file = os.path.join(
            "model", f"TU_{args['dataset']}{args['img_size']}",
            f"TU_{args['vit_name']}_skip{args['n_skip']}_epo{args['max_epochs']}_bs{args['batch_size']}_lr{args['base_lr']}_{args['img_size']}",
            f"epoch_{args['max_epochs'] - 1}.pth"
        )
    
    if not os.path.exists(snapshot_file):
        raise FileNotFoundError(f"❌ No se encontró el checkpoint en: {snapshot_file}")




    checkpoint = torch.load(snapshot_file)
    missing_keys, unexpected_keys = net.load_state_dict(checkpoint, strict=False)

    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # Logs
    log_folder = './test_log/test_log_' + f"TU_{dataset_name}{args.img_size}"
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, f"{dataset_name}_test.log"),
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir, f"TU_{dataset_name}{args.img_size}")
        os.makedirs(test_save_path, exist_ok=True)

    return inference(args, net, test_save_path)


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_testing_pipeline(config.get("testing", {}))



