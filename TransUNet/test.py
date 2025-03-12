import argparse
import logging
import os
import random
import sys
import numpy as np
import wandb
import io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import matplotlib.pyplot as plt
from pipeline.video_input import frame_split
from pipeline.list_generator import npz_files_list
from pipeline.center_of_mass import place_centroids
from pipeline.video_reconstruction import reconstruct_video
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

wandb.init(project = 'deeppatella', group = 'testing', name = 'test_data_augmentation_13_02_2025', resume = 'allow')

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=5, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=12,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--video_path', type=str, default='./datasets/videos/example_video.mp4', help='path to the input video')
parser.add_argument('--original_images', type=str, default='../data/Synapse/original_images.npy', help='path to saved original images')
parser.add_argument('--npz_files', type=str, default='../data/Synapse/test_vol_h5', help='Path to the NPZ files folder')
parser.add_argument('--output_txt_file', type=str, default='./lists/lists_Synapse/test_vol.txt', help='Output folder for the txt file')
parser.add_argument('--predictions_dir', type=str, default='./outputs/predicted_images', help='Predicted images folder')
parser.add_argument('--original_images_file', type=str, default='../data/Synapse/original_images.npy', help='path to the original images numpy file')
parser.add_argument('--placed_centroids_folder', type=str, default='./outputs/placed_center_of_mass', help='placed centroids folder')
parser.add_argument('--output_video_file', type=str, default='./outputs/reconstructed_video.mp4', help='folder for the output video with insertions')
args = parser.parse_args()

print('Video pre-processing...')

try:
    frame_split(args.video_path, args.volume_path, args.original_images)
    print('Video pre-processing finished')
except Exception as e:
    print(f'Error encountered during video pre-processing: {e}')
    exit(1)

print('Generating NPZ file list')

try:
    npz_files_list(args.npz_files, args.output_txt_file)
    print ('NPZ files list generated in {output_txt_file}')
except Exception as e:
    print(f'Coudl not generate the NPZ list: {e}')
    exit(1)

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    # Save predictions to folder

    predictions_dir = './outputs/predicted_images'
    os.makedirs(predictions_dir, exist_ok=True)

    # Resume original code

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        with torch.no_grad():
            output = model(image.to(torch.float32).cuda())
            prediction = torch.sigmoid(output).squeeze(0).cpu().detach().numpy()

        if test_save_path:
            np.save(os.path.join(test_save_path, f"{case_name}_prediction.npy"), prediction)

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

if __name__ == "__main__":

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
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 1,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    #net.load_state_dict(torch.load(snapshot))

    
    checkpoint = torch.load(snapshot)

    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

    wandb.finish()

print('Plotting insertion coordinates over original images...')
place_centroids(args.npz_files, args.predictions_dir, args.original_images_file, args.placed_centroids_folder)
print('Plotting finished')

print('Building video with predictions')
reconstruct_video(args.placed_centroids_folder, args.output_video_file)
print('Video reconstruction finished')