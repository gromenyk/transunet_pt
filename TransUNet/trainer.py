import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from scipy.ndimage import center_of_mass

### Initialize Wandb

wandb.init(project="deeppatella", entity="deeppatella", group = 'training', name='train_data_augmentation_13_02_2025')

def euclidean_distance(real, pred):
    return torch.sqrt(torch.sum((real - pred) ** 2, dim=1)).mean()


def find_centers_of_mass_for_hottest_pixels(prediction):
    """Encuentra los centroides en el mapa de calor de la predicción."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.squeeze().cpu().detach().numpy()
    else:
        prediction = prediction.squeeze()  

    h, w = prediction.shape
    mid = w // 2

    left_half = prediction[:, :mid]
    right_half = prediction[:, mid:]

    left_max_value = np.max(left_half)
    right_max_value = np.max(right_half)

    left_mask = (left_half == left_max_value).astype(np.float32)
    right_mask = (right_half == right_max_value).astype(np.float32)

    left_com = center_of_mass(left_mask)
    right_com = center_of_mass(right_mask)

    left_center = (left_com[0], left_com[1])
    right_center = (right_com[0], right_com[1] + mid)

    return left_center, right_center


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = 1
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=None) #transforms.Compose(
                                   #[RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val", transform=None)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    scale_x = 512 / 224
    scale_y = 512 / 224

    ce_loss = nn.BCELoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        total_train_loss = 0
        total_euclidean_error = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, coords_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['coords']
            image_batch, label_batch, coords_batch = image_batch.cuda().float(), label_batch.cuda(), coords_batch.cuda()
            outputs = model(image_batch)
            loss_bce = ce_loss(torch.sigmoid(outputs), label_batch.float().unsqueeze(1))
            loss_dice = dice_loss(torch.sigmoid(outputs), label_batch)
            loss = 0.8 * loss_bce + 0.2 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()	

            predicted_coords = []
            for output in outputs:
                pred_distal, pred_proximal = find_centers_of_mass_for_hottest_pixels(output.squeeze().cpu().detach().numpy())
                pred_coords = torch.tensor([[pred_distal[1] * scale_x, pred_distal[0] * scale_y], 
                                            [pred_proximal[1] * scale_x, pred_proximal[0] * scale_y]])
                predicted_coords.append(pred_coords)

            predicted_coords = torch.stack(predicted_coords).cuda()
            coords_batch = coords_batch.cuda().float()
            train_euclidean_error = euclidean_distance(coords_batch, predicted_coords)

            total_euclidean_error += train_euclidean_error.item()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            total_train_loss += loss.item()

            print(f"Epoch {epoch_num} | Iteration {iter_num} | Train Loss: {loss.item():.4f} | BCE Loss: {loss_bce.item():.4f} | Euclidean Error: {train_euclidean_error.item():.4f}")

            wandb.log({
                'Train Total Loss': loss.item(),
                'Train BCE Loss': loss_bce.item(),
                'Train DICE Loss': loss_dice.item(),
                'Train Euclidean Distance Error': train_euclidean_error.item()
            })

        avg_train_loss = total_train_loss / len(trainloader)
        avg_train_euclidean_error = total_euclidean_error / len(trainloader)    
    

        model.eval()
        total_val_loss = 0
        total_val_euclidean_error = 0
        with torch.no_grad():
            for val_batch in val_loader:  
                val_images, val_labels, val_coords = val_batch['image'].cuda().float(), val_batch['label'].cuda().float(), val_batch['coords'].cuda().float()
                val_outputs = model(val_images)

                val_loss_bce = ce_loss(torch.sigmoid(val_outputs), val_labels.float().unsqueeze(1))
                val_loss_dice = dice_loss(torch.sigmoid(val_outputs), val_labels)
                val_loss = 0.8 * val_loss_bce + 0.2 * val_loss_dice

                total_val_loss += val_loss.item()

                predicted_coords = []
                for output in val_outputs:
                    pred_distal, pred_proximal = find_centers_of_mass_for_hottest_pixels(output.squeeze().cpu().detach().numpy())
                    pred_coords = torch.tensor([[pred_distal[1] * scale_x, pred_distal[0] * scale_y], 
                                                [pred_proximal[1] * scale_x, pred_proximal[0] * scale_y]])
                    predicted_coords.append(pred_coords)

                predicted_coords = torch.stack(predicted_coords).cuda()
                val_euclidean_error = euclidean_distance(val_coords, predicted_coords)

                total_val_euclidean_error += val_euclidean_error.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_euclidean_error = total_val_euclidean_error / len(val_loader)

        wandb.log({
            'Train Loss (epoch)': avg_train_loss,
            'Total Train Euclidean Distance Error (epoch)': avg_train_euclidean_error,
            'Validation Loss (epoch)': avg_val_loss,
            'Total Validation Euclidean Distance Error (epoch)': avg_val_euclidean_error
        })

        #logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        logging.info('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_bce.item(), loss_dice.item()))
        
        
        if iter_num % 20 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, iter_num)
            outputs = torch.sigmoid(outputs).detach()
            writer.add_image('train/Prediction', outputs[1, 0, :, :].unsqueeze(0), iter_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iter_num)
        

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()

    wandb.finish()

    return "Training Finished!"