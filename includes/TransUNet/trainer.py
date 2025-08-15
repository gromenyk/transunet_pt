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
from includes.TransUNet.utils import DiceLoss
from torchvision import transforms
from scipy.ndimage import center_of_mass

def euclidean_distance(real, pred):
    return torch.sqrt(torch.sum((real - pred) ** 2, dim=1)).mean()


def find_centers_of_mass_for_hottest_pixels(prediction):
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
    from includes.TransUNet.datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = 1
    batch_size = args.batch_size * args.n_gpu
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=None) 

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    scale_x = 512 / 224
    scale_y = 512 / 224

    ce_loss = nn.BCELoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001) 
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader) 
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

            predicted_coords = []
            for output in outputs:
                pred_distal, pred_proximal = find_centers_of_mass_for_hottest_pixels(output.squeeze().cpu().detach().numpy())
                pred_coords = torch.tensor([[pred_distal[1] * scale_x, pred_distal[0] * scale_y], 
                                            [pred_proximal[1] * scale_x, pred_proximal[0] * scale_y]], requires_grad=True)
                predicted_coords.append(pred_coords)

            predicted_coords = torch.stack(predicted_coords).cuda()
            coords_batch = coords_batch.cuda().float()
            loss_euclidean = euclidean_distance(coords_batch, predicted_coords)

            total_train_loss += loss.item()	

            train_euclidean_error = euclidean_distance(coords_batch, predicted_coords)
            total_euclidean_error += train_euclidean_error.item()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            print(f"Epoch {epoch_num} | Iteration {iter_num} | Train Loss: {loss.item():.4f} | BCE Loss: {loss_bce.item():.4f} | Euclidean Error: {train_euclidean_error.item():.4f}")

        avg_train_loss = total_train_loss / len(trainloader)
        avg_train_euclidean_error = total_euclidean_error / len(trainloader)    
    
        model.eval()
       
        logging.info('iteration %d : loss : %f, loss_bce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_bce.item(), loss_dice.item()))
        
        
        if iter_num % 20 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, iter_num)
            outputs = torch.sigmoid(outputs).detach()
            writer.add_image('train/Prediction', outputs[1, 0, :, :].unsqueeze(0), iter_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iter_num)
        

        save_interval = 50  
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

    return "Training Finished!"