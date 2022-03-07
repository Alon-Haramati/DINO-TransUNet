# Imports
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.utils import DiceLoss
from torchvision import transforms
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_sartorius import Sartorius_dataset, RandomGenerator

# Create writer for tensorboard
writer = SummaryWriter()

# Set project path
path = '/home/burshtein2/project/TransUnet_copy'
models_path = path + '/pretrained_vit/'

# Set device
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Hyper-Parameters
base_lr = 0.01
num_classes = 2
batch_size = 25
max_epoch = 240  # 150
n_skip = 3
vit_name = 'R50-ViT-B_16'
vit_patches_size = 16
z_spacing = 1
deterministic = True

cudnn.benchmark = not deterministic
cudnn.deterministic = deterministic

# Dataset Definitions
train_base_dir = '/home/burshtein2/project/data/train_npz'
test_base_dir = '/home/burshtein2/project/data/test_npz'
list_dir = path + '/lists/lists_Sartorius'

img_size = 224  # original image size [520, 704]
rand_seed = 1234

db_train = Sartorius_dataset(base_dir=train_base_dir, list_dir=list_dir, split="train",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[img_size, img_size])]))
print("The length of train set is: {}".format(len(db_train)))


def worker_init_fn(worker_id):
    random.seed(rand_seed + worker_id)


trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                         worker_init_fn=worker_init_fn)

db_test = Sartorius_dataset(base_dir=test_base_dir, list_dir=list_dir, split="test")
testloader = DataLoader(db_test, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                        worker_init_fn=worker_init_fn)


def train_sartorius():

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations.")
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            # Write to tensorboard
            writer.add_scalar("Dice-Loss/train", loss_dice, iter_num)
            writer.add_scalar("Loss/train", loss, iter_num)

            print(f"iteration {iter_num} : loss : {loss.item()}, loss_ce: {loss_ce.item()}")

            if iter_num % 20 == 0 or iter_num == 1:
                for test_batch in testloader:
                    image_batch, label_batch = test_batch['image'], test_batch['label']
                    
                    model.to('cpu')
                    model.eval()

                    outputs = model(image_batch)

                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice

                    model.to('cuda')
                    model.train()

                    # Write to tensorboard
                    writer.add_scalar("Dice-Loss/test", loss_dice, iter_num)
                    writer.add_scalar("Loss/test", loss, iter_num)
                    print("test {} : loss : {}, loss_ce: {}".format(iter_num/20, loss.item(), loss_ce.item()))
                    break

    iterator.close()
    print("Training Finished!")


# Set random seeds for reproducibility
random.seed(rand_seed)
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = n_skip

if vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

dino_weights = torch.load(models_path+'dino_vitbase16_pretrain.pth')
res_weights = np.load(config_vit.pretrained_path)

model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
model.load_from_dino(weights=dino_weights, res_weights=res_weights)

# Train model
train_sartorius()

# Make sure all writer data was written to disk and close it
writer.flush()
writer.close()

# Save trained model
torch.save(model, path + f'/DINO-TransUNET_model-{max_epoch}_epochs_with_patches')
