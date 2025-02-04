import numpy as np
from msHead_3D.network_backbone import MSHEAD_ATTN
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
from monai.losses import DiceCELoss
set_determinism(123)

import datetime
import time
import os

data_dir = "./data/fullres/train"
logdir = f"./logs/segmamba"
# model_name = "model_loss_dice_opt_adamw"
model_name = "segmamba_setup"
data_list_path = f"./data_list"

# run_id = datetime.datetime.today().strftime('%m-%d-%y_%H%M')
# print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')

# logdir = os.path.join(logdir, "model_upsample_inside_wd_1e-5_4_gpu")
logdir = os.path.join(logdir, model_name)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(data_list_path):
    os.makedirs(data_list_path)

    
# augmentation = "nomirror"
augmentation = True

# env = "pytorch"
env = "DDP"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 4
device = "cuda:0"
roi_size = [128, 128, 128]

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, train_process = 12, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        # from model_segmamba.segmamba import SegMamba
        out_classes = 4
        self.model = MSHEAD_ATTN(
            img_size=(128, 128, 128),
            patch_size=2,
            in_chans=4,
            out_chans=out_classes,
            depths=[2,2,2,2],
            feat_size=[48,96,192,384],
            num_heads = [3,6,12,24],
            drop_path_rate=0.1,
            use_checkpoint=False,
        )

        # self.model = SegMamba(in_chans=4,
        #                 out_chans=4,
        #                 depths=[2,2,2,2],
        #                 feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.train_process = train_process
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=1e-5,
                                    momentum=0.99, nesterov=True)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        
        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()
        # self.dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        # print(f'########### in training step image:{image.shape} label:{label.shape} ###################')
        # unique_values = torch.unique(label)
        # print(f'in trainng unique values: {unique_values}')
        pred = self.model(image)
        # print(f'pred:{pred.shape}')

        loss = self.cross(pred, label)
        # print(f' ------------- loss:{loss} global step:{self.global_step} ------------- ')
        self.log("training_loss", loss, step=self.global_step)

        return loss 
    
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
    
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image)
        output = output.argmax(dim=1)
        
        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

        print(f"dices is {tc, wt, et}")

        mean_dice = (tc + wt + et) / 3 
        
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)
        print(f'####### Epoch:{self.epoch} ---- mean dice:{mean_dice} ###############')

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, self.optimizer, mean_dice,
                                            os.path.join(logdir, 
                                            f"best_model_{mean_dice:.4f}.pth"), 
                                            scheduler=self.scheduler,
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, self.optimizer, mean_dice,
                                        os.path.join(logdir, 
                                        f"final_model_{mean_dice:.4f}.pth"), 
                                        scheduler=self.scheduler,
                                        delete_symbol="final_model")


        if (self.epoch + 1) % 100 == 0:
            if self.scheduler is not None:
                save_state = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.scheduler.state_dict(),
                    'dice_score': mean_dice}
            else:
                save_state = {'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'dice_score': mean_dice}

            torch.save(save_state, os.path.join(logdir, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pth"))

        print(f"mean_dice is {mean_dice}")

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            train_process=12,
                            master_port=17759,
                            training_script=__file__)

    # split_path = model_name
    split_path = "model_loss_dice_opt_adamw"
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir, data_list_path, split_path)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
