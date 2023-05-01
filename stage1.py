import os
import gc
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import dataset_configs
import src.utils.custom_transformations as CT
from src.utils import pyutils, datautils, torchutils, metrics
from src.models.resnet38_mhpda import ResNet38MHPDA


class Engine:
    def __init__(
            self,
            checkpoints_path,
            train_data_path,
            init_weights_path,
            val,
            batch_size,
            num_classes,
            optimizer_lr,
            optimizer_wt_dec,
            scheduler_step_size,
            scheduler_gamma,
            device,
            max_epochs,
    ):
        self.gs = [1, 1]
        self.max_epochs = max_epochs
        self.device = device

        # Address to store weights
        self.save_weights_path = os.path.join(checkpoints_path, 'weights')

        # Create directories if not exists
        os.makedirs(name=self.save_weights_path, exist_ok=True)
        os.makedirs(name=os.path.join(checkpoints_path, 'classification', 'train'), exist_ok=True)
        os.makedirs(name=os.path.join(checkpoints_path, 'classification', 'val'), exist_ok=True)

        # Create writers
        self.train_cls_writer = SummaryWriter(log_dir=os.path.join(checkpoints_path, 'classification', 'train'))
        self.val_cls_writer = SummaryWriter(log_dir=os.path.join(checkpoints_path, 'classification', 'val'))

        # Splitting classification data into train/val
        train_cls_data, val_cls_data = datautils.split_classification_data(images_path=train_data_path, val=val)

        # Create data loaders
        self.train_cls_loader = datautils.get_LUAD_HistoSeg_classification_dataloader(
            images_path=train_data_path,
            image_names=train_cls_data,
            trans=CT.Compose([
                CT.Random90Rotation(p=0.5),
                CT.RandomHorizontalFlip(p=0.5),
                CT.Random180Rotation(p=0.5),
                CT.RandomVerticalFlip(p=0.5),
                CT.Random270Rotation(p=0.5),
                CT.ToTensor(),
                CT.RandomRemovePatch(p=0.5, num_patches=14, img_size=(224, 224), patch_size=(28, 28), batch=False)
            ]),
            batch_size=batch_size,
            shuffle=True,
        )

        self.val_cls_loader = datautils.get_LUAD_HistoSeg_classification_dataloader(
            images_path=train_data_path,
            image_names=val_cls_data,
            trans=CT.ToTensor(),
            batch_size=batch_size,
            shuffle=False,
        )

        # Get class Weights
        pw = torchutils.get_class_weights(loader=self.train_cls_loader)

        # Create model
        self.model = ResNet38MHPDA(num_classes=num_classes)
        self.model.load_weights(weights_path=init_weights_path)

        # Create optimizer
        param_groups = self.model.get_parameter_groups()
        self.optimizer = torch.optim.SGD(params=[
            {'params': param_groups[0], 'lr': optimizer_lr, 'weight_decay': optimizer_wt_dec},
            {'params': param_groups[1], 'lr': optimizer_lr * 2, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': optimizer_lr * 10, 'weight_decay': optimizer_wt_dec},
            {'params': param_groups[3], 'lr': optimizer_lr * 20, 'weight_decay': 0},
        ], lr=optimizer_lr, weight_decay=optimizer_wt_dec)

        # Create Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma
        )

        # Create criterion for loss
        self.cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')#, pos_weight=pw.to(device=self.device))

        # Create evaluator for accuracy
        self.cls_evaluator = metrics.ClassificationEvaluator(num_classes=num_classes)
        print(f'{"#" * 40}')

    def save_checkpoint(self, epoch, history):
        state_dict = {

            'global_steps': self.gs,
            'epoch': epoch,
            'max_epochs': self.max_epochs,
            'save_weights_path': self.save_weights_path,

            'train_cls_writer_logdir': self.train_cls_writer.get_logdir(),
            'train_cls_writer_logname': self.train_cls_writer.file_writer.event_writer._file_name,
            'val_cls_writer_logdir': self.val_cls_writer.get_logdir(),
            'val_cls_writer_logname': self.val_cls_writer.file_writer.event_writer._file_name,

            'train_cls_loader': self.train_cls_loader,
            'val_cls_loader': self.val_cls_loader,

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'scheduler': self.scheduler,

            'cls_criterion': self.cls_criterion,

            'cls_evaluator': self.cls_evaluator,

            'history': history
        }

        torch.save(obj=state_dict, f=os.path.join(self.save_weights_path, f'E{epoch}.pth'))

    def resume_checkpoint(self, state_dict_path=None):

        sd = torch.load(f=state_dict_path, map_location='cpu')

        self.gs = sd['global_steps']
        epoch = sd['epoch']
        self.max_epochs = sd['max_epochs']
        self.save_weights_path = sd['save_weights_path']

        self.train_cls_writer.log_dir = sd['train_cls_writer_logdir']
        self.train_cls_writer.file_writer.event_writer._file_name = sd['train_cls_writer_logname'],
        self.val_cls_writer.log_dir = sd['val_cls_writer_logdir']
        self.val_cls_writer.file_writer.event_writer._file_name = sd['val_cls_writer_logname'],

        self.train_cls_loader = sd['train_cls_loader']
        self.val_cls_loader = sd['val_cls_loader']

        self.model.load_state_dict(state_dict=sd['model_state_dict'])

        self.optimizer.load_state_dict(state_dict=sd['optimizer_state_dict'])

        self.scheduler = sd['scheduler']

        self.cls_criterion = sd['cls_criterion']

        self.cls_evaluator = sd['cls_evaluator']

        history = sd['history']

        print('Checkpoint Loaded')

        self.fit(ep=epoch + 1, epoch_history=history)

    def fit(self, ep=1, epoch_history=None):
        self.model = self.model.to(device=self.device)

        if epoch_history is None:
            epoch_history = {'train_cls': [], 'val_cls': []}

        for ep in range(ep, self.max_epochs + 1):
            """ Training Classification """
            self.model.train()
            self.cls_evaluator.reset_cms()
            epoch_history['train_cls'].append(self.train_cls_one_epoch(ep=ep))

            """ Validating Classification """
            self.model.eval()
            self.cls_evaluator.reset_cms()
            epoch_history['val_cls'].append(self.validate_cls_one_epoch(ep=ep))

            self.save_checkpoint(epoch=ep, history=epoch_history)

            self.scheduler.step()

    def train_cls_one_epoch(self, ep):

        print(f'{"#" * 20} Train Classification E{ep} {"#" * 20}')

        h = {
            'L': [],  # Loss
            'A': [],  # Accuracy
            'EM': [],  # Exact Match

            'TE_A': [],
            'NEC_A': [],
            'LYM_A': [],
            'TAS_A': [],

            'TE_L': [],
            'NEC_L': [],
            'LYM_L': [],
            'TAS_L': [],
        }
        for i in range(len(self.optimizer.param_groups)):
            h[f'LRG{i}'] = []

        t = tqdm(self.train_cls_loader)

        for batch, sample in enumerate(t):
            images = sample['img'].to(device=self.device)
            labels = sample['label'].type(torch.FloatTensor).to(device=self.device)

            logits = self.model(images).view(labels.size(0), -1)

            loss_class = self.cls_criterion(input=logits, target=labels).mean(dim=0)
            loss = loss_class.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            loss_class = loss_class.detach().cpu().numpy()
            probs = torch.sigmoid(input=logits).detach().cpu().numpy()
            labels = labels.type(torch.LongTensor).detach().cpu().numpy()

            s = self.cls_evaluator(probs=probs, target=labels)

            h['L'].append(loss)
            h['A'].append(s[0])
            h['EM'].append(s[1])

            h['TE_A'].append(s[2][0])
            h['NEC_A'].append(s[2][1])
            h['LYM_A'].append(s[2][2])
            h['TAS_A'].append(s[2][3])

            h['TE_L'].append(loss_class[0])
            h['NEC_L'].append(loss_class[1])
            h['LYM_L'].append(loss_class[2])
            h['TAS_L'].append(loss_class[3])

            self.train_cls_writer.add_scalar(tag='Batch/General/Loss', scalar_value=loss, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/General/Accuracy', scalar_value=s[0], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/General/ExactMatch', scalar_value=s[1], global_step=self.gs[0])

            self.train_cls_writer.add_scalar(tag='Batch/TE/Accuracy', scalar_value=s[2][0], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/Accuracy', scalar_value=s[2][1], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/Accuracy', scalar_value=s[2][2], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/Accuracy', scalar_value=s[2][3], global_step=self.gs[0])

            self.train_cls_writer.add_scalar(tag='Batch/TE/Loss', scalar_value=loss_class[0], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/Loss', scalar_value=loss_class[1], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/Loss', scalar_value=loss_class[2], global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/Loss', scalar_value=loss_class[3], global_step=self.gs[0])

            TN, FN, FP, TP = s[3].flatten()
            self.train_cls_writer.add_scalar(tag='Batch/TE/TN', scalar_value=TN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/FN', scalar_value=FN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/FP', scalar_value=FP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/TP', scalar_value=TP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[0])

            TN, FN, FP, TP = s[4].flatten()
            self.train_cls_writer.add_scalar(tag='Batch/NEC/TN', scalar_value=TN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/FN', scalar_value=FN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/FP', scalar_value=FP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/TP', scalar_value=TP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[0])

            TN, FN, FP, TP = s[5].flatten()
            self.train_cls_writer.add_scalar(tag='Batch/LYM/TN', scalar_value=TN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/FN', scalar_value=FN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/FP', scalar_value=FP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/TP', scalar_value=TP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[0])

            TN, FN, FP, TP = s[6].flatten()
            self.train_cls_writer.add_scalar(tag='Batch/TAS/TN', scalar_value=TN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/FN', scalar_value=FN, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/FP', scalar_value=FP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/TP', scalar_value=TP, global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[0])

            for i, p in enumerate(self.optimizer.param_groups):
                self.train_cls_writer.add_scalar(tag=f'Batch/LRG{i}', scalar_value=p['lr'], global_step=self.gs[0])
                h[f'LRG{i}'].append(p['lr'])

            gc.collect()
            torch.cuda.empty_cache()

            self.gs[0] += 1

            t.set_description(desc=f"E{ep} Train Loss: {loss:0.4f}, Acc: {s[0]:0.4f}, EM: {s[1]:0.4f}")

        self.train_cls_writer.add_scalar(tag='Epoch/Loss', scalar_value=np.mean(h['L']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/Accuracy', scalar_value=np.mean(h['A']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/ExactMatch', scalar_value=np.mean(h['EM']), global_step=ep)

        self.train_cls_writer.add_scalar(tag='Epoch/TE/Accuracy', scalar_value=np.mean(h['TE_A']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/Accuracy', scalar_value=np.mean(h['NEC_A']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/Accuracy', scalar_value=np.mean(h['LYM_A']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/Accuracy', scalar_value=np.mean(h['TAS_A']), global_step=ep)

        self.train_cls_writer.add_scalar(tag='Epoch/TE/Loss', scalar_value=np.mean(h['TE_L']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/Loss', scalar_value=np.mean(h['NEC_L']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/Loss', scalar_value=np.mean(h['LYM_L']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/Loss', scalar_value=np.mean(h['TAS_L']), global_step=ep)

        TE_cm, NEC_cm, LYM_cm, TAS_cm = self.cls_evaluator.get_cms()

        TN, FN, FP, TP = TE_cm.flatten()
        self.train_cls_writer.add_scalar(tag='Epoch/TE/TN', scalar_value=TN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/FN', scalar_value=FN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/FP', scalar_value=FP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/TP', scalar_value=TP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = NEC_cm.flatten()
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/TN', scalar_value=TN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/FN', scalar_value=FN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/FP', scalar_value=FP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/TP', scalar_value=TP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = LYM_cm.flatten()
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/TN', scalar_value=TN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/FN', scalar_value=FN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/FP', scalar_value=FP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/TP', scalar_value=TP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = TAS_cm.flatten()
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/TN', scalar_value=TN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/FN', scalar_value=FN, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/FP', scalar_value=FP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/TP', scalar_value=TP, global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        h['TE_cm'] = TE_cm
        h['NEC_cm'] = NEC_cm
        h['LYM_cm'] = LYM_cm
        h['TAS_cm'] = TAS_cm

        return h

    def validate_cls_one_epoch(self, ep):

        print(f'{"#" * 20} Validating Classification E{ep} {"#" * 20}')

        h = {
            'L': [],  # Loss
            'A': [],  # Accuracy
            'EM': [],  # Exact Match

            'TE_A': [],
            'NEC_A': [],
            'LYM_A': [],
            'TAS_A': [],

            'TE_L': [],
            'NEC_L': [],
            'LYM_L': [],
            'TAS_L': [],
        }

        t = tqdm(self.val_cls_loader)

        for batch, sample in enumerate(t):
            images = sample['img'].to(device=self.device)
            labels = sample['label'].type(torch.FloatTensor).to(device=self.device)

            with torch.no_grad():
                logits = self.model(images).view(labels.size(0), -1)

            loss_class = self.cls_criterion(input=logits, target=labels).mean(dim=0)

            loss_class = loss_class.detach().cpu().numpy()
            loss = loss_class.sum()
            probs = torch.sigmoid(input=logits).detach().cpu().numpy()
            labels = labels.type(torch.LongTensor).detach().cpu().numpy()

            s = self.cls_evaluator(probs=probs, target=labels)

            h['L'].append(loss)
            h['A'].append(s[0])
            h['EM'].append(s[1])

            h['TE_A'].append(s[2][0])
            h['NEC_A'].append(s[2][1])
            h['LYM_A'].append(s[2][2])
            h['TAS_A'].append(s[2][3])

            h['TE_L'].append(loss_class[0])
            h['NEC_L'].append(loss_class[1])
            h['LYM_L'].append(loss_class[2])
            h['TAS_L'].append(loss_class[3])

            self.val_cls_writer.add_scalar(tag='Batch/General/Loss', scalar_value=loss, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/General/Accuracy', scalar_value=s[0], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/General/ExactMatch', scalar_value=s[1], global_step=self.gs[1])

            self.val_cls_writer.add_scalar(tag='Batch/TE/Accuracy', scalar_value=s[2][0], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/Accuracy', scalar_value=s[2][1], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/Accuracy', scalar_value=s[2][2], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/Accuracy', scalar_value=s[2][3], global_step=self.gs[1])

            self.val_cls_writer.add_scalar(tag='Batch/TE/Loss', scalar_value=loss_class[0], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/Loss', scalar_value=loss_class[1], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/Loss', scalar_value=loss_class[2], global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/Loss', scalar_value=loss_class[3], global_step=self.gs[1])

            TN, FN, FP, TP = s[3].flatten()
            self.val_cls_writer.add_scalar(tag='Batch/TE/TN', scalar_value=TN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/FN', scalar_value=FN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/FP', scalar_value=FP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/TP', scalar_value=TP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[1])

            TN, FN, FP, TP = s[4].flatten()
            self.val_cls_writer.add_scalar(tag='Batch/NEC/TN', scalar_value=TN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/FN', scalar_value=FN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/FP', scalar_value=FP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/TP', scalar_value=TP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[1])

            TN, FN, FP, TP = s[5].flatten()
            self.val_cls_writer.add_scalar(tag='Batch/LYM/TN', scalar_value=TN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/FN', scalar_value=FN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/FP', scalar_value=FP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/TP', scalar_value=TP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[1])

            TN, FN, FP, TP = s[6].flatten()
            self.val_cls_writer.add_scalar(tag='Batch/TAS/TN', scalar_value=TN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/FN', scalar_value=FN, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/FP', scalar_value=FP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/TP', scalar_value=TP, global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/SEN', scalar_value=TP / (TP + FN), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/SPE', scalar_value=TN / (TN + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/PPV', scalar_value=TP / (TP + FP), global_step=self.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS/NPV', scalar_value=TN / (TN + FN), global_step=self.gs[1])

            gc.collect()
            torch.cuda.empty_cache()

            t.set_description(desc=f"E{ep} Val Loss: {loss:0.4f}, Acc: {s[0]:0.4f}, EM: {s[1]:0.4f}")

            self.gs[1] += 1

        self.val_cls_writer.add_scalar(tag='Epoch/Loss', scalar_value=np.mean(h['L']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/Accuracy', scalar_value=np.mean(h['A']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/ExactMatch', scalar_value=np.mean(h['EM']), global_step=ep)

        self.val_cls_writer.add_scalar(tag='Epoch/TE/Accuracy', scalar_value=np.mean(h['TE_A']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/Accuracy', scalar_value=np.mean(h['NEC_A']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/Accuracy', scalar_value=np.mean(h['LYM_A']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/Accuracy', scalar_value=np.mean(h['TAS_A']), global_step=ep)

        self.val_cls_writer.add_scalar(tag='Epoch/TE/Loss', scalar_value=np.mean(h['TE_L']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/Loss', scalar_value=np.mean(h['NEC_L']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/Loss', scalar_value=np.mean(h['LYM_L']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/Loss', scalar_value=np.mean(h['TAS_L']), global_step=ep)

        TE_cm, NEC_cm, LYM_cm, TAS_cm = self.cls_evaluator.get_cms()

        TN, FN, FP, TP = TE_cm.flatten()
        self.val_cls_writer.add_scalar(tag='Epoch/TE/TN', scalar_value=TN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/FN', scalar_value=FN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/FP', scalar_value=FP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/TP', scalar_value=TP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = NEC_cm.flatten()
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/TN', scalar_value=TN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/FN', scalar_value=FN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/FP', scalar_value=FP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/TP', scalar_value=TP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = LYM_cm.flatten()
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/TN', scalar_value=TN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/FN', scalar_value=FN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/FP', scalar_value=FP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/TP', scalar_value=TP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        TN, FN, FP, TP = TAS_cm.flatten()
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/TN', scalar_value=TN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/FN', scalar_value=FN, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/FP', scalar_value=FP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/TP', scalar_value=TP, global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/SEN', scalar_value=TP / (TP + FN), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/SPE', scalar_value=TN / (TN + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/PPV', scalar_value=TP / (TP + FP), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS/NPV', scalar_value=TN / (TN + FN), global_step=ep)

        h['TE_cm'] = TE_cm
        h['NEC_cm'] = NEC_cm
        h['LYM_cm'] = LYM_cm
        h['TAS_cm'] = TAS_cm

        return h


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=os.getcwd(), type=str)
    parser.add_argument("--session", default="Stage1WeightedBCE", type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--val', default=0.2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--init_weights_path", default='/home/naeim_md93/Projects/WSSS/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--optimizer_lr", default=1e-4, type=float)
    parser.add_argument("--optimizer_wt_dec", default=5e-4, type=float)
    parser.add_argument("--scheduler_step_size", default=1, type=float)
    parser.add_argument("--scheduler_gamma", default=0.9, type=float)
    parser.add_argument("--max_epochs", default=40, type=int)
    args = parser.parse_args()

    pyutils.set_seed(seed=args.seed)

    args.dataset_path = os.path.join(args.root_path, 'datasets', 'LUAD-HistoSeg')
    args.checkpoints_path = os.path.join(args.root_path, 'checkpoints', args.session)

    args.train_data_path = os.path.join(args.dataset_path, 'train')
    args.val_data_path = os.path.join(args.dataset_path, 'val')
    args.test_data_path = os.path.join(args.dataset_path, 'test')

    args.luad_configs = dataset_configs.get_LUAD_HistoSeg_configs(dataset_path=args.dataset_path)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    print(args)

    datautils.check_LUAD_HistoSeg_dataset(dataset_configs=args.luad_configs)

    engine = Engine(
        checkpoints_path=args.checkpoints_path,
        train_data_path=args.train_data_path,
        val=args.val,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        init_weights_path=args.init_weights_path,
        optimizer_lr=args.optimizer_lr,
        optimizer_wt_dec=args.optimizer_wt_dec,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        device=args.device,
        max_epochs=args.max_epochs
    )

    print('Training from scratch')
    engine.fit()
