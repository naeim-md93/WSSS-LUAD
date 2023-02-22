import gc
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import dataset_configs
import src.utils.custom_transformations as CT
from src.utils import datautils, pyutils, torchutils, metrics
from src.models.resnet38d_cls import ResNet38ClassificationModel
cudnn.enabled = True


class Engine:
    def __init__(self, args):
        self.args = args
        self.args.gs = [1, 1]

        os.makedirs(name=self.args.train_cls_log_path, exist_ok=True)
        os.makedirs(name=self.args.val_cls_log_path, exist_ok=True)
        os.makedirs(name=self.args.val_cam_log_path, exist_ok=True)
        os.makedirs(name=self.args.test_cam_log_path, exist_ok=True)

        self.train_cls_writer = SummaryWriter(log_dir=self.args.train_cls_log_path)
        self.val_cls_writer = SummaryWriter(log_dir=self.args.val_cls_log_path)
        self.val_cam_writer = SummaryWriter(log_dir=self.args.val_cam_log_path)
        self.test_cam_writer = SummaryWriter(log_dir=self.args.test_cam_log_path)

        train_cls_data, val_cls_data = datautils.split_classification_data(
            images_path=self.args.train_data_path,
            split_size=self.args.val_size,
        )

        self.train_cls_loader = datautils.get_LUAD_HistoSeg_classification_dataloader(
            images_path=self.args.train_data_path,
            image_names=train_cls_data,
            trans=CT.Compose([
                CT.RandomHorizontalFlip(p=0.5),
                CT.RandomVerticalFlip(p=0.5),
                CT.Random90Rotation(p=0.5),
                CT.Random180Rotation(p=0.5),
                CT.Random270Rotation(p=0.5),
                CT.ToTensor(),
            ]),
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        self.val_cls_loader = datautils.get_LUAD_HistoSeg_classification_dataloader(
            images_path=self.args.train_data_path,
            image_names=val_cls_data,
            trans=CT.Compose([
                CT.ToTensor(),
            ]),
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.val_cam_loader = datautils.get_LUAD_HistoSeg_segmentation_dataloader(
            images_path=os.path.join(self.args.val_data_path, 'img'),
            masks_path=os.path.join(self.args.val_data_path, 'mask'),
            trans=CT.Compose([
                CT.ToTensor(),
            ]),
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.test_cam_loader = datautils.get_LUAD_HistoSeg_segmentation_dataloader(
            images_path=os.path.join(self.args.test_data_path, 'img'),
            masks_path=os.path.join(self.args.test_data_path, 'mask'),
            trans=CT.Compose([
                CT.ToTensor(),
            ]),
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.model = ResNet38ClassificationModel(num_classes=self.args.num_classes)
        self.train_cls_writer.add_graph(model=self.model, input_to_model=torch.Tensor(torch.zeros(size=(1, 3, 224, 224))))

        if hasattr(self.args, 'init_weights'):
            if args.init_weights is not None:
                wp = os.path.join(self.args.init_weights_path, self.args.init_weights)
                if self.args.init_weights[-7:] == '.params':
                    weights_dict = torchutils.convert_mxnet_weights_to_torch(weights_path=wp)

                    # Strict=False because of linear1000 and fc8
                    self.model.load_state_dict(state_dict=weights_dict, strict=False)
                    print('Initialize model with MXNet weights')

                elif self.args.init_weights[-4:] == '.pth':
                    weights_dict = torch.load(f=wp, map_location='cpu')
                    self.model.load_state_dict(state_dict=weights_dict, strict=True)
                    print('Initialize model with user-defined weights')

                else:
                    raise NotImplementedError('Invalid model weights')
        else:
            print('Initialize model with random weights')

        self.model = self.model.to(device=self.args.device)

        param_groups = self.model.get_parameter_groups()
        self.optimizer = torch.optim.SGD(params=[
            {'params': param_groups[0], 'lr': self.args.lr, 'weight_decay': self.args.wt_dec},
            {'params': param_groups[1], 'lr': self.args.lr * 2, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': self.args.lr * 10, 'weight_decay': self.args.wt_dec},
            {'params': param_groups[3], 'lr': self.args.lr * 20, 'weight_decay': 0},
        ], lr=self.args.lr, weight_decay=self.args.wt_dec)

        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.85)

        self.cls_criterion = metrics.WeightedMultiLabelSoftMarginLoss()
        self.cls_evaluator = metrics.IoUAccuracy()
        self.cam_evaluator = metrics.PseudoMaskEvaluator(num_classes=self.args.num_classes)

    def resume_checkpoint(self, state_dict_path=None):

        state_dict = torch.load(f=state_dict_path, map_location='cpu')
        session = state_dict['session_name']
        assert session == self.args.session_name, f"State dict {session} is not for session {self.args.session_name}"
        print('Checkpoint Loaded')

        epoch = state_dict['epoch']
        self.args.gs = state_dict['global_steps']

        self.train_cls_writer.log_dir = state_dict['train_cls_writer_logdir']
        self.train_cls_writer.file_writer.event_writer._file_name = state_dict['train_cls_writer_logname'],
        self.val_cls_writer.log_dir = state_dict['val_cls_writer_logdir']
        self.val_cls_writer.file_writer.event_writer._file_name = state_dict['val_cls_writer_logname'],
        self.val_cam_writer.log_dir = state_dict['val_cam_writer_logdir']
        self.val_cam_writer.file_writer.event_writer._file_name = state_dict['val_cam_writer_logname'],
        self.test_cam_writer.log_dir = state_dict['test_cam_writer_logdir']
        self.test_cam_writer.file_writer.event_writer._file_name = state_dict['test_cam_writer_logname'],

        self.train_cls_loader = state_dict['train_cls_loader']
        self.val_cls_loader = state_dict['val_cls_loader']
        self.val_cam_loader = state_dict['val_cam_loader']
        self.test_cam_loader = state_dict['test_cam_loader']

        self.model.load_state_dict(state_dict=state_dict['model_state_dict'])
        # self.model.mu = state_dict['model_mu']
        # self.model.enable_PDA = state_dict['model_enable_PDA']

        self.optimizer.load_state_dict(state_dict=state_dict['optimizer_state_dict'])

        history = state_dict['history']
        self.fit(ep=epoch + 1, epoch_history=history)

    def save_checkpoint(self, epoch, history, state_dict_path):
        state_dict = {
            'session_name': self.args.session_name,
            'epoch': epoch,
            'global_steps': self.args.gs,

            'train_cls_writer_logdir': self.train_cls_writer.get_logdir(),
            'train_cls_writer_logname': self.train_cls_writer.file_writer.event_writer._file_name,
            'val_cls_writer_logdir': self.val_cls_writer.get_logdir(),
            'val_cls_writer_logname': self.val_cls_writer.file_writer.event_writer._file_name,
            'val_cam_writer_logdir': self.val_cam_writer.get_logdir(),
            'val_cam_writer_logname': self.val_cam_writer.file_writer.event_writer._file_name,
            'test_cam_writer_logdir': self.test_cam_writer.get_logdir(),
            'test_cam_writer_logname': self.test_cam_writer.file_writer.event_writer._file_name,

            'train_cls_loader': self.train_cls_loader,
            'val_cls_loader': self.val_cls_loader,
            'val_cam_loader': self.val_cam_loader,
            'test_cam_loader': self.test_cam_loader,

            'model_state_dict': self.model.state_dict(),
            # 'model_mu': self.model.mu,
            # 'model_enable_PDA': self.model.enable_PDA,

            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': history
        }

        torch.save(obj=state_dict, f=state_dict_path)

    def fit(self, ep=1, epoch_history=None):

        wp = os.path.join(self.args.checkpoints_path, 'weights')
        os.makedirs(name=wp, exist_ok=True)

        if epoch_history is None:
            epoch_history = {
                'train_cls': [],
                'val_cls': [],
                'val_cam': [],
                'test_cam': [],
            }

        for ep in range(ep, self.args.max_epochs + 1):
            """ Training Classification """
            self.model.train()

            ########################################################
            #        Start of Progressive Dropout Attention
            ########################################################
            # if ep >= self.args.start_PDA:
            #     self.model.enable_PDA = True
            #
            #     # init_mu = 1
            #     if self.model.mu > self.args.l:  # mu > 0.65
            #         self.model.mu = self.model.mu * self.args.sigma  # mu = mu * 0.985
            # else:
            #     self.model.enable_PDA = False
            ########################################################
            #        End of Progressive Dropout Attention
            ########################################################

            epoch_history['train_cls'].append(self.train_cls_one_epoch(ep=ep, print_incorrects=True))

            """ Validating Classification """
            self.model.eval()
            self.model.enable_PDA = False
            epoch_history['val_cls'].append(self.validate_cls_one_epoch(ep=ep, print_incorrects=True))

            """ Validating CAMs """
            self.cam_evaluator.reset()
            epoch_history['val_cam'].append(self.validate_cam_one_epoch(ep=ep))

            """ Testing CAMs """
            self.cam_evaluator.reset()
            epoch_history['test_cam'].append(self.test_cam_one_epoch(ep=ep))

            self.scheduler.step()

            wn = f'{self.args.session_name}_E{ep}_checkpoint_trained_on_luad.pth'
            self.save_checkpoint(epoch=ep, history=epoch_history, state_dict_path=os.path.join(wp, wn))

    def train_cls_one_epoch(self, ep, thresh=0.5, print_incorrects=False):
        print(f'{"#" * 20} Train Classification E{ep} {"#" * 20}')

        h = {
            'MLSMLoss': [],
            'IoUAccuracy': [],
            'ExactMatch': [],
            'TE_IoUAccuracy': [],
            'NEC_IoUAccuracy': [],
            'LYM_IoUAccuracy': [],
            'TAS_IoUAccuracy': [],
            'TE_MLSMLoss': [],
            'NEC_MLSMLoss': [],
            'LYM_MLSMLoss': [],
            'TAS_MLSMLoss': [],
        }
        for i, params in enumerate(self.optimizer.param_groups):
            h[f'LRG{i}'] = []

        t = tqdm(self.train_cls_loader)

        for iter, sample in enumerate(t):
            names = sample['name']
            images = sample['img'].to(device=self.args.device)
            labels = sample['label'].to(device=self.args.device)

            x = self.model(images)
            x = x.view(x.size(0), -1)

            loss_class = self.cls_criterion.train_call(input=x, target=labels)

            loss = loss_class.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()
            loss = loss.item()
            loss_class = loss_class.detach().cpu().numpy()
            probs = torch.sigmoid(input=x).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            scores = self.cls_evaluator(probs=probs, y=labels)

            if print_incorrects:
                torchutils.print_incorrects(names=names, probs=probs, labels=labels, thresh=thresh)

            h['MLSMLoss'].append(loss)
            h['IoUAccuracy'].append(scores['accuracy'])
            h['ExactMatch'].append(scores['exact_match'])
            h['TE_IoUAccuracy'].append(scores['class_acc'][0])
            h['NEC_IoUAccuracy'].append(scores['class_acc'][1])
            h['LYM_IoUAccuracy'].append(scores['class_acc'][2])
            h['TAS_IoUAccuracy'].append(scores['class_acc'][3])
            h['TE_MLSMLoss'].append(loss_class[0])
            h['NEC_MLSMLoss'].append(loss_class[1])
            h['LYM_MLSMLoss'].append(loss_class[2])
            h['TAS_MLSMLoss'].append(loss_class[3])

            self.train_cls_writer.add_scalar(tag='Batch/MLSMLoss', scalar_value=loss, global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/IoUAccuracy', scalar_value=scores['accuracy'], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/ExactMatch', scalar_value=scores['exact_match'], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE_IoUAccuracy', scalar_value=scores['class_acc'][0], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC_IoUAccuracy', scalar_value=scores['class_acc'][1], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM_IoUAccuracy', scalar_value=scores['class_acc'][2], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS_IoUAccuracy', scalar_value=scores['class_acc'][3], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TE_MLSMLoss', scalar_value=loss_class[0], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/NEC_MLSMLoss', scalar_value=loss_class[1], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/LYM_MLSMLoss', scalar_value=loss_class[2], global_step=self.args.gs[0])
            self.train_cls_writer.add_scalar(tag='Batch/TAS_MLSMLoss', scalar_value=loss_class[3], global_step=self.args.gs[0])

            for i, params in enumerate(self.optimizer.param_groups):
                self.train_cls_writer.add_scalar(tag=f'Batch/LRG{i}', scalar_value=params['lr'], global_step=self.args.gs[0])
                h[f'LRG{i}'].append(params['lr'])

            self.args.gs[0] += 1

            t.set_description(
                desc=f"E{ep} Train Loss: {loss:0.4f}, "
                f"Acc: {scores['accuracy']:0.4f}, "
                f"EM: {scores['exact_match']:0.4f}, "
                f"Mu: {self.model.mu:0.4f}, "
                f"PDA: {self.model.enable_PDA}"
            )

        self.train_cls_writer.add_scalar(tag='Epoch/MLSMLoss', scalar_value=np.nanmean(h['MLSMLoss']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/IoUAccuracy', scalar_value=np.nanmean(h['IoUAccuracy']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/ExactMatch', scalar_value=np.nanmean(h['ExactMatch']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE_IoUAccuracy', scalar_value=np.nanmean(h['TE_IoUAccuracy']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC_IoUAccuracy', scalar_value=np.nanmean(h['NEC_IoUAccuracy']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM_IoUAccuracy', scalar_value=np.nanmean(h['LYM_IoUAccuracy']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS_IoUAccuracy', scalar_value=np.nanmean(h['TAS_IoUAccuracy']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TE_MLSMLoss', scalar_value=np.nanmean(h['TE_MLSMLoss']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/NEC_MLSMLoss', scalar_value=np.nanmean(h['NEC_MLSMLoss']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/LYM_MLSMLoss', scalar_value=np.nanmean(h['LYM_MLSMLoss']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/TAS_MLSMLoss', scalar_value=np.nanmean(h['TAS_MLSMLoss']), global_step=ep)
        self.train_cls_writer.add_scalar(tag='Epoch/Mu', scalar_value=self.model.mu, global_step=ep)

        for k, v in h.items():
            print(f'Train Classification {k}: {np.nanmean(h[k]):0.4f}')
        print(f'Train Classification model_mu: {self.model.mu}')
        print(f'Train Classification model_enable_PDA: {self.model.enable_PDA}')
        return h

    def validate_cls_one_epoch(self, ep, thresh=0.5, print_incorrects=False):
        print(f'{"#" * 20} Validating Classification E{ep} {"#" * 20}')

        h = {
            'MLSMLoss': [],
            'IoUAccuracy': [],
            'ExactMatch': [],
            'TE_IoUAccuracy': [],
            'NEC_IoUAccuracy': [],
            'LYM_IoUAccuracy': [],
            'TAS_IoUAccuracy': [],
            'TE_MLSMLoss': [],
            'NEC_MLSMLoss': [],
            'LYM_MLSMLoss': [],
            'TAS_MLSMLoss': [],
        }

        t = tqdm(self.val_cls_loader)

        for iter, sample in enumerate(t):
            names = sample['name']
            images = sample['img'].to(device=self.args.device)
            labels = sample['label'].to(device=self.args.device)

            with torch.no_grad():
                x = self.model(images)
                x = x.view(x.size(0), -1)

            loss_class = self.cls_criterion.val_call(input=x, target=labels)

            gc.collect()
            torch.cuda.empty_cache()

            loss_class = loss_class.detach().cpu().numpy()
            loss = loss_class.mean()
            probs = torch.sigmoid(input=x).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            scores = self.cls_evaluator(probs=probs, y=labels)
            if print_incorrects:
                torchutils.print_incorrects(names=names, probs=probs, labels=labels, thresh=thresh)

            h['MLSMLoss'].append(loss)
            h['IoUAccuracy'].append(scores['accuracy'])
            h['ExactMatch'].append(scores['exact_match'])
            h['TE_IoUAccuracy'].append(scores['class_acc'][0])
            h['NEC_IoUAccuracy'].append(scores['class_acc'][1])
            h['LYM_IoUAccuracy'].append(scores['class_acc'][2])
            h['TAS_IoUAccuracy'].append(scores['class_acc'][3])
            h['TE_MLSMLoss'].append(loss_class[0])
            h['NEC_MLSMLoss'].append(loss_class[1])
            h['LYM_MLSMLoss'].append(loss_class[2])
            h['TAS_MLSMLoss'].append(loss_class[3])

            t.set_description(
                desc=f"E{ep} Val Loss: {loss:0.4f}, "
                f"Acc: {scores['accuracy']:0.4f}, "
                f"EM: {scores['exact_match']:0.4f}, "
                f"Mu: {self.model.mu:0.4f}, "
                f"PDA: {self.model.enable_PDA}"
            )

            self.val_cls_writer.add_scalar(tag='Batch/MLSMLoss', scalar_value=loss, global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/IoUAccuracy', scalar_value=scores['accuracy'], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/ExactMatch', scalar_value=scores['exact_match'], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE_IoUAccuracy', scalar_value=scores['class_acc'][0], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC_IoUAccuracy', scalar_value=scores['class_acc'][1], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM_IoUAccuracy', scalar_value=scores['class_acc'][2], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS_IoUAccuracy', scalar_value=scores['class_acc'][3], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TE_MLSMLoss', scalar_value=loss_class[0], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/NEC_MLSMLoss', scalar_value=loss_class[1], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/LYM_MLSMLoss', scalar_value=loss_class[2], global_step=self.args.gs[1])
            self.val_cls_writer.add_scalar(tag='Batch/TAS_MLSMLoss', scalar_value=loss_class[3], global_step=self.args.gs[1])

            self.args.gs[1] += 1

        self.val_cls_writer.add_scalar(tag='Epoch/MLSMLoss', scalar_value=np.nanmean(h['MLSMLoss']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/IoUAccuracy', scalar_value=np.nanmean(h['IoUAccuracy']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/ExactMatch', scalar_value=np.nanmean(h['ExactMatch']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE_IoUAccuracy', scalar_value=np.nanmean(h['TE_IoUAccuracy']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC_IoUAccuracy', scalar_value=np.nanmean(h['NEC_IoUAccuracy']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM_IoUAccuracy', scalar_value=np.nanmean(h['LYM_IoUAccuracy']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS_IoUAccuracy', scalar_value=np.nanmean(h['TAS_IoUAccuracy']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TE_MLSMLoss', scalar_value=np.nanmean(h['TE_MLSMLoss']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/NEC_MLSMLoss', scalar_value=np.nanmean(h['NEC_MLSMLoss']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/LYM_MLSMLoss', scalar_value=np.nanmean(h['LYM_MLSMLoss']), global_step=ep)
        self.val_cls_writer.add_scalar(tag='Epoch/TAS_MLSMLoss', scalar_value=np.nanmean(h['TAS_MLSMLoss']), global_step=ep)

        for k, v in h.items():
            print(f'Validate Classification {k}: {np.nanmean(h[k]):0.4f}')
        print(f'Validate Classification model_mu: {self.model.mu}')
        print(f'Validate Classification model_enable_PDA: {self.model.enable_PDA}')

        return h

    def validate_cam_one_epoch(self, ep=1, threshold=0.25):
        print(f'{"#" * 20} Validating CAMs E{ep} {"#" * 20}')

        t = tqdm(self.val_cam_loader, desc=f'E{ep} Validating CAMs')

        for iter, sample in enumerate(t):
            names = sample['name']
            images = sample['img'].to(device=self.args.device)
            masks = sample['mask'].numpy()

            b, c, h, w = images.size()

            with torch.no_grad():
                cams, labels = self.model.forward_cam(images)

            cams = cams.detach().cpu()
            labels = labels.detach().cpu()

            labels = torch.greater(input=labels, other=threshold)
            cams = F.interpolate(input=cams, size=(h, w), mode='bilinear', align_corners=False)

            cams = cams.numpy()
            labels = labels.numpy()

            cams = cams * labels
            cams = np.argmax(a=cams, axis=1).astype(dtype=np.uint8)

            cams[masks == 4] = 4
            self.cam_evaluator.add_batch(gt_mask=masks, pred_mask=cams)

        scores = self.cam_evaluator.get_scores()

        self.val_cam_writer.add_scalar(tag='Epoch/PixelAccuracy', scalar_value=scores['pa'], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/MeanClassAccuracy', scalar_value=scores['ma'], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/TE_IoUAccuracy', scalar_value=scores['iou'][0], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/NEC_IoUAccuracy', scalar_value=scores['iou'][1], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/LYM_IoUAccuracy', scalar_value=scores['iou'][2], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/TAS_IoUAccuracy', scalar_value=scores['iou'][3], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/MIoU', scalar_value=scores['miou'], global_step=ep)
        self.val_cam_writer.add_scalar(tag='Epoch/FWIoU', scalar_value=scores['fwiou'], global_step=ep)

        for k, v in scores.items():
            print(f'Validate CAMs {k}: {np.nanmean(scores[k]):0.4f}')
        print(f'Validate CAMs model_mu: {self.model.mu}')
        print(f'Validate CAMs model_enable_PDA: {self.model.enable_PDA}')

        return scores

    def test_cam_one_epoch(self, ep=1, threshold=0.25):
        print(f'{"#" * 20} Testing CAMs E{ep} {"#" * 20}')

        t = tqdm(self.test_cam_loader, desc=f'E{ep} Testing CAMs')

        for iter, sample in enumerate(t):
            names = sample['name']
            images = sample['img'].to(device=self.args.device)
            masks = sample['mask'].numpy()

            b, c, h, w = images.size()

            with torch.no_grad():
                cams, labels = self.model.forward_cam(images)

            cams = cams.detach().cpu()
            labels = labels.detach().cpu()

            labels = torch.greater(input=labels, other=threshold)
            cams = F.interpolate(input=cams, size=(h, w), mode='bilinear', align_corners=False)

            cams = cams.numpy()
            labels = labels.numpy()

            cams = cams * labels
            cams = np.argmax(a=cams, axis=1).astype(dtype=np.uint8)

            cams[masks == 4] = 4
            self.cam_evaluator.add_batch(gt_mask=masks, pred_mask=cams)

        scores = self.cam_evaluator.get_scores()

        self.test_cam_writer.add_scalar(tag='Epoch/PixelAccuracy', scalar_value=scores['pa'], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/MeanClassAccuracy', scalar_value=scores['ma'], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/TE_IoUAccuracy', scalar_value=scores['iou'][0], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/NEC_IoUAccuracy', scalar_value=scores['iou'][1], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/LYM_IoUAccuracy', scalar_value=scores['iou'][2], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/TAS_IoUAccuracy', scalar_value=scores['iou'][3], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/MIoU', scalar_value=scores['miou'], global_step=ep)
        self.test_cam_writer.add_scalar(tag='Epoch/FWIoU', scalar_value=scores['fwiou'], global_step=ep)

        for k, v in scores.items():
            print(f'Test CAMs {k}: {np.nanmean(scores[k]):0.4f}')
        print(f'Test CAMs model_mu: {self.model.mu}')
        print(f'Test CAMs model_enable_PDA: {self.model.enable_PDA}')

        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default=os.getcwd(), type=str)
    parser.add_argument("--session_name", default="Stage_1", type=str)
    parser.add_argument('--val_size', default=2000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--init_mu", default=1, type=float)
    parser.add_argument("--sigma", default=0.985, type=float)
    parser.add_argument("--l", default=0.65, type=float)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--start_PDA", default=5, type=int)
    parser.add_argument("--init_weights", default='ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--resume_path", default=None, type=str)

    args = parser.parse_args()
    args.dataset_path = os.path.join(args.root_path, 'datasets', 'LUAD-HistoSeg')
    args.checkpoints_path = os.path.join(args.root_path, 'checkpoints', args.session_name)
    args.init_weights_path = os.path.join(args.root_path, 'init_weights')

    args.train_cls_log_path = os.path.join(args.checkpoints_path, 'train_cls')
    args.val_cls_log_path = os.path.join(args.checkpoints_path, 'val_cls')
    args.val_cam_log_path = os.path.join(args.checkpoints_path, 'val_cam')
    args.test_cam_log_path = os.path.join(args.checkpoints_path, 'test_cam')

    args.train_data_path = os.path.join(args.dataset_path, 'train')
    args.val_data_path = os.path.join(args.dataset_path, 'val')
    args.test_data_path = os.path.join(args.dataset_path, 'test')

    args.luad_configs = dataset_configs.get_LUAD_HistoSeg_configs(root_path=os.path.join(args.dataset_path))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')
    print(args)

    pyutils.set_seed(seed=args.seed)
    datautils.check_LUAD_HistoSeg_dataset(dataset_configs=args.luad_configs)

    engine = Engine(args=args)

    if (args.resume_path is not None) and (os.path.exists(path=args.resume_path) or os.path.exists(path=os.path.join(args.root_path, args.resume_path))):
        print('State Dict Found')
        engine.resume_checkpoint(state_dict_path=args.resume_path)
    else:
        print('Training from scratch')
        engine.fit()

