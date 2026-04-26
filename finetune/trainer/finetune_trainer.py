from torchvision.transforms.transforms import F
from .base_trainer import BaseTrainer
from models import build_base25d_avg_model_from_cfg, build_trans25d_classification_model_from_cfg
import torch
import traceback
from data import VolumeDataset, EvalVolumeDataset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
from utils.focal_loss import FocalLossBCE, FocalLossCE

class Finetune25D_Avg_Trainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def initlize(self):
        self.get_logger()
        self.get_data()
        self.get_criterion()
        self.get_metric()
        self.get_model()
        self.setup_model()
        self.get_optimizer()
        self.get_scheduler()
        self.load_checkpoint()
        
    def get_criterion(self):
        if self.args.optim.loss_type == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.args.optim.loss_type == 'CE_Focal':
            self.criterion = FocalLossCE(alpha=self.args.optim.focal_loss_alpha, gamma=self.args.optim.focal_loss_gamma)
        elif self.args.optim.loss_type == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.args.optim.loss_type == 'BCE_weight':
            labels = self.train_dataset.labels_list
            labels = np.array(labels)
            if self.num_classes == 1:
                pos = np.sum(labels == 1)
                neg = np.sum(labels == 0)
                weight = neg / pos if pos > 0 else 1.0
                pos_weights = [weight]
            else:
                pos_weights = []
                for i in range(self.num_classes):
                    pos = np.sum(labels[:, i] == 1)
                    neg = np.sum(labels[:, i] == 0)
                    weight = neg / pos if pos > 0 else 1.0
                    pos_weights.append(weight)
            pos_weights = torch.FloatTensor(pos_weights).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        elif self.args.optim.loss_type == 'BCE_Focal':
            self.criterion = FocalLossBCE(alpha=self.args.optim.focal_loss_alpha, gamma=self.args.optim.focal_loss_gamma)
        else:
            raise ValueError(f"Invalid loss type: {self.args.optim.loss_type}")
        
    def get_data(self): 
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.args.crops.global_crops_size, scale=(self.args.crops.global_crops_scale[0], self.args.crops.global_crops_scale[1])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=[90, 270]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((self.args.crops.global_crops_size, self.args.crops.global_crops_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        assert self.args.data.train_dataset is not None
        self.train_dataset = VolumeDataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = VolumeDataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = VolumeDataset(self.args.data.test_dataset, transform=self.test_transform)
        else:
            self.test_dataset = self.val_dataset
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        
        self.datasets_dict = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        
        self.dataloaders_dict = {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
        
        self.logger.info(f"Train dataset: {self.args.data.train_dataset} / Length: {len(self.train_dataset)} / loader: {len(self.train_dataloader)}")
        self.logger.info(f"Val dataset: {self.args.data.val_dataset} / Length: {len(self.val_dataset)} / loader: {len(self.val_dataloader)}")
        self.logger.info(f"Test dataset: {self.args.data.test_dataset} / Length: {len(self.test_dataset)} / loader: {len(self.test_dataloader)}")
    
    def get_model(self):
        self.model, self.embed_dim = build_base25d_avg_model_from_cfg(cfg=self.args, only_teacher=True)
        
    def get_optimizer(self):
        params_group = self.model.get_optimizer_param_groups(self.args.optim.lr)
        if self.args.optim.optimizer.lower() in ['adam', 'adamw']:
            self.optimizer = torch.optim.AdamW(params_group, weight_decay=self.args.optim.weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params_group, weight_decay=self.args.optim.weight_decay, momentum=self.args.optim.sgd_momentum)
        else:
            raise ValueError(f"Optimizer {self.args.optim.optimizer} not supported.")    
    
    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.args.model.pretrained_weights}")
        if self.args.model.pretrained_weights == "":
            self.logger.info("No checkpoint loaded.")
            return
        try:
            model_state_dict = self.model.backbone.state_dict()
            checkpoint = torch.load(self.args.model.pretrained_weights, map_location='cpu', weights_only=True)
            if 'pos_embed' in checkpoint.keys():
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():
                for key, values in checkpoint['teacher'].items():
                    if 'backbone' in key:
                        model_state_dict[key.replace('backbone.', '')] = values
                self.model.backbone.load_state_dict(model_state_dict, strict=True)
            else:
                raise ValueError("Invalid checkpoint")
            self.logger.info(f"Loaded pretrained model from {self.args.model.pretrained_weights}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model from {self.args.model.pretrained_weights}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())
        
        if self.args.resume:
            ckpt_files = [os.path.join(self.args.output_dir, 'checkpoints', file_name) for file_name in os.listdir(os.path.join(self.args.output_dir, 'checkpoints')) if file_name.endswith('.pth')]
            last_ckpt_path = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            self.logger.info(f"Resuming training from {last_ckpt_path}")
            checkpoint = torch.load(last_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"Resumed training from {last_ckpt_path} and start from epoch {self.start_epoch}")
        
        
    def train(self, n_epoch):
        self.model.train()
        self.logger.info(f'Start training at epoch {n_epoch}/{self.total_epoch}')
        with tqdm(total=len(self.train_dataloader), desc=f"Train Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(self.train_dataloader):
                images = data_dict['image']
                labels = data_dict['label']
                orig_z = data_dict['orig_z']
                if self.args.data.random_z_flip:
                    for b_idx in range(images.shape[0]):
                        if np.random.rand() < self.args.data.random_z_flip_ratio:
                            valid_slices = orig_z[b_idx].item()
                            images[b_idx, :valid_slices] = images[b_idx, :valid_slices].flip(0)
                        
                images = images.to(self.device)
                labels = labels.to(self.device)
                orig_z = orig_z.to(self.device)
                
                self.optimizer.zero_grad()

                with self.amp_context:
                    outputs = self.model(images, orig_z)
                    if 'BCE' in self.args.optim.loss_type:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    loss = self.criterion(outputs, labels)

                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
                self.log_tensorboard.add_scalar('train_loss', loss.item(), n_epoch * len(self.train_dataloader) + i)

                if self.max_step_per_epoch != -1 and i >= self.max_step_per_epoch:
                    break
                
            avg_loss = epoch_loss / len(self.train_dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                if isinstance(value, str) and value.upper() == 'N/A':
                    value = 0
                elif value is None:
                    value = 0
                self.log_tensorboard.add_scalar(f'train_{key}', float(value), n_epoch)
            
            self.logger.info(f'Finish training epoch {n_epoch}')
            return avg_loss, metrics_result
        
    @torch.no_grad()
    def val(self, n_epoch, split='val'):
        val_results_dict = {}
        self.model.eval()
        dataloader = self.dataloaders_dict[split]
        self.logger.info(f'Start evaluate {split} epoch {n_epoch}')
        with tqdm(total=len(dataloader), desc=f"{split} Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(dataloader):
                images = data_dict['image'].to(self.device)
                labels = data_dict['label'].to(self.device)
                orig_z = data_dict['orig_z'].to(self.device)
                image_names = data_dict['image_name']
                
                with self.amp_context:
                    outputs = self.model(images, orig_z)
                    if 'BCE' in self.args.optim.loss_type:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    loss = self.criterion(outputs, labels)
                    
                for image_name, label, output in zip(image_names, labels, outputs):
                    val_results_dict[image_name] = {'label': label.cpu().numpy(), 'pred': output.cpu().numpy()}
                        
                self.metric.update(outputs.detach().cpu(), labels)
                epoch_loss += loss.item()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            
            save_file_name = os.path.join(self.pred_save_dir, f'{split}_epoch_{n_epoch}.npz')
            np.savez(save_file_name, val_results_dict)
            self.logger.info(f"Save prediction results to {save_file_name}")
            
            avg_loss = epoch_loss / len(dataloader)
            metrics_result = self.metric.results()
            for key, value in metrics_result.items():
                if isinstance(value, str) and value.upper() == 'N/A':
                    value = 0
                elif value is None:
                    value = 0
                self.log_tensorboard.add_scalar(f'{split}_{key}', value, n_epoch)
                
            self.logger.info(f'Finish evaluate {split} epoch {n_epoch}')
            return avg_loss, metrics_result



class FinetuneTrans25D_Trainer(Finetune25D_Avg_Trainer):
    def get_model(self):
        self.model, self.embed_dim = build_trans25d_classification_model_from_cfg(cfg=self.args, only_teacher=True)
    
    def get_optimizer(self):
        params_group = self.model.get_optimizer_param_groups(self.args.optim.lr)
        if self.args.optim.optimizer.lower() in ['adam', 'adamw']:
            self.optimizer = torch.optim.AdamW(params_group, weight_decay=self.args.optim.weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params_group, weight_decay=self.args.optim.weight_decay, momentum=self.args.optim.sgd_momentum)
        else:
            raise ValueError(f"Optimizer {self.args.optim.optimizer} not supported.")    
        
class EvalTrans25D_Trainer(FinetuneTrans25D_Trainer):
    def get_data(self): 
        self.test_transform = transforms.Compose([
            transforms.Resize((self.args.crops.global_crops_size, self.args.crops.global_crops_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        assert self.args.data.train_dataset is not None
        self.train_dataset = EvalVolumeDataset(self.args.data.train_dataset, transform=self.test_transform, logger=self.logger)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        assert self.args.data.val_dataset is not None
        self.val_dataset = EvalVolumeDataset(self.args.data.val_dataset, transform=self.test_transform, logger=self.logger)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = EvalVolumeDataset(self.args.data.test_dataset, transform=self.test_transform, logger=self.logger)
        else:
            self.test_dataset = self.val_dataset
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        
        self.datasets_dict = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
        
        self.dataloaders_dict = {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }
        
        self.logger.info(f"Train dataset: {self.args.data.train_dataset} / Length: {len(self.train_dataset)} / loader: {len(self.train_dataloader)}")
        self.logger.info(f"Val dataset: {self.args.data.val_dataset} / Length: {len(self.val_dataset)} / loader: {len(self.val_dataloader)}")
        self.logger.info(f"Test dataset: {self.args.data.test_dataset} / Length: {len(self.test_dataset)} / loader: {len(self.test_dataloader)}")
    
    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.args.model.pretrained_weights}")
        if self.args.model.pretrained_weights == "":
            self.logger.info("No checkpoint loaded.")
            return
        try:
            model_state_dict = self.model.backbone.state_dict()
            checkpoint = torch.load(self.args.model.pretrained_weights, map_location='cpu', weights_only=True)
            if 'pos_embed' in checkpoint.keys():
                if checkpoint['pos_embed'].shape != model_state_dict['pos_embed'].shape:
                    checkpoint['pos_embed'] = model_state_dict['pos_embed']
                self.model.backbone.load_state_dict(checkpoint, strict=True)
            elif 'teacher' in checkpoint.keys():
                for key, values in checkpoint['teacher'].items():
                    if 'backbone' in key:
                        model_state_dict[key.replace('backbone.', '')] = values
                self.model.backbone.load_state_dict(model_state_dict, strict=True)
            else:
                raise ValueError("Invalid checkpoint")
            self.logger.info(f"Loaded pretrained model from {self.args.model.pretrained_weights}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained model from {self.args.model.pretrained_weights}")
            self.logger.error(f"Exception: {e}")
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Loading checkpoint from {self.args.model.ckpt_path}")
        checkpoint = torch.load(self.args.model.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.logger.info(f"Loaded checkpoint from {self.args.model.ckpt_path}")
            
    def run(self):
        n_epoch = 0
        _, self.results_dict['test'][n_epoch] = self.val(n_epoch, split='test')
        test_str = self.format_results(self.results_dict['test'][n_epoch])
        self.logger.info(f"Test:\n{test_str}")
        
        test_results = self.results_dict['test'][n_epoch]
        if isinstance(test_results, dict) and len(test_results) > 0:
            all_metrics = {}
            num_tasks = 0
            
            for task_name, task_metrics in test_results.items():
                if isinstance(task_metrics, dict):
                    num_tasks += 1
                    for metric_name, metric_value in task_metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)
            
            if num_tasks > 0:
                avg_metrics_str = "Average across all tasks:\n"
                for metric_name, values in all_metrics.items():
                    numeric_values = [
                        value for value in values
                        if isinstance(value, (int, float, np.floating, np.integer))
                    ]
                    if len(numeric_values) > 0:
                        avg_value = sum(numeric_values) / len(numeric_values)
                        avg_metrics_str += f"{metric_name}: {avg_value:.4f}\n"
                
                self.logger.info(avg_metrics_str)
