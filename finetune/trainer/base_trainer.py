import os
import torch
import torch.utils
from utils.logger import get_logger
from data import SeriesClassificationDataset
from utils.metrics import Classification, MultiTask_BinaryClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
import numpy as np
from utils.focal_loss import FocalLossBCE, FocalLossCE
import pandas as pd
class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.data.batch_size
        self.num_workers = args.data.num_workers
        self.num_classes = args.model.num_classes
        self.total_epoch = args.optim.epochs
        self.start_epoch = 0
        self.max_step_per_epoch = args.optim.step_per_epoch
        self.use_amp = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_context = autocast(device_type=self.device.type) if self.use_amp else nullcontext()
        self.results_dict = {
            'train': {},
            'val': {},
            'test': {}
        }
        self.scaler = GradScaler()
        
        self.initlize()
        
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
        
        
        
        
    
        
        
    def get_logger(self):
        output_path = self.args.output_dir
        self.logger = get_logger(os.path.join(output_path, 'log.txt'), display=self.args.log_display)
        tensorboard_log_dir = os.path.join(output_path, 'tensorboard')
        if os.path.exists(tensorboard_log_dir) is False:
            os.makedirs(tensorboard_log_dir)
        self.log_tensorboard = SummaryWriter(tensorboard_log_dir)
        
        self.pred_save_dir = os.path.join(self.args.output_dir, 'pred_npz')
        os.makedirs(self.pred_save_dir, exist_ok=True)
        
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
            transforms.Resize(self.args.crops.global_crops_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        assert self.args.data.train_dataset is not None
        self.train_dataset = SeriesClassificationDataset(self.args.data.train_dataset, transform=self.train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        assert self.args.data.val_dataset is not None
        self.val_dataset = SeriesClassificationDataset(self.args.data.val_dataset, transform=self.test_transform)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        if self.args.data.test_dataset is not None:
            self.test_dataset = SeriesClassificationDataset(self.args.data.test_dataset, transform=self.test_transform)
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
        self.model = None
    
    def load_checkpoint(self):
        pass
        

    def setup_model(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            
            if num_gpus > 1:
                self.logger.info(f"Using {num_gpus} GPUs with DataParallel")
                self.model = torch.nn.DataParallel(self.model)
                self.dataparallel = True
                self.logger.info(f"Using {num_gpus} GPUs with DataParallel")
            else:
                print("Using a single GPU")
                self.dataparallel = False
            self.model = self.model.to(self.device)
        else:
            self.logger.info("No GPU available, using CPU")
    
    def get_optimizer(self):
        learning_rate = self.args.optim.lr
        weight_decay = self.args.optim.weight_decay
        if self.args.optim.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(self.args.optim.adamw_beta1, self.args.optim.adamw_beta2))
        elif self.args.optim.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=self.args.optim.sgd_momentum)
        else:
            raise ValueError(f"Invalid optimizer: {self.args.optim.optimizer}")
        
    
    def get_scheduler(self):
        warmup_epochs = self.args.optim.warmup_epochs
        min_lr = self.args.optim.min_lr
        if self.args.optim.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_epoch, eta_min=self.args.optim.min_lr)
        elif self.args.optim.lr_scheduler == 'warmup_cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=warmup_epochs, T_mult=1, eta_min=min_lr)
        elif self.args.optim.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.optim.step_size, gamma=self.args.optim.step_gamma)
        else:
            raise ValueError(f"Invalid lr_scheduler: {self.args.optim.lr_scheduler}")
    
    def get_criterion(self):
        if self.args.optim.loss_type == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.args.optim.loss_type == 'CE_Focal':
            self.criterion = FocalLossCE(alpha=self.args.optim.focal_loss_alpha, gamma=self.args.optim.focal_loss_gamma)
        elif self.args.optim.loss_type == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.args.optim.loss_type == 'BCE_Focal':
            self.criterion = FocalLossBCE(alpha=self.args.optim.focal_loss_alpha, gamma=self.args.optim.focal_loss_gamma)
        else:
            raise ValueError(f"Invalid loss type: {self.args.optim.loss_type}")
        
    def get_metric(self):
        if self.args.optim.loss_type in ['CE', 'CE_Focal'] and self.args.model.num_classes > 2:
            self.metric = Classification()
        elif self.args.optim.loss_type in ['CE', 'CE_Focal'] and self.args.model.num_classes == 2:
            self.metric = MultiTask_BinaryClassification(num_tasks=1)
        elif self.args.optim.loss_type in ['BCE', 'BCE_Focal', 'BCE_weight'] :
            self.metric = MultiTask_BinaryClassification(num_tasks=self.args.model.num_classes)
        else:
            raise ValueError(f"Invalid loss type: {self.args.optim.loss_type}")
    
    def train(self, n_epoch):
        self.model.train()
        self.logger.info(f'Start training at epoch {n_epoch}/{self.total_epoch}')
        with tqdm(total=len(self.train_dataloader), desc=f"Train Epoch [{n_epoch}/{self.total_epoch}]", unit="batch") as pbar:
            epoch_loss = 0.0
            for i, data_dict in enumerate(self.train_dataloader):
                volumes, labels = data_dict['image'], data_dict['label']
                if 'image_name' in data_dict:
                    image_names = data_dict['image_name']
                
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()

                with self.amp_context:
                    outputs = self.model(volumes)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
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
                if isinstance(value, (float, int, np.floating, np.integer)):
                    self.log_tensorboard.add_scalar(f'train_{key}', value, n_epoch)
            
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
                volumes, labels = data_dict['image'], data_dict['label']
                if 'image_name' in data_dict:
                    image_names = data_dict['image_name']
                
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                with self.amp_context:
                    outputs = self.model(volumes)
                    if self.args.optim.loss_type in ['BCE', 'BCE_Focal']:
                        labels = labels.float()
                        if outputs.shape != labels.shape and self.num_classes == 1:
                            labels = labels.unsqueeze(1)
                    loss = self.criterion(outputs, labels)
                    
                if 'image_name' in data_dict:
                    for j in range(len(image_names)):
                        vol_name = image_names[j]
                        val_results_dict[vol_name] = {'label': labels[j].cpu().numpy(), 'pred': outputs[j].cpu().numpy()}
                        
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
                if isinstance(value, (float, int, np.floating, np.integer)):
                    self.log_tensorboard.add_scalar(f'{split}_{key}', value, n_epoch)
                
            self.logger.info(f'Finish evaluate {split} epoch {n_epoch}')
            return avg_loss, metrics_result
    def save_checkpoint(self, n_epoch):
        output_dir = os.path.join(self.args.output_dir, 'checkpoints')
        if os.path.exists(output_dir) is False:
            os.makedirs(output_dir)
        
        model_dict = self.model.state_dict()
        
        save_model_dict = {}
        for key, value in model_dict.items():
            if 'backbone' not in key:
                save_model_dict[key] = value
        save_dict = {
            'model': save_model_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': n_epoch,
            'results': self.results_dict
        }
        
        torch.save(save_dict, os.path.join(output_dir, f'epoch_{n_epoch}.pth'))
        self.logger.info(f"Save checkpoint at epoch {n_epoch}")
    
    def format_results(self, results_dict):
        formatted_str = ""
        for key, value in results_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (float, np.float64)):
                        formatted_str += f"{key}_{sub_key}: {sub_value:.4f}\n"
                    else:
                        formatted_str += f"{key}_{sub_key}: {sub_value}\n"
            else:
                if isinstance(value, (float, np.float64)):
                    formatted_str += f"{key}: {value:.4f}\n"
                else:
                    formatted_str += f"{key}: {value}\n"
        return formatted_str
    
    def run(self):
        for n_epoch in range(self.start_epoch, self.total_epoch):
            _, self.results_dict['train'][n_epoch] = self.train(n_epoch)
            _, self.results_dict['val'][n_epoch] =  self.val(n_epoch, split='val')
            if self.args.data.val_dataset == self.args.data.test_dataset or self.args.data.test_dataset == '':
                self.results_dict['test'][n_epoch] = self.results_dict['val'][n_epoch]
            else:
                _, self.results_dict['test'][n_epoch] =  self.val(n_epoch, split='test')
            self.scheduler.step()
            
            train_str = self.format_results(self.results_dict['train'][n_epoch])
            val_str = self.format_results(self.results_dict['val'][n_epoch])
            test_str = self.format_results(self.results_dict['test'][n_epoch])

            self.logger.info(f"Epoch {n_epoch}\nTrain:\n{train_str}\nVal:\n{val_str}\nTest:\n{test_str}")
            self.save_checkpoint(n_epoch)
