import sys
sys.path.append(sys.path[0].replace('utils', ''))
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
import torch

class Classification(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
    
    def update(self, pred, label, easy_model=False):
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        pred = pred.cpu()
        label = label.cpu()
        
        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)
            
    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        self.init()
        return result_dict


class Binary_Classification(object):
    def __init__(self):
        self.init()
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.output_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
    
    def update(self, pred, label, loss_func=torch.nn.BCEWithLogitsLoss()):
        self.output_list.append(pred.detach().cpu().numpy())
        
        pred = pred.cpu()
        label = label.cpu()
        
        loss = loss_func(pred, label).item() * len(label)
        self.loss += loss
        
        prob_pred = torch.sigmoid(pred)
        self.pred_list.extend(prob_pred.numpy())
        self.label_list.extend(label.numpy())
        
        binary_pred = (prob_pred > 0.5).float()
        self.correct_count += binary_pred.eq(label).sum().item()
        self.total_count += len(label)
    
    def results(self):
        all_preds = np.concatenate(self.pred_list)
        all_labels = np.concatenate(self.label_list)
        
        result_dict = {}
        binary_pred = (all_preds > 0.5).astype(int) 
        result_dict['acc'] = accuracy_score(all_labels, binary_pred)
        
        if len(np.unique(all_labels)) > 1:
            result_dict['auc'] = roc_auc_score(all_labels, all_preds)
        else:
            result_dict['auc'] = 'N/A'
                
        result_dict['f1'] = f1_score(all_labels, binary_pred, average='macro')
        
        result_dict['mAP'] = average_precision_score(all_labels, all_preds)
        
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        
        self.init()
        
        return result_dict
    
class MultiTask_BinaryClassification(object):
    def __init__(self, num_tasks=1):
        self.num_tasks = num_tasks
        self.init()
    
    def init(self):
        self.pred_list = [[] for _ in range(self.num_tasks)]
        self.label_list = [[] for _ in range(self.num_tasks)]
        self.output_list = [[] for _ in range(self.num_tasks)]
        self.correct_count = [0] * self.num_tasks
        self.total_count = [0] * self.num_tasks
        self.loss = [0] * self.num_tasks
    
    def update(self, pred, label, loss_func=torch.nn.BCEWithLogitsLoss()):
        if self.num_tasks == 1:
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
                
            pred = pred.unsqueeze(1)
            label = label.unsqueeze(1)
        
        
        for i in range(self.num_tasks):
            self.output_list[i].append(pred[:, i].detach().cpu().numpy())
        
        pred = pred.cpu()
        label = label.cpu()
        
        for i in range(self.num_tasks):
            task_pred = pred[:, i]
            task_label = label[:, i]
            
            loss = loss_func(task_pred.float(), task_label.float()).item() * len(task_label)
            self.loss[i] += loss
            
            prob_pred = torch.sigmoid(task_pred)
            self.pred_list[i].extend(prob_pred.numpy())
            self.label_list[i].extend(task_label.numpy())
            
            binary_pred = (prob_pred > 0.5).float()
            self.correct_count[i] += binary_pred.eq(task_label).sum().item()
            self.total_count[i] += len(task_label)
                
    def results(self):
        result_dict = {}
        for i in range(self.num_tasks):
            task_preds = np.concatenate([np.array([x]) for x in self.pred_list[i]])
            task_labels = np.concatenate([np.array([x]) for x in self.label_list[i]])
            
            binary_preds = (task_preds > 0.5).astype(int)
            result_dict[f'task_{i}_acc'] = accuracy_score(task_labels, binary_preds)
            
            if len(np.unique(task_labels)) > 1:
                result_dict[f'task_{i}_auc'] = roc_auc_score(task_labels, task_preds)
            else:
                result_dict[f'task_{i}_auc'] = 'N/A'
            
            result_dict[f'task_{i}_f1'] = f1_score(task_labels, binary_preds)
            result_dict[f'task_{i}_mAP'] = average_precision_score(task_labels, task_preds)
            result_dict[f'task_{i}_loss'] = float(self.loss[i]) / float(self.total_count[i])
        
        self.init()
        
        return result_dict
