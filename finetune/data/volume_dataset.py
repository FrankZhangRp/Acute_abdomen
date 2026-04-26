import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import nibabel as nib

class VolumeDataset(object):
    def __init__(self, data_path=None, transform=None, logger=None):
        assert os.path.exists(data_path), f"{data_path} does not exist"
        self.data_path = data_path
        self.data_dir = os.path.dirname(os.path.abspath(data_path))
        self.transform = transform
        self.flie_type = data_path.split('.')[-1]
        self.logger = logger
        
        self.images_list, self.labels_list = self._read_data()
        if '.npy' in self.images_list[0]:
            self.image_type = 'npy'
        elif '.nii.gz' in self.images_list[0]:
            self.image_type = 'nii.gz'
        else:
            raise ValueError('Unsupported image type')
        
    def _read_data(self):
        if self.flie_type == 'txt':
            images_list = []
            labels_list = []
            with open(self.data_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if ' ' in line and ',' not in line:
                        split_list = line.split(' ')
                    else:
                        split_list = line.split(',')

                    images_list.append(self._resolve_image_path(split_list[0]))

                    if len(split_list) > 2:
                        labels = [int(label) for label in split_list[1:]]
                        labels = np.array(labels)
                        labels_list.append(labels)
                    else:
                        labels_list.append(int(split_list[1]))
        elif self.flie_type == 'csv':
           file_df = pd.read_csv(self.data_path)
           images_list = [self._resolve_image_path(value) for value in file_df.iloc[:, 0].values]
           if len(file_df.columns) > 2:
               labels_list = file_df.iloc[:, 1:].values
           else:
               labels_list = file_df.iloc[:, 1].values
               
        return images_list, labels_list

    def _resolve_image_path(self, image_path):
        image_path = str(image_path)
        if os.path.isabs(image_path):
            return image_path
        return os.path.abspath(os.path.join(self.data_dir, image_path))
    
    def __getitem__(self, index):
        image_path = self.images_list[index]
        label = self.labels_list[index]
        if self.image_type == 'npy':
            volume = np.load(image_path)
            volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
            volume = volume.astype(np.uint8)
        elif self.image_type == 'nii.gz':
            volume = nib.load(image_path).get_fdata()
            volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
            volume = volume.astype(np.uint8)
            volume = np.transpose(volume, (2, 1, 0))
        image_size = volume.shape
        transformed_slices = []
        
        max_slices = 300
        step = max(1, volume.shape[0] // max_slices)
        
        for z in range(0, volume.shape[0], step):
            slice_img = volume[z] 
            slice_rgb = np.stack([slice_img] * 3, axis=2)
            slice_pil = Image.fromarray(slice_rgb.astype(np.uint8))
            
            if self.transform is not None:
                slice_transformed = self.transform(slice_pil)
                transformed_slices.append(slice_transformed)
                
            if len(transformed_slices) >= max_slices:
                break
                
        while len(transformed_slices) < max_slices:
            transformed_slices.append(transformed_slices[-1])
            
        image = torch.stack(transformed_slices)
        data_dict = {'image': image, 'label': label, 'orig_z': volume.shape[0], 'image_name': image_path.split('/')[-1].replace('.npy', '')}
        return data_dict
    
    def __len__(self):
        return len(self.images_list)

class EvalVolumeDataset(VolumeDataset):
    def __init__(self, data_path=None, transform=None, logger=None, max_slices=300):
        super(EvalVolumeDataset, self).__init__(data_path=data_path, transform=transform, logger=logger)
        self.max_slices = max_slices
        
    def __getitem__(self, index):
        image_path = self.images_list[index]
        label = self.labels_list[index]
        if self.image_type == 'npy':
            volume = np.load(image_path)
            volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
            volume = volume.astype(np.uint8)
        elif self.image_type == 'nii.gz':
            volume = nib.load(image_path).get_fdata()
            volume = (volume - volume.min()) / (volume.max() - volume.min()) * 255
            volume = volume.astype(np.uint8)
            volume = volume[::-1, ::-1, ::-1]
            volume = np.transpose(volume, (2, 1, 0))
        image_size = volume.shape

        transformed_slices = []
        
        max_slices = self.max_slices
        
        if volume.shape[0] > max_slices:
            step = volume.shape[0] / max_slices
            indices = [int(i * step) for i in range(max_slices)]
            indices = [min(idx, volume.shape[0] - 1) for idx in indices]
            volume = volume[indices]
        
        for z in range(volume.shape[0]):
            slice_img = volume[z] 
            slice_rgb = np.stack([slice_img] * 3, axis=2)
            slice_pil = Image.fromarray(slice_rgb.astype(np.uint8))
            
            if self.transform is not None:
                slice_transformed = self.transform(slice_pil)
                transformed_slices.append(slice_transformed)
                
        while len(transformed_slices) < max_slices:
            transformed_slices.append(transformed_slices[-1])
            
        image = torch.stack(transformed_slices)
        data_dict = {'image': image, 'label': label, 'orig_z': volume.shape[0], 'image_name': image_path.split('/')[-1].replace('.npy', '')}
        return data_dict
    
    def __len__(self):
        return len(self.images_list)
