import os
import pandas as pd
import numpy as np
from PIL import Image

class BasicClassificationDataset(object):
    def __init__(self, data_path=None, transform=None, logger=None):
        assert os.path.exists(data_path), '{} not exist 文件不存在'.format(data_path)
        self.data_path = data_path
        self.transform = transform
        self.flie_type = data_path.split('.')[-1]
        self.logger = logger
        
        self.images_list, self.labels_list = self._read_data()
        
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

                    images_list.append(split_list[0])

                    if len(split_list) > 2:
                        labels = [int(label) for label in split_list[1:]]
                        labels = np.array(labels)
                        labels_list.append(labels)
                    else:
                        labels_list.append(int(split_list[1]))
        elif self.flie_type == 'csv':
           file_df = pd.read_csv(self.data_path)
           images_list = file_df.iloc[:, 0].values
           if len(file_df.columns) > 2:
               labels_list = file_df.iloc[:, 1:].values
           else:
               labels_list = file_df.iloc[:, 1].values
               
        return images_list, labels_list
    
    def __getitem__(self, index):
        image_path = self.images_list[index]
        label = self.labels_list[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        data_dict = {'image': image, 'label': label}
        return data_dict
    
    def __len__(self):
        return len(self.images_list)
class SeriesClassificationDataset(BasicClassificationDataset):
    def __init__(self, root_dir, transform=None):
        super(SeriesClassificationDataset, self).__init__(root_dir, transform)
        self.image_name_dict = {
            'image_name': [],
            'study_series_name': []
        }
        self.series_name_list = []
        self.get_image_list()
    
    def get_image_list(self):
        series_name_list = set()         
        for image_path, label in zip(self.images_list, self.labels_list):
            image_name = os.path.basename(image_path)
            self.image_name_dict['image_name'].append(image_name)
            
            split_image_name = image_name.split('_')
            study_id = split_image_name[0]
            series_name = split_image_name[1]
            slice_id = split_image_name[2]
            total_slice = split_image_name[3].split('.')[0]

            study_series_name = study_id + '_' + series_name
            self.image_name_dict['study_series_name'].append(study_series_name)
            series_name_list.add(study_series_name)

        self.series_name_list = list(series_name_list)
            
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        data_dict.update({'image_name': self.image_name_dict['image_name'][index], 'study_series_name': self.image_name_dict['study_series_name'][index]})
        return data_dict
