from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional
from PIL import Image
from fastai.vision.all import Path, get_image_files, verify_images
import os
import json
from dinov2.data.datasets.extended import ExtendedVisionDataset

class MyDataset(ExtendedVisionDataset):
    def __init__(self, root: str, extra: str, verify: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.root = Path(root).expanduser()
        self.labels = self._load_labels('')
        print('labels', self.labels)
        self.class_to_idx = self._load_class_index('')
        print('calss_index', self.class_to_idx)
        image_paths = get_image_files(self.root)
        invalid_images = set()
        if verify:
            invalid_images = set(verify_images(image_paths))
        self.image_paths = [p for p in image_paths if p not in invalid_images]
        
    def _load_labels(self, labels_file: str):
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split()
                labels[image_name] = label
        return labels

    def _load_class_index(self, class_index_file: str):
        with open(class_index_file, 'r') as f:
            class_to_idx = json.load(f)
        return class_to_idx

    def find_target(self, data, search_key):
        for index, value in data.items():
            if value[0] == search_key:
                return index
        return None   
    
    def get_image_data(self, index: int) -> bytes:
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        return img
        
    def get_target(self, index: int) -> Any:
        image_path = self.image_paths[index]
        image_name = os.path.basename(image_path)
        synset = self.labels.get(image_name, -1)
        if synset != -1:
            target = self.find_target(self.class_to_idx, synset)
        else:
            target = -1
        return int(target)

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
