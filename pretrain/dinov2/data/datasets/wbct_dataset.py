'''
Whole Body CT Dataset
'''
import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from io import BytesIO
from typing import Any

from PIL import Image
import h5py

class Decoder:
    def decode(self) -> Any:
        raise NotImplementedError

class ImageDataDecoder(Decoder):
    def __init__(self, image_data: bytes) -> None:
        self._image_data = image_data

    def decode(self) -> Image:
        f = BytesIO(self._image_data)
        img = Image.open(f).convert("RGB")
        return img
        
        

class TargetDecoder(Decoder):
    def __init__(self, target: Any):
        self._target = target

    def decode(self) -> Any:
        return self._target


from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class WBCTDataset(ExtendedVisionDataset):
    '''
    Whole Body CT Dataset 2D slice的dataset
    '''
    def __init__(
        self,
        *,
        split: str,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        
        self._extra_root = extra
        self._split = split
        if os.path.isfile(os.path.join(self._extra_root, f"{self._split}.npz")):
            total_data = np.load(os.path.join(self._extra_root, f"{self._split}.npz"), allow_pickle=True)
            self._entries = total_data['entries']
            self._split_study_sequence_ids = total_data['study_sequence_ids']
            self._study_sequence_names = total_data['study_sequence_names']
        else:
            self._entries = self._load_extra(f"{self._split}_entries.npy")
            self._study_sequence_ids = self._load_extra(f"{self._split}_study_sequence_ids.npy")
            self._study_sequence_names = self._load_extra(f"{self._split}_study_sequence_names.npy")
    
    def _load_extra(self, extra_name):
        return np.load(os.path.join(self._extra_root, extra_name))
    
    def get_image_data(self, index: int) -> bytes:
        file_path = self._entries[index][2]
        with open(file_path, "rb") as f:
            return f.read()
    
    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(f"{self._split}_entries.npy")
        assert self._entries is not None
        return self._entries

    def _get_study_sequence_ids(self) -> np.ndarray:
        if self._study_sequence_ids is None:
            self._study_sequence_ids = self._load_extra(f"{self._split}_study_sequence_ids.npy")
        assert self._study_sequence_ids is not None
        return self._study_sequence_ids

    def _get_study_sequence_names(self) -> np.ndarray:
        if self._study_sequence_names is None:
            self._study_sequence_names = self._load_extra(f"{self._split}_study_sequence_names.npy")
        assert self._study_sequence_names is not None
        return self._study_sequence_names
    
    def get_target(self, index: int):
        entries = self._get_entries()
        study_sequence_id = entries[index][0]
        return int(study_sequence_id)

    def get_targets(self) -> Optional[np.ndarray]:
        study_sequence_ids = self._get_study_sequence_ids
        return study_sequence_ids

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        study_sequence_name = entries[index][1]
        return str(study_sequence_name)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image_data = self.get_image_data(index)
            image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self._entries)
    
