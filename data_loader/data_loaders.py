from torch.utils.data import DataLoader
from .dataset import dataset_evstack, dataset_timetensor
from base.base_data_loader import BaseDataLoader


class EvstackTrainDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, training=True, transform=None, num_workers=8):
        
        self.dataset = dataset_evstack.TrainDataset(data_dir, transform=transform, training=training)

        super(EvstackTrainDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class EvstackInferDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, transform=None, num_workers=2):
        transform = None
        self.dataset = dataset_evstack.InferDataset(data_dir, transform=transform)

        super(EvstackInferDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)