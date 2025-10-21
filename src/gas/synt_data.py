import torch
import pickle
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union, List, Tuple

SyntDataType = Tuple[
    torch.Tensor, torch.Tensor, 
    Optional[torch.Tensor],
    Optional[Union[torch.Tensor, List[str]]]
]

class SyntDataset(Dataset):
    """Dataset class. 
    Expects dataset in format as done in generate.py file.
    """
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as fp:
            self.data = pickle.load(fp)

        self.noise_key = 'noise'
        self.images_key = 'images'
        self.latent_key = 'latents'
        self.condition_key = 'condition'

    def __len__(self):
        return len(self.data[self.images_key])

    def __getitem__(self, idx):
        return (
            self.data[self.noise_key][idx], self.data[self.images_key][idx],
            self.data[self.latent_key][idx], self.data[self.condition_key][idx])


class SyntDataLoaders:
    """Synthetic dataset loaders class.
    
    Class contatining all required dataloaders for GS/GAS training: 
    train and test loaders, batch for visulization.  
    Does not shuffle the dataset for reproducibility. 
    
    Attributes:
        train_loader (DataLoader): Dataloader with train data subset.
            Contains first `config.train_size` items from the whole dataset (teacher pickle file).
        test_loader (DataLoader): Dataloader with test data subset.
            Contains first `config.validation_size` items from the whole dataset (teacher pickle file).
        vis_batch (tuple): The first batch of the train subset for logging visualization purposes.
    """

    def __init__(self, config: ConfigDict):
        self.config = config

        dataset = SyntDataset(dataset_path=self.config.teacher_pkl)

        assert len(dataset) >= self.config.train_size + self.config.validation_size, f"""
            You'll have train data in validation split:
            your train_size={self.config.train_size}, val_size={self.config.validation_size},
            while the dataset size is {len(dataset)}
        """

        train_dataset = torch.utils.data.Subset(
            dataset, range(self.config.train_size))

        test_dataset = torch.utils.data.Subset(
            dataset, range(len(dataset) - self.config.validation_size, len(dataset)))

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.validation_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn
        )

        self.vis_batch = next(
            iter(
                DataLoader(
                    train_dataset,
                    batch_size=self.config.size_vis,
                    shuffle=False,
                    collate_fn=self.collate_fn
                )
            )
        )

        print(f"""
            -------------- Dataloader info --------------
            \tUse latents = {self.config.use_latents}
            \tUse condition = {self.config.use_condition}
            \tlen(train_loader) = {len(self.train_loader)}
            \tlen(test_loader) = {len(self.test_loader)}
        """)

    def collate_fn(self, batch: Tuple[SyntDataType]) -> SyntDataType:
        """Collates synthetic dataset from teacher pickle into batch.
        
        First two arguments are treated like torch.Tensor noise and images samples.
        Second two arguments are optional and can be used for latent diffusion models.
        They are treated as latents tensors and conditions.

        Args:
            batch (tuple[SyntDataType]): Sequense of tuples is SyntDataType format.

        Returns:
            SyntDataType: Collated batch.
        """
        noise, images, latents, condition = zip(*batch)

        noise = torch.stack(noise)
        images = torch.stack(images)
        latents = torch.stack(latents) if self.config.use_latents else None

        if self.config.use_condition:
            condition = torch.stack(condition) if isinstance(condition[0], torch.Tensor) else list(condition)
        else:
            condition = None

        return noise, images, latents, condition
