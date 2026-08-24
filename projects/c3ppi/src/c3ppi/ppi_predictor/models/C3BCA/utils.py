import os
import pandas as pd
import torch

from torch.utils.data import DataLoader,Dataset

class PPIDataset(Dataset):
    """
    Dataset class for Protein-Protein Interaction (PPI) data.

    Attributes:
        dataDF (pd.DataFrame): DataFrame containing PPI pairs and their labels.
        processed_data_dir (str): Directory containing processed protein graph data.
    """
    def __init__(self, dataDF: pd.DataFrame, processed_data_dir: str):
        """
        Initialize the dataset.

        Args:
            dataDF (pd.DataFrame): DataFrame with columns "P1", "P2", and "Label".
            processed_data_dir (str): Directory containing preprocessed graph and feature data.
        """
        self.dataDF = dataDF
        self.processed_data_dir = processed_data_dir

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataDF)

    def __getitem__(self, idx):
        """
        Get the data for a single PPI sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing two protein embeddings (protein 1 and protein 2) and the interaction label.
        """
        # Get sample: 1 PPI (1: inter/0: not inter)
        sample = self.dataDF.iloc[idx, :]
        p1 = sample["P1"]
        p2 = sample["P2"]
        label = sample["Label"]

        try:
            # Load embeddings
            prot_g1 = torch.load(os.path.join(self.processed_data_dir, f"{p1}_embedding.pt"))
            prot_g2 = torch.load(os.path.join(self.processed_data_dir, f"{p2}_embedding.pt"))

            return prot_g1, prot_g2, torch.tensor(label, dtype=torch.float32)

        except FileNotFoundError as e:
            print(e)
        
def get_ppi_data_loader(data_path, processed_data_dir, batch_size=64, shuffle=True, num_workers=0, pin_memory=True):
    """
    Create a data loader for the PPI dataset.

    This function loads the dataset from a CSV file, initializes a `PPIDataset` instance, and returns a PyTorch DataLoader
    for batching, shuffling, and parallel data loading.

    Args:
        data_path (str): Path to the CSV file containing PPI data with columns "P1", "P2", and "Label".
        processed_data_dir (str): Directory containing preprocessed graph and feature data.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        pin_memory (bool, optional): Whether to use pinned memory for faster data transfer to GPU. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the PPI dataset.
    """
    # Load the dataset as a DataFrame
    dataDF = pd.read_csv(data_path)

    # Initialize the dataset
    dataset = PPIDataset(dataDF, processed_data_dir)
    # Create the DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return data_loader