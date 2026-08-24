import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pathlib import Path
from c3ppi.ppi_predictor.models.C3BCA import PPI_Model

class ProteinPairsDataset(Dataset):
    def __init__(self, p1, pool_B, embedding_cache, processed_data_dir):
        """
        Initialize the dataset.

        Args:
            p1 (str): The id of a protein.
            pool_B (list): A list of ids of proteins in the pool to be paired with p1.
            embedding_cache (dict): Dictionary of embeddings for pool_B proteins.
            processed_data_dir (str): Directory containing preprocessed protein data files.
        """
        self.p1 = p1
        self.p1_embedding = torch.load(os.path.join(processed_data_dir, f"{p1}_embedding.pt"))
        self.pool_B = pool_B
        self.embedding_cache = embedding_cache

    def __len__(self):
        return len(self.pool_B)

    def __getitem__(self, idx):
        sample = self.pool_B[idx]
        prot_g2 = self.embedding_cache[sample]  # <-- preloaded embedding
        return self.p1_embedding, prot_g2, sample


class Inference():
    def __init__(self, embeddings_dir, output_dir, pairs, threshold=0.5, best_model_path="best_model.pt"):
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.pairs = pairs
        self.best_model_path = best_model_path
        self.batch_size = 512
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.per_prot_dir = os.path.join(self.output_dir, "per_protein_results")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.per_prot_dir, exist_ok=True)

    def get_ppi_data_loader(self, p1, pool_B, bac_embedding_cache, shuffle=False, num_workers=0, pin_memory=False):  
        """
        Creates a DataLoader for a protein pair dataset.

        Args:
            p1 (str): The id of a protein.
            pool_B (list): A list of ids of proteins in the pool to be paired with p1.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to False.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 8.
            pin_memory (bool, optional): Whether to use pinned memory for faster data transfer to GPU. Defaults to True.
p
        Returns:
            torch.utils.data.DataLoader: DataLoader instance for the protein pair dataset.
        """
        dataset = ProteinPairsDataset(p1, pool_B, bac_embedding_cache, self.embeddings_dir)
        # Create the DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory,
        )
        return data_loader

    def run(self):
        pool_A, pool_B = self.pairs["Pool A"], self.pairs["Pool B"]

        pool_A = [p for p in pool_A if os.path.isfile(os.path.join(self.embeddings_dir, f"{p}_embedding.pt"))]
        pool_B = [p for p in pool_B if os.path.isfile(os.path.join(self.embeddings_dir, f"{p}_embedding.pt"))]

        print(f"Pool A: {len(pool_A)}")
        print(f"Pool B: {len(pool_B)}")

        bac_embedding_cache = {
            prot_id: torch.load(os.path.join(self.embeddings_dir, f"{prot_id}_embedding.pt"))
            for prot_id in pool_B
        }

        # Load model
        model = PPI_Model()
        model.load_state_dict(torch.load(self.best_model_path))
        model.to(self.device)
        model.eval()

        for idx, p1 in enumerate(pool_A):

            if os.path.isfile(os.path.join(self.per_prot_dir,"{}_predictions.parquet".format(p1))):
                continue

            print(f"Running inference for protein {p1}: {idx+1}/{len(pool_A)}")

            data_loader = self.get_ppi_data_loader(p1, pool_B, bac_embedding_cache)

            predictions = []

            with torch.no_grad():
                for p1_emb, prot_g2, b_ids in tqdm(data_loader):
                    p1_emb, prot_g2 = p1_emb.to(self.device).float(), prot_g2.to(self.device).float()
                    outputs = torch.sigmoid(model(p1_emb, prot_g2))

                    for score, b_id in zip(outputs, b_ids):
                        if score >= self.threshold:
                            predictions.append({
                                "bacterial_protein": b_id,
                                "score": round(score.item(), 4)
                            })

            # To parquet
            df = pd.DataFrame(predictions)
            df.to_parquet(os.path.join(self.per_prot_dir, "{}_predictions.parquet".format(p1)), index=False)

            print(df)

import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # Define paths to the protein data files
    bac_proteins = "./data/bac_proteins.csv"

    ###
    human_we_want = "/home/c3biolab/c3biolab_projects/doctorals/DPK/c3ppi/notebooks/GUT_MB_Clinical/data/idmapping_chem_exp_2.tsv"
    hww_df = pd.read_csv(human_we_want, sep="\t", header=0)
    #print(hww_df)
    ###

    human_proteins = "./data/human_gut_brain_proteins.csv"

    hum_prot_df = pd.read_csv(human_proteins)
    bac_prot_df = pd.read_csv(bac_proteins)

    ###
    #  Keep human_proteins that are in the chemical experiment data
    hum_prot_df = hum_prot_df[hum_prot_df["From"].isin(hww_df["Entry"].tolist())]
    print(hum_prot_df)
    ###
    protein_pairs = {
        "Pool A": hum_prot_df["Entry"].tolist(),
        "Pool B": bac_prot_df["Entry"].tolist()
    }

    inf = Inference(    
        embeddings_dir="./data/embeddings",
        output_dir="./data/inference_results_fffffffffff",
        pairs=protein_pairs,
        threshold=0.4679,
        best_model_path="/home/c3biolab/c3biolab_projects/doctorals/DPK/c3ppi/registry/c3ppi_model/best_model.pt"
    )
    inf.run()