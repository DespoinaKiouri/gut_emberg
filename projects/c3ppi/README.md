# c3ppi — ProstT5-based Deep PPI Model for Gut Bacteria–Human PPIs

A deep-learning PPI classifier that replaces the MAPE-PPI structural VAE
embedder of `struct_ppi_pred` with the **ProstT5** protein language model
(`Rostlab/ProstT5_fp16`). Sequence-only embeddings (1024-dim) go through
per-protein projection, a **Bi-directional Cross-Attention** fusion module and
an MLP head trained with **Focal Loss**. Same architectural spine, no
structural pre-processing pipeline required.

The package is installable and structured for reuse:

```
src/c3ppi/ppi_predictor/
├── models/C3BCA/
│   ├── model.py               # PPI_Model, BiDirectionalCrossAttention, FocalLoss
│   ├── utils.py               # PPIDataset + get_ppi_data_loader
│   └── embedder/prost5.py     # ProteinEmbeddingService (mean-pooled ProstT5)
├── trainer/train.py           # Trainer (Adam + ReduceLROnPlateau + early stop)
├── evaluator/evaluate.py      # Evaluator (metrics, PR/ROC plots, threshold pick)
└── utils/config.py            # ROOT / MODEL_REGISTRY / OUTPUT paths
```

Model artefacts land in `registry/<save_name>/best_model.pt`; evaluator output
lands in `output/<save_name>_results/`. Both directories are created at
project root (auto-resolved from `config.py`).

## 1. Method overview

- **Embedding.** Each UniProt sequence is passed through ProstT5 with the
  `<AA2fold>` prefix, cleaned of `[U, Z, O, B]`, mean-pooled over residues
  (prefix token dropped) to a 1024-d vector. `ProteinEmbeddingService` caches
  the model under `registry/ProstT5/` and uses fp16 on GPU.
- **Fusion.** Two linear projections (1024 → 256) followed by
  `BiDirectionalCrossAttention` (4 heads): `p1` attends to `p2` and vice
  versa, both attention outputs are residual-added and concatenated, then
  projected back to 256.
- **Head.** `Linear(256→256) → Dropout(0.5) → Linear(256→128) →
  Dropout(0.3) → Linear(128→1)` with ReLU activations. Loss is
  `FocalLoss(α=1, γ=2)` for the class-imbalanced setting.
- **Optimisation.** Adam, `lr=1e-3`, `ReduceLROnPlateau` on val loss, early
  stopping (default `patience=3`), up to 100 epochs.
- **Threshold.** After training, the evaluator picks the threshold that
  maximises validation F1 and reuses it on the test set.

## 2. Environment

Managed via `rye`, but any Python 3.12 environment with the pyproject
dependencies works:

```bash
# rye
rye sync

# or with pip
pip install -e .
```

Dependencies (`pyproject.toml`): torch ≥ 2.7, transformers ≥ 4.52,
sentencepiece, protobuf, tqdm, scikit-learn, pandas, pyarrow, plotly, kaleido,
statsmodels, openpyxl, xlsxwriter.

GPU is assumed. The ProstT5 embedder hard-codes `cuda:0` in
`prost5.py:41`; adjust that line to `cuda if available else cpu` for
CPU-only inspection.

## 3. Data layout

`c3ppi` does not ship data. The C3PPI PPI dataset is built from human PPI
databases via the notebooks in `notebooks/PPI_Model/`. The expected layout
after step 3.1 is:

```
notebooks/PPI_Model/data/
├── train.csv                  # columns: P1, P2, Label
├── val.csv
├── test.csv
├── idmapping_1.tsv.gz         # UniProt ID-mapping export for P1, P2
├── idmapping_2.tsv.gz
├── embeddings/                # <UniProtID>_embedding.pt (1-D fp16 tensor)
├── filtered_ppi_data_train.csv
├── filtered_ppi_data_val.csv
└── filtered_ppi_data_test.csv
```

For the gut-microbiome clinical inference (`notebooks/GUT_MB_Clinical/`),
sequences of the bacterial pool are pulled through UniProt ID mapping and
embedded the same way; predictions are stored as one Parquet file per
Pool-A protein under `output/<run_name>/per_protein_results/`.

## 4. Reproduction — PPI model

The notebooks are the source of truth. Run them from
`projects/c3ppi/notebooks/PPI_Model/`:

### 4.1 Build the dataset (`01.C3PPI_Data.ipynb`)

- Loads a human PPI corpus, splits into train/val/test with columns
  `P1, P2, Label`.
- Exports unique proteins in batches of 90k for UniProt ID mapping (the two
  `idmapping_*.tsv.gz` files must be produced manually via UniProt's ID
  mapping service and dropped into `./data/`).
- Merges the ID-mapping tables, drops rows without a sequence, and writes back
  the cleaned CSVs.

### 4.2 Embed proteins (`02.EmbedProteins.ipynb`)

For every UniProt ID in the dataset:

```python
from c3ppi.ppi_predictor.models.C3BCA.embedder.prost5 import ProteinEmbeddingService

service = ProteinEmbeddingService()          # loads Rostlab/ProstT5_fp16
emb = service.embed_single_sequence(seq)     # (1024,) torch.Tensor
torch.save(emb.cpu(), f"./data/embeddings/{uid}_embedding.pt")
```

Batch mode is also available via `embed_batch_sequences`.

### 4.3 Train and evaluate (`03.ModelDev.ipynb`)

```python
from c3ppi.ppi_predictor.trainer.train import Trainer
from c3ppi.ppi_predictor.evaluator.evaluate import Evaluator

# Filter CSVs to proteins that actually have an embedding on disk.

Trainer(
    train_data_path="./data/filtered_ppi_data_train.csv",
    val_data_path="./data/filtered_ppi_data_val.csv",
    embeddings_dir="./data/embeddings",
    save_name="c3ppi_model",
    batch_size=512,
).run()

# Validation: reports metrics and selects the F1-maximising threshold.
Evaluator(mode="val",
          data_path="./data/filtered_ppi_data_val.csv",
          embeddings_dir="./data/embeddings",
          model_name="c3ppi_model",
          batch_size=512).run()

# Test: reuses the chosen threshold; writes PR/ROC curves and the confusion matrix.
Evaluator(mode="test",
          data_path="./data/filtered_ppi_data_test.csv",
          embeddings_dir="./data/embeddings",
          model_name="c3ppi_model",
          threshold=<val_threshold>,
          batch_size=512).run()
```

Best checkpoint: `registry/c3ppi_model/best_model.pt`. Metrics, PR/ROC SVGs
and `confusion_matrix.csv`: `output/c3ppi_model_results/`.

Reference numbers from the accompanying notebook run
(`03.ModelDev.ipynb`, validation set):
`Precision=0.988, Recall=0.961, F1=0.974, MCC=0.973, AUROC=0.998,
AP=0.991` at threshold `0.4261`.

## 5. Gut-microbiome inference

Notebooks in `notebooks/GUT_MB_Clinical/`:

1. `01.Init_EDA.ipynb` — sample-level EDA on `sampleID.csv`.
2. `02.BacAnalysis.ipynb` — builds the bacterial protein pool per disease
   (Healthy, CRC, …) from the Gut Microbiome Atlas metadata.
3. `03.EmbedProteins.ipynb` — pulls the sequences and embeds them with
   ProstT5 as above.
4. `04.Inference.ipynb` / `inference_script.py` — runs `PPI_Model` over the
   Cartesian product of `Pool_A` (human) × `Pool_B` (bacterial):

   ```python
   from inference_script import Inference

   Inference(
       embeddings_dir="./data/embeddings",
       output_dir="./output/Healthy",
       pairs={"Pool A": [...human ids...], "Pool B": [...bacterial ids...]},
       threshold=0.4261,
       best_model_path="registry/c3ppi_model/best_model.pt",
   ).run()
   ```

   The script pre-loads all `Pool B` embeddings into a cache and streams
   `Pool A` proteins one at a time, writing
   `output/<run>/per_protein_results/<p1>_predictions.parquet`.

5. `04.Prediction_Analysis.ipynb`, `05.Suppl.ipynb`, `06.chem_exp.ipynb` —
   downstream network / disorder / chemical-context analyses of the
   predicted interactome. These are exploratory and expect the outputs of
   step 4 on disk.

## 6. Notes / current limitations

- Notebook paths are relative to `notebooks/PPI_Model/` and
  `notebooks/GUT_MB_Clinical/`. There is no single CLI entry point yet — this
  is intentional; the plan for `gut_emberg` is to consolidate all three
  projects under one CLI.
- `ProteinEmbeddingService` requires the initial ProstT5 download
  (~1.1 GB in fp16) which is cached under `registry/ProstT5/`.
- Model registry / output roots are resolved four levels above `config.py`,
  i.e. `projects/c3ppi/`. Set `MODEL_REGISTRY` / `OUTPUT` there or via
  environment variables if a different location is preferred.
