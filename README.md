# gut_emberg

Personal research monorepo for my PhD work on **computational prediction of
protein–protein interactions (PPIs) between gut bacteria and the human host**.

Three complementary methods live here, spanning the axis from lightweight
domain-based statistics to structure-aware deep learning:

| # | Method | Representation | Model | Location |
|---|---|---|---|---|
| 1 | `pfam_ml`         | Pfam2Vec (Word2Vec on domain pairs) | Random Forest (+ XGB, Association Score, LP baselines) | `projects/pfam_ml/` |
| 2 | `struct_ppi_pred` | MAPE-PPI VAE structural embeddings   | Bi-directional Cross-Attention + MLP + Focal Loss     | external — [github.com/c3biolab/struct_ppi_pred](https://github.com/c3biolab/struct_ppi_pred) |
| 3 | `c3ppi`           | ProstT5 sequence embeddings          | Bi-directional Cross-Attention + MLP + Focal Loss     | `projects/c3ppi/` |

Method 2 is not yet vendored in this monorepo; the eventual goal is to
migrate and refine it here so that all three methods share one package and
CLI.

## Scope and motivation

Experimentally verified human–bacterial PPIs are scarce because the two
organisms are difficult to co-purify at scale. Computational predictors are
therefore the only way to obtain proteome-scale interactomes, and each of the
three methods here targets a different point on the accuracy/cost/coverage
curve:

- **`pfam_ml` — domain-based, cheap, interpretable.** Represents a PPI as the
  set of Pfam domain pairs formed by its two proteins, learns a Pfam2Vec
  embedding via Word2Vec over the DDI vocabulary, and classifies with a
  Random Forest tuned against a golden-standard DDI set (3did). Association
  Score and Linear Programming baselines are included. Suitable for
  proteome-scale scoring on CPU, no structures required.
- **`struct_ppi_pred` — structure-aware, deep.** Uses the pre-trained
  MAPE-PPI variational autoencoder to encode protein 3D contact graphs into
  fixed-size embeddings, then fuses the two protein embeddings with a
  Bi-directional Cross-Attention module and predicts interaction with an MLP
  head trained under Focal Loss. Published in Kiouri, Batsis, Chasapis,
  *Proteomes* 2025 (13, 10), [doi:10.3390/proteomes13010010](https://doi.org/10.3390/proteomes13010010).
- **`c3ppi` — sequence-only, deep, plug-and-play.** Same fusion + head as
  method 2, but the structural embedder is replaced with **ProstT5**
  (`Rostlab/ProstT5_fp16`) mean-pooled per residue. This removes the entire
  structural pre-processing pipeline (contact maps, node features, VAE
  checkpoint) and lets the model score any UniProt ID for which a sequence
  exists.

The three approaches share a common evaluation protocol (train / val / test
with class-imbalanced negatives, F1-optimal threshold on validation, PR/ROC
on test) and a common downstream analysis: score every candidate pair between
a bacterial pool (per disease mode from the Gut Microbiome Atlas) and the
human gut+brain protein pool, then run network analytics (degree, IVI,
UniRef90 / fuzzy clusters) on the predicted interactome.

## Repository layout

```
gut_emberg/
├── data/                 # small shared reference tables
│   ├── UniProtNormalizedTabular-default.txt   # "Pickla" human PPI reference
│   ├── dl_net.txt                             # sample predicted network
│   └── sampleID.csv                           # sample metadata
├── projects/
│   ├── pfam_ml/          # method 1 — Pfam2Vec + RF (see projects/pfam_ml/README.md)
│   └── c3ppi/            # method 3 — ProstT5 + BiCA (see projects/c3ppi/README.md)
├── README.md             # this file
└── friend.md             # onboarding note for collaborators
```

Method 2 lives at [github.com/c3biolab/struct_ppi_pred](https://github.com/c3biolab/struct_ppi_pred)
until it is refactored into this repo.

## Reproducing each project

Each project has its own README with concrete, high-level steps. In short:

- **`projects/pfam_ml`**
  ```bash
  cd projects/pfam_ml/src
  python data_parser.py            --project_path /path/to/DDA --gut_mode Healthy
  python models/random_forest.py   --project_path /path/to/DDA
  python model_selection.py        --project_path /path/to/DDA
  python deployment.py             --project_path /path/to/DDA --gut_mode Healthy,Unspecified
  ```
  Full details: [`projects/pfam_ml/README.md`](projects/pfam_ml/README.md).

- **`projects/c3ppi`**
  ```bash
  cd projects/c3ppi
  rye sync            # or pip install -e .
  # then run notebooks in order:
  #   notebooks/PPI_Model/01.C3PPI_Data.ipynb
  #   notebooks/PPI_Model/02.EmbedProteins.ipynb
  #   notebooks/PPI_Model/03.ModelDev.ipynb
  # gut-microbiome inference:
  #   notebooks/GUT_MB_Clinical/01..04*.ipynb  (or inference_script.py)
  ```
  Full details: [`projects/c3ppi/README.md`](projects/c3ppi/README.md).

- **`struct_ppi_pred`** — follow the `rye`-based instructions in the
  upstream repo. Data preparation is handled entirely by the unmodified
  [MAPE-PPI pipeline](https://github.com/LirongWu/MAPE-PPI); processed
  feature files used in the paper are available from the corresponding
  author on request.

## Current status

The `projects/` subtrees are the source used to write the manuscripts and
should be treated as **research drafts**:

- Hard-coded absolute paths still linger in `pfam_ml` and are overridable via
  the `-pp/--project_path` CLI flag.
- `c3ppi` is packaged (`pyproject.toml`) but is driven mainly through
  notebooks — no unified CLI yet.
- `struct_ppi_pred` is the most polished of the three (Rye-managed, `task`
  targets for embedding / training / evaluation / inference) and will serve
  as the template when the three methods are consolidated.

The end goal is a single installable package (`gut_emberg`) exposing a CLI
that can dispatch to any of the three predictors with a shared data schema.

## Citation

If you use any part of this repository, please cite the relevant publication
(added here as the papers are released):

- Method 2 — Kiouri, D. P.; Batsis, G. C.; Chasapis, C. T. *Structure-Based
  Deep Learning Framework for Modeling Human–Gut Bacterial Protein
  Interactions.* Proteomes **2025**, 13, 10.
  [https://doi.org/10.3390/proteomes13010010](https://doi.org/10.3390/proteomes13010010)
- Method 1 and Method 3 — publications forthcoming.

## License

TBD. Third-party code retains its own license (MAPE-PPI is MIT; ProstT5 is
governed by the Rostlab model card).
