# pfam_ml — Domain-based ML for Gut Bacteria–Human PPI Prediction

Structure-agnostic modelling of gut bacteria–human protein–protein interactions
(PPIs) based on **Pfam domain composition**. Each PPI is represented as the set
of Pfam pairs formed by the Cartesian product of the two proteins' domains, and
these pairs are embedded via a Word2Vec model trained on the DDI vocabulary
("Pfam2Vec"). A Random Forest classifier is trained on aggregated pair
representations. Two baselines are included: a statistical **Association Score**
and a **Linear Programming** DDI inference.

> Status: research draft. Paths inside the scripts are hard-coded to the
> author's workstation (`/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA`)
> and must be overridden with the `-pp/--project_path` CLI flag (and, in a few
> ad-hoc scripts, edited directly). The pipeline is reproducible but is not yet
> packaged.

---

## 1. Method overview

For every PPI `(P1, P2)`:

1. Look up the Pfam domain lists of `P1` and `P2` in UniProt.
2. Enumerate the sorted domain pairs `D_i × D_j` (Cartesian product).
3. Represent each pair as `concat(w2v(D_i), w2v(D_j))` (dim 200) using a
   Word2Vec model where sentences are the domain-pair sets of training PPIs.
4. Aggregate pair vectors of a PPI as a frequency-weighted sum (DDI frequency
   computed from the training PPIs, then normalised to `[0, 1]`).
5. Feed the 200-dim PPI vector into a scaled Random Forest (grid-searched over
   `n_estimators ∈ {50, 100, 150}` and `max_features ∈ {sqrt, log2, None}`),
   selecting the decision threshold that maximises F1 on the validation set.

Baselines:

- **Association Score** (`src/models/association_method.py`): pair frequency
  normalised by the product of the two domains' marginal counts; threshold
  chosen by matching the 3did golden standard.
- **Linear Programming** (`src/models/linear_programming.py`): assigns a
  probability in `[0, 1]` to each domain pair by minimising the total sum under
  the constraints that (i) golden-standard pairs are pinned to 1, and (ii) the
  sum of pair-probabilities per training PPI is ≥ 1.

## 2. Data layout

The scripts expect a project root with (paths shown relative to `--project_path`):

```
data/
├── ExperimentalInteractions.txt            # Pan human–bacterial PPIs (HPIBD, IntAct, PHISTO, MorCVD)
├── PPIdomainminerinput.txt                 # PPIs from MINT, IntAct, DIP, HPRD, BioGrid, SIFTS
├── HumanProteomeUniprotIDs.xlsx            # All human UniProt IDs (reviewed + unreviewed)
├── 3did2020.csv                            # Golden standard DDIs from 3did
├── ProteinDatabase/                        # UniProt JSONs, one per protein (auto-populated)
├── ClusterDatabase/                        # UniRef90 JSONs (auto-populated)
├── human_proteins.json                     # UniProtID -> file name in ProteinDatabase
├── human_proteins_domains.json             # UniProtID -> list[PfamID]
├── ExperimentalProteins.json               # same, for experimental proteins
├── ExperimentalProteinsDomains.json
├── Dataset/
│   ├── dataset_ppis.txt                    # combined experimental + PPIDomainMiner PPIs
│   ├── dataset_prot_domains.json           # union of all protein → PfamIDs
│   ├── train.csv / val.csv / test.csv      # produced by data_parser.py
│   └── Negative Dataset/
│       ├── UniProtNormalizedTabular-default.txt  # Pickla human PPI reference
│       ├── normal_tissue.csv               # Protein Atlas tissue expression
│       ├── prot_organ.json                 # Protein → organs (auto-populated)
│       └── negative_pairs.txt              # Sampled negatives (auto-populated)
└── Gut Data/
    ├── GM_bacteria_taxid_and_proteome_ids_mapping.xlsx
    ├── GM_bacteria_Gut_Microbiome_Atlas.xlsx
    ├── vect_atlas.csv
    ├── Human_proteins_gut_f.csv
    └── Human_proteins_brain_f.csv
```

See the file-by-file description at the bottom of this README.

## 3. Environment

Python ≥ 3.8 with:

- `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`
- `gensim` (Word2Vec)
- `pulp` (LP baseline)
- `plotly`, `kaleido` (figures)
- `openpyxl`, `xlsxwriter` (Excel I/O)
- `requests`, `thefuzz` (UniProt REST + fuzzy gene matching)
- `sknetwork`, `python-igraph`, `influential` (network analysis in
  `deployment.py` / `network_analysis_alessandro.py`)

The project has no lock file; a minimal install is sufficient to run the
training pipeline, and network-analysis extras only need to be installed for
the deployment step.

## 4. Reproduction — training pipeline

All scripts accept a `-pp / --project_path` flag pointing at your data root.
Run from `projects/pfam_ml/src/`.

### 4.1 Build the dataset

```bash
python data_parser.py \
    --project_path /path/to/DDA \
    --gut_mode Healthy
```

`data_parser.py` runs four stages in order:

1. `ExperimentalProteins` — resolves UniProt IDs of the experimental +
   PPIDomainMiner PPIs, downloads their entries into `ProteinDatabase/`, and
   caches Pfam domain lists.
2. `HumanProteins` — same for the reviewed human proteome.
3. `DatasetParser` — generates negatives by:
   - mapping proteins to organs via `normal_tissue.csv`,
   - keeping pairs whose two proteins share **no organ**, are not in Pickla or
     in the experimental PPIs, and whose domain pairs do not appear in
     `3did2020.csv`;
   then samples negatives to match the positive count and splits
   train/val/test with a DDI-disjointness constraint (test PPIs preferentially
   contain DDIs unseen in training).
4. `GutDataParser` / `CreteGutDataParser` — builds a bacterial protein pool for
   the requested disease mode (`Healthy`, `Unspecified`, `Colorectal cancer`, …
   or `Crete_Patients`), downloads bacterial proteomes from the EBI Proteins
   API and produces UniRef90 clusters and pFam-based "fuzzy" clusters.

Output: `data/Dataset/{train,val,test}.csv` (columns `P1, P2, label`) and
`data/Dataset/dataset_prot_domains.json`.

### 4.2 Train the Random Forest

```bash
python models/random_forest.py --project_path /path/to/DDA
```

Steps performed by `run()`:

1. Build the corpus of Pfam pair sentences (one sentence per PPI, one word per
   sorted domain pair) into `Results/ml_method/corpus.txt`.
2. Train `Word2Vec(vector_size=100, window=2, sg=1, negative=5, epochs=8)` and
   save it as `pfam2vec.model`.
3. Compute per-DDI normalised frequencies (`ddi_inst.txt`).
4. Vectorise `train.csv` and `val.csv` into `Results/ml_method/vec/*.npz`.
5. Grid-search the RF pipeline (`StandardScaler` + `RandomForestClassifier`)
   with a `PredefinedSplit` (train + val), tune the decision threshold on the
   validation PR curve, and store:
   - `best_RF_params.txt`, `RF_cv_results.csv`
   - `RF_model.pkl`, `RF_metrics.json`

### 4.3 (Optional) Train the XGBoost variant

```bash
python models/xgb.py --project_path /path/to/DDA
```

Same pipeline, tuning `max_depth` and `subsample`. Random Forest outperformed
XGBoost in our experiments; XGBoost is kept for ablation.

### 4.4 Baselines

```bash
python models/association_method.py --project_path /path/to/DDA
python models/linear_programming.py  --project_path /path/to/DDA
```

Both write metrics under `Results/{as_method,lp_method}/`.

### 4.5 Refit on train + validation and evaluate on test

```bash
python model_selection.py --project_path /path/to/DDA
```

This script:

1. Aggregates the metrics of RF / XGB / AS / LP into a comparison table.
2. Concatenates train + val, recomputes DDI frequencies and vectors under
   `Results/final_model/`.
3. Fits the RF with the best hyper-parameters, dumps `final_model.pkl`.
4. Evaluates on the test set: writes `predictions.npz`,
   `precision_recall_curve.png`, `roc_curve.svg` and prints
   accuracy / F1 / precision / recall / AUC.

By default the script also calls `test_gut_exp()`, which evaluates the model on
a small hand-curated set of Crete gut PPIs (`testingdf.csv`). Comment it out if
you don't have that file.

## 5. Deployment (network prediction)

`src/deployment.py` scores every possible pair between the bacterial pool of
one or more disease modes and the human gut+brain protein pool, then runs
network analysis on the resulting graph.

```bash
python deployment.py \
    --project_path /path/to/DDA \
    --gut_mode Healthy,Unspecified
```

Output (under `Results/Prediction/<mode>/`):

- `Predicted_PPIs_<mode>.txt` — tab-separated `bac_prot`, `hum_prot`, score.
- `protDegree.xlsx`, `Bacteria_info_<mode>.csv` — per-protein / per-bacterium
  statistics.
- `Uniref90_ClusterDegree.xlsx`, `Fuzzy_ClusterDegree.xlsx` — cluster-level
  networks (UniRef90 and gene+Pfam fuzzy clusters).
- `iviDF.csv` — Integrated Value of Influence per node.
- `Aggregated_results_<mode>.xlsx` — final deliverable spreadsheet.

Auxiliary scripts in `projects/pfam_ml/`:

- `network_analysis_alessandro.py` — additional network descriptors and
  visualisations for the Alessandro collaboration.
- `mobiDB_run.py` — queries MobiDB to compute intrinsic disorder fraction for
  hub human proteins in the predicted network.
- `pan_scirpt.py` — annotates the top human hubs with GO Biological Process /
  Molecular Function terms.
- `cc_pres.py` — sanity check of the most frequent predicted DDIs against the
  3did golden standard.

These are single-use analysis scripts; edit the paths at the top of each file
before running.

## 6. Data files (reference)

1. **GM_bacteria_taxid_and_proteome_ids_mapping.xlsx** — representative
   UniProt ProteomeID (highest BUSCO) per gut bacterial species, organised by
   phylum, with optional taxid mapping.
2. **GM_bacteria_Gut_Microbiome_Atlas.xlsx** — species → MSP code, region
   enrichment, disease association.
3. **vect_atlas.csv** — per-sample abundance vectors indexed by MSP.
4. **Human_proteins_gut_f.csv** / **Human_proteins_brain_f.csv** — Protein
   Atlas gut / brain proteins with tissue, cell type, reliability.
5. **UniProtNormalizedTabular-default.txt** — human direct PPI reference
   ("Pickla") used to exclude known positives from the negative pool.
6. **normal_tissue.csv** — Protein Atlas per-gene tissue expression, used to
   build the protein-to-organ map that drives negative sampling.
7. **ExperimentalInteractions.txt** — experimentally verified pan
   human–bacterial PPIs from HPIBD, IntAct, PHISTO, MorCVD.
8. **PPIdomainminerinput.txt** — PPIDomainMiner training PPIs from six PPI
   databases (STRING excluded).
9. **HumanProteomeUniprotIDs.xlsx** — full reviewed + unreviewed human
   proteome UniProt IDs.
10. **3did2020.csv** — 3did golden-standard DDIs used as positive DDI priors
    and, in LP, as pinned constraints.
