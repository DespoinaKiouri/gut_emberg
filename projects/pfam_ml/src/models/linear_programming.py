import os
import argparse
import pandas as pd
import json
import pulp
from itertools import product
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

def optimization(prot_train_pairs, prot_domains):
    prob = pulp.LpProblem("Domain-Domain_Interaction_Inference", pulp.LpMinimize)
    # Read the golden standard data:
    gs = pd.read_csv(os.path.join(args.project_path, args.gold_standard))
    # gs to set of tuples
    gs_set = set()
    for _, row in gs.iterrows():
        d1 = row["D1"]
        d2 = row["D2"]
        # Sort the tuple to ensure consistent ordering
        pfam_pair = tuple(sorted((d1, d2)))
        gs_set.add(pfam_pair)

   # Read protein interaction training data
   # Domain pair set, no need for specific case P1==P2 --> set only adds unique elements
    domain_pairs = set()
    # Find domain pairs for every line in prot_train_pairs(csv)
    for index, row in prot_train_pairs.iterrows():
        p1 = row["P1"]
        p2 = row["P2"]
        p1_domains = prot_domains[p1]
        p2_domains = prot_domains[p2]
        # Cartesian product: all possible combinations of domain pairs for every row
        pfamPairs = list(product(p1_domains, p2_domains))
        # Sort the tuple to ensure consistent ordering,  add to domain_pairs
        for pfamA, pfamB in pfamPairs:
            pfam_pair = tuple(sorted((pfamA, pfamB)))
            domain_pairs.add(pfam_pair)
    # Unique domain pairs
    domain_pairs = set(domain_pairs)
    print("Number of domain pairs: ", len(domain_pairs))

    print("Start LP")

    print("Define variables and objective function")
    # Variables: x_ij: probability that domain i interacts with domain j (0-1), Continuous
    x = pulp.LpVariable.dicts("x", domain_pairs, lowBound=0, upBound=1, cat=pulp.LpContinuous)

    # Objective Function
    prob += pulp.lpSum([x[i, j] for i, j in domain_pairs]), "Total_Interaction_Score"

    print("Define constraints")
    # Constraints
    # 1. If pair in golden standard set, then x_ij = 1
    for p in gs_set:
        # See if the gs pair is in the domain_pairs set
        if p in domain_pairs:
            prob += x[p[0], p[1]] == 1
            
    #2. For every PPI, the sum of x_ijs for all domain pairs in the PPI must be >= 1
    for index, row in prot_train_pairs.iterrows():
        p1 = row["P1"]
        p2 = row["P2"]
        # Find domain pairs for every line in prot_train_pairs(csv) fro prot_domains_dict
        p1_domains = prot_domains[p1]
        p2_domains = prot_domains[p2]
        #Pair calculation (cartesian)
        pfamPairs = list(product(p1_domains, p2_domains))
        # Sort the tuple to ensure consistent ordering,  add to domain_pairs
        ddi_per_ppi = set()
        for pfamA, pfamB in pfamPairs:
            pfam_pair = tuple(sorted((pfamA, pfamB)))
            ddi_per_ppi.add(pfam_pair)
        # Add constraint2: sum of x_ijs >= 1
        prob += pulp.lpSum(x[pair[0], pair[1]] for pair in ddi_per_ppi) >= 1

    print("Solving LP...")
    # Solve the linear program
    prob.solve()   
    # Print status: See if the problem has been solved
    print("Status:", pulp.LpStatus[prob.status])
    # Save values:
    print("Save values...")
    values = {}
    for v in prob.variables():
        if v.varValue > 0:
            values[v.name] = v.varValue
    # Value - counts of values
    vc = Counter(values.values())
    print(vc)
    return values

def validation(opt_scores, prot_domains):
     # Read test dataset
    test = pd.read_csv(os.path.join(args.project_path, args.dataset, "val.csv"))

    # DDI prediction for test dataset
    predictions = list()
    ppi_label_true = list()
    # Get Uniprot IDs for P1, P2 from test dataset
    for k,v in test.T.to_dict().items():
        P1 = v["P1"]
        P2 = v["P2"]
        # Get pfam domains for P1, P2
        P1_pfam = prot_domains[P1]
        P2_pfam = prot_domains[P2]
        # Cartesian product: all possible combinations
        pfamPairs = product(P1_pfam, P2_pfam)
        #Initialization of interaction=False
        interaction = False
        for pfamA, pfamB in pfamPairs:
            pair = tuple(sorted((pfamA, pfamB)))
            pair = "x_('{}',_'{}')".format(pair[0], pair[1])
            if pair in opt_scores:
                predictions.append(1)
                interaction = True
                break
        if not interaction:
            predictions.append(0)

        ppi_label_true.append(v["label"])

    acc = accuracy_score(ppi_label_true, predictions)
    f1 = f1_score(ppi_label_true, predictions)
    precision = precision_score(ppi_label_true, predictions)
    recall = recall_score(ppi_label_true, predictions)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    print(metrics)
    return metrics  

def run(args):
    """
        Run the algorithm.
    """
    resDir = os.path.join(args.project_path, args.results_dir)
    algDir = os.path.join(args.project_path, args.alg)
    optFile = os.path.join(algDir, args.opt_scores)
    if not os.path.isfile(optFile):
        os.makedirs(resDir,exist_ok=True)
        os.makedirs(algDir,exist_ok=True)
        # Load training dataset and prot_domains.json
        train = pd.read_csv(os.path.join(args.project_path, args.train))
        prot_train_pairs = train[train["label"] == 1][["P1", "P2"]]
        prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
        with open(prot_domains_path, 'r') as f:
            prot_domains = json.load(f)
        values = optimization(prot_train_pairs, prot_domains)
        # Save values to json
        with open(optFile,'w+') as f:
            json.dump(values, f, indent=4)
    #Read opt.scores and prot_domains.json
    with open(optFile,'r') as f:
        opt_scores = json.load(f)
    prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
    with open(prot_domains_path, 'r') as f:
        prot_domains = json.load(f)
    metrics = validation(opt_scores, prot_domains)
    # Save metrics to json
    metricsFile = os.path.join(algDir, args.metrics)
    with open(metricsFile,'w+') as f:
        json.dump(metrics, f, indent=4)

def argParse():
    """
        Parse script arguments.
    """
    parser = argparse.ArgumentParser(description='Domain - Domain interaction prediction.')
    # Project path
    parser.add_argument('-pp', '--project_path', type=str, help='Project path.', default="/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA")
    # Dataset path
    parser.add_argument('-ds', '--dataset', type=str, help='Dataset to parse.', default='data/Dataset')
    # Golden standard data
    parser.add_argument('-gs', '--gold_standard', type=str, help='Golden standard data.', default="data/3did2020.csv")
    #  "prot_domains.json"
    parser.add_argument('-ds_prot_dom', '--dataset_prot_domains', type=str, help='prot_domains.json.', default="data/Dataset/dataset_prot_domains.json")
    # Train dataset
    parser.add_argument('-train', '--train', type=str, help='Train dataset.', default="data/Dataset/train.csv")
    # Results directory
    parser.add_argument('-res', '--results_dir', type=str, help='Results directory.', default="Results")
    # Results/alg
    parser.add_argument('-alg', '--alg', type=str, help='Algorithm.', default="Results/lp_method")
    # opt_scores
    parser.add_argument('-opt', '--opt_scores', type=str, help='opt_scores.', default="opt.json")
    # metrics
    parser.add_argument('-metrics', '--metrics', type=str, help='metrics.', default="metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = argParse()
    run(args)