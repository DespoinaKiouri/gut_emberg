import os
import pandas as pd
import argparse
import itertools
import json
import numpy as np
import math
import plotly.figure_factory as ff
import plotly.graph_objects as go
from collections import defaultdict
from plotly.graph_objs import *
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_pfam_counts(train,prot_domains):
    """
        Get the PFAM counts.
    """
    print("Get PFAM counts...")
    pfam_counts = defaultdict(int)
    #Find unique proteins in train
    un_proteins = list(set(train["P1"].unique().tolist() + train["P2"].unique().tolist()))

    #Filter prot_domains based on unique proteins in train
    new_prot_dict = {}
    for p in un_proteins:
        new_prot_dict[p] = prot_domains[p]
    
    prot_domains = new_prot_dict

    for k,v in prot_domains.items():
        pfam = v
        for pf in pfam:
            pfam_counts[pf] += 1

    pfam_counts = dict(pfam_counts)
    return pfam_counts, prot_domains

def association_score(train, prot_domains, pfam_counts):
    #Calculate Pfam pair frequency
    pair_info = defaultdict(int)
    print("Calculate Pfam pair frequency...")
    
    for k, v in train.T.to_dict().items():
        if v["label"] == 1:
            PA = v["P1"]
            PB = v["P2"]
            
            PA_pfam = prot_domains[PA]
            PB_pfam = prot_domains[PB]
            
            # Cartesian product: all possible combinations
            pfamPairs = itertools.product(PA_pfam, PB_pfam)
            
            temp = set()
            for pfamA, pfamB in pfamPairs:
                # Sort the tuple to ensure consistent ordering
                pfam_pair = tuple(sorted((pfamA, pfamB)))
                temp.add(pfam_pair)
            pfamPairs = temp
            
            for pfamA, pfamB in pfamPairs:
                # Sort the tuple to ensure consistent ordering
                pfam_pair = tuple(sorted((pfamA, pfamB)))
                pair_info[pfam_pair] += 1

    pair_info = dict(pair_info)

    '''
        Calculate the AS score: 
        Let Mij is the number of interacting pairs (Pm, Pn) in which Di  D(m) and Dj  D(n). 
        The association measure of (Di, Dj), which is the ratio of the number of occurrences to the 
        total number of protein pairs containing (Di, Dj):'''
    as_score = {}
    for k,v in pair_info.items():
        pfamA = k[0]
        pfamB = k[1]
        as_score[k] = v / (pfam_counts[pfamA] * pfam_counts[pfamB])
        if as_score[k] > 1:
            print(v,pfam_counts[pfamA],pfam_counts[pfamB], k)

    return as_score

def load_scores(as_path):
    """
    """
    print("Load scores...")
    dic = ''

    with open(as_path,'r') as f:
        for i in f.readlines():
            dic=i
    as_score = eval(dic) # this is orignal dict with instace dict
    df_as_score = pd.DataFrame.from_dict(as_score, orient='index', columns=['AS Scores'])
    return as_score, df_as_score

def distribution_plots (args,df_as_score):
    """
    """
    algDir = os.path.join(args.project_path, args.alg)
    
    data = df_as_score["AS Scores"].values
    kde = gaussian_kde(data)
    x_values = np.linspace(min(data), max(data), 1000)
    pdf_values = kde(x_values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=pdf_values, mode='lines', name='PDF'))

    fig.update_layout(title="Probability Density Function",
                    xaxis_title="Value",
                    yaxis_title="Density",
                    # Shitch off y axis ticks
                    yaxis=dict(
                        showticklabels=False
                    ),
                    )

    #fig.show()
    # Save high guality image
    fig.write_image(os.path.join(algDir, "pdf.png"), width=800, height=600, scale=2)

    fig = go.Figure()
    fig.add_trace(go.Box(x=df_as_score["AS Scores"].values))
    #fig.show()
    # Save high guality image
    fig.write_image(os.path.join(algDir, "box.png"), width=800, height=600, scale=2)

def expQuantile(p,l):
    """
    """
    return -1 * (math.log(1-p) / l)

def calculate_statistics(df_as_score):
    """
    """
    minVal = df_as_score["AS Scores"].min()
    maxVal = df_as_score["AS Scores"].max()
    mean = df_as_score["AS Scores"].mean()

    l = 1/mean

    Q1 = expQuantile(0.25,l)
    Q2 = expQuantile(0.5,l)
    Q3 = expQuantile(0.75,l)
    
    # Interquartile range:
    lower = Q1 - 1.5 * (Q3 - Q1)
    upper = Q3 + 1.5 * (Q3 - Q1)
    
    stats = {
        "min": minVal,
        "max": maxVal,
        "mean": mean,
        "Q1": Q1,
        "Q2": Q2,
        "Q3": Q3,
        "Lower_bound": lower,
        "Upper_bound": upper
    }
    
    return stats

def tune_thresh(as_score, prot_domains, threshold):
    # Read test dataset
    test = pd.read_csv(os.path.join(args.project_path, args.dataset, "val.csv"))

    # Predictions
    predictions = list()
    true = list()
    for k,v in test.T.to_dict().items():
        P1 = v["P1"]
        P2 = v["P2"]

        P1_pfam = prot_domains[P1]
        P2_pfam = prot_domains[P2]

        # Cartesian product: all possible combinations
        pfamPairs = itertools.product(P1_pfam, P2_pfam)
        interaction = False
        for pfamA, pfamB in pfamPairs:
            pair = tuple(sorted((pfamA, pfamB)))
            if pair in as_score:
                score = as_score[pair]
                if score >= threshold:
                    predictions.append(1)
                    interaction = True
                    break
            else:
                score = 0
        
        if not interaction:
            predictions.append(0)

        true.append(v["label"])

    acc = accuracy_score(true, predictions)
    f1 = f1_score(true, predictions)
    precision = precision_score(true, predictions)
    recall = recall_score(true, predictions)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

    return metrics        

def golden_standard_test_accuracy (as_score, threshold):
    gold_standard = pd.read_csv(os.path.join(args.project_path, args.gold_standard))
    Golden_strandard_scores = {}
    
    for k,v in gold_standard.T.to_dict().items():
        D1 = v["D1"]
        D2 = v["D2"]
        
        pair = tuple(sorted((D1, D2)))
        if pair in as_score:
            score = as_score[pair]
            if score >= threshold:
                pred = 1
            else:
                pred = 0
            Golden_strandard_scores[pair] = {"score":score, "pred":pred, "true":1}
        else:
            Golden_strandard_scores[pair] = {"score":0, "pred":0, "true":0}

    gsDF = pd.DataFrame.from_dict(Golden_strandard_scores, orient='index', columns=['score', 'pred', 'true'])
    accuracy = accuracy_score(gsDF["true"].values, gsDF["pred"].values)
    return accuracy

def run(args):
    """
        Run the algorithm.
    """
    resDir = os.path.join(args.project_path, args.results_dir)
    algDir = os.path.join(args.project_path, args.alg)
    asScoreFile = os.path.join(algDir, args.as_score)

    if not os.path.isfile(asScoreFile):
        os.makedirs(resDir,exist_ok=True)
        os.makedirs(algDir,exist_ok=True)

        # Load training dataset and prot_domains.json
        train = pd.read_csv(os.path.join(args.project_path, args.train))
        prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
        with open(prot_domains_path, 'r') as f:
            prot_domains = json.load(f)

        pfam_counts, prot_domains = get_pfam_counts(train,prot_domains)
        as_score = association_score(train, prot_domains, pfam_counts)

        # Save as_score
        with open(asScoreFile, 'w') as f:
            f.write(str(as_score))

    # Load as_score
    scoreDict, df_as_score = load_scores(asScoreFile)
    print("Potential DDIs on training data {}".format(len(df_as_score)))
    statsFile = os.path.join(algDir, args.stats)
    if not os.path.isfile(statsFile):
        distribution_plots (args,df_as_score)
        stats = calculate_statistics(df_as_score)
        with open(statsFile,'w+') as f:
            json.dump(stats, f, indent=4)

    with open(statsFile,'r') as f:
        stats = json.load(f)

    prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
    with open(prot_domains_path, 'r') as f:
        prot_domains = json.load(f)

    print("Statistics:")
    print(json.dumps(stats, indent=4))

    # Metrics file:
    metricsFile = os.path.join(algDir, args.metrics)
    if not os.path.isfile(metricsFile):
        metrics = {}
        
        metrics["GS"] = {}

        for meas, value in {
                            "Q1":stats["Q1"], 
                            "Q2":stats["Q2"], 
                            "Q3":stats["Q3"],
                            "Upper Bound":stats["Upper_bound"]}.items():
            
            acc = golden_standard_test_accuracy(scoreDict, value)
            print("{} - {:.2}, Accuracy: {}".format(meas,value,acc))
            print("-"*50)
            metrics["GS"][meas] = acc

        # Best threshold according to max accuracy
        thres = max(metrics["GS"], key=metrics["GS"].get)
        thres = stats[thres]

        metrics_final = tune_thresh(scoreDict, prot_domains, thres)
        
        print(metrics_final)
        
        metrics["Tuned"] = metrics_final

        with open(metricsFile,'w+') as f:
            json.dump(metrics, f, indent=4)
    
    else:
        with open(metricsFile,'r') as f:
            metrics = json.load(f)
        
        print("Metrics:")
        print(json.dumps(metrics, indent=4))


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
    parser.add_argument('-alg', '--alg', type=str, help='Algorithm.', default="Results/as_method")
    # as_score
    parser.add_argument('-as', '--as_score', type=str, help='as_score.', default="as_scores.txt")
    # stats
    parser.add_argument('-stats', '--stats', type=str, help='stats.', default="stats.json")  
    # Metrics
    parser.add_argument('-metrics', '--metrics', type=str, help='metrics.', default="metrics.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = argParse()
    run(args)