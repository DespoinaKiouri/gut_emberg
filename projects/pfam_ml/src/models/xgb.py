import os
import argparse
import time
import logging  # Setting up the loggings to monitor gensim
import multiprocessing
import pickle
import pandas as pd
import numpy as np
import json
from itertools import product
from collections import defaultdict
from itertools import product
import xgboost as xgb
from gensim.models import Word2Vec
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

def buid_corpus(train,prot_domains,corpus_path):
    with open (corpus_path, 'w') as f:
        #Get all domains per UniProtID from train and prot_domains
        for k,v in train.T.to_dict().items():
            p1_domains = prot_domains[v["P1"]]
            p2_domains = prot_domains[v["P2"]]
            # Cartesian product: all possible combinations of domain pairs for every row
            pfamPairs = list(product(p1_domains, p2_domains))
            temp = set()
            # Sort the tuple to ensure consistent ordering,  add to domain_pairs to temp
            for pfamA, pfamB in pfamPairs:
                #Alphanumeric sorting
                pfam_pair = tuple(sorted((pfamA, pfamB)))
                temp.add(pfam_pair)
            # Every row(PPI) =  a sentence, with every possible domain pair as a word
            # Write sentences to corpus.txt
            pfam_pairs = temp
            sentences = " ".join([p[0] + " " + p[1] for p in pfam_pairs])
            f.write(sentences + "\n")

def word2vec_model(corpus_path):
    # Train Word2Vec model
    model = Word2Vec(vector_size=100, 
                     window=2, 
                     sg = 1,
                     negative=5,
                     min_count=0)

    model.build_vocab(corpus_file=corpus_path)

    model.train(corpus_file=corpus_path, 
                total_examples=model.corpus_count,
                total_words=model.corpus_total_words, 
                epochs=8,
                report_delay=1)
    # Save model
    model.save(os.path.join(args.project_path, args.pfam2vec))

def get_ddi_inst(train,prot_domains):
    pair_inst = defaultdict(int)
    for k, v in train.iterrows():  
        PA_pfam = prot_domains[v["P1"]]
        PB_pfam = prot_domains[v["P2"]]
        # Cartesian product: all possible combinations
        pfamPairs = product(PA_pfam, PB_pfam)
        temp = set()
        for pfamA, pfamB in pfamPairs:
            # Sort the tuple to ensure consistent ordering
            pfam_pair = tuple(sorted((pfamA, pfamB)))
            temp.add(pfam_pair)
        pfamPairs = temp
        
        for pfamA, pfamB in pfamPairs:
            # Sort the tuple to ensure consistent ordering
            pfam_pair = tuple(sorted((pfamA, pfamB)))
            pair_inst[pfam_pair] += 1
    pair_inst = dict(pair_inst)

    # Pair instances to frequency: inst/total instances
    pair_freq = {}
    for k,v in pair_inst.items():
        pair_freq[k] = v/len(pair_inst)
    
    # Normalized frequency
    max_freq = max(pair_freq.values())
    
    for k,v in pair_freq.items():
        pair_freq[k] = v/max_freq

    # Pair_freq min and max values
    min_freq = min(pair_freq.values())
    max_freq = max(pair_freq.values())

    print("Min freq: ", min_freq)
    print("Max freq: ", max_freq)

    # Save to txt:
    with open(os.path.join(args.project_path, args.ddi_inst), 'w') as f:
        f.write(str(pair_freq))

def obtain_features(model, df, prot_domains, ddi_freq):
    x = list()
    y = list()
    pfamVocab = model.wv.key_to_index
    for k,v in df.iterrows():
        p1_domains = prot_domains[v["P1"]]
        p2_domains = prot_domains[v["P2"]]
        
        pfam_pairs = {tuple(sorted(pair)) for pair in product(p1_domains, p2_domains) if pair[0] in pfamVocab and pair[1] in pfamVocab}
        reprDict = {p: np.concatenate((model.wv[p[0]], model.wv[p[1]])) for p in pfam_pairs}
        
        if len(pfam_pairs) == 1:
            ppi_repr = reprDict[list(pfam_pairs)[0]]
        else:
            ppi_repr = np.zeros(200)
            for p in pfam_pairs:
                freq = ddi_freq.get(p)    
                ppi_repr += reprDict[p] * freq
        
        # If all zeros
        if np.all(ppi_repr == 0):
            print("All zeros: ", v)
            continue
        else:
            x.append(ppi_repr)
            y.append(v["label"])

    return np.array(x), np.array(y)

def obtain_vector(prot_domains,pfam2vec_model,ddi_inst, mode="train"):
    data = pd.read_csv(os.path.join(args.project_path, args.dataset, mode + ".csv"))
    x, y = obtain_features(pfam2vec_model, data, prot_domains, ddi_inst)
    np.savez(os.path.join(args.project_path, args.vec, mode + "_vec.npz"), x=x, y=y)

def XGB_grid_search(x_train, y_train, x_val, y_val):

    hps = {
        'xgb__max_depth': [3, 5, 7],
        'xgb__subsample': [0.5, 0.7, 1]
        }

    cv = PredefinedSplit(test_fold=[-1 for _ in range(x_train.shape[0])] + [0 for _ in range(x_val.shape[0])])

    pipe = Pipeline([('scaler', StandardScaler()),
                        ('xgb', xgb.XGBClassifier(device="cuda",n_jobs=multiprocessing.cpu_count()))])

    grid = GridSearchCV(pipe, hps, cv=cv, verbose=4)
    
    # Concatenate train and validation datasets
    x = np.concatenate((x_train, x_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    grid.fit(x, y)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()

    # Parameters to file: 
    with open(os.path.join(args.project_path, args.best_XGB_params), 'w') as f:
        f.write(str(grid.best_params_))

    # Save cv_results_
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(os.path.join(args.project_path, args.cv_results), index=False)

def dev_model(x_train, y_train, x_val, y_val):

    # Read best parameters and import
    with open(os.path.join(args.project_path, args.best_XGB_params), 'r') as f:
        best_params = eval(f.read())

    pipe = Pipeline([('scaler', StandardScaler()),
                    ('xgb', xgb.XGBClassifier(device="cuda",n_jobs=multiprocessing.cpu_count()))])

    pipe.set_params(**best_params)
    
    print("Fit model with best parameters: ", best_params)
    # Train model
    pipe.fit(x_train, y_train)
    # Prediction of probability to validation set
    yhat = pipe.predict_proba(x_val)
    probs = yhat[:, 1]

    # Precision-recall threshold, precision-recall curve (a curve that gives prec/rec values for multiple threshholds)
    prec, rec, thresholds = precision_recall_curve(y_val, probs)
    
    # Convert to f score
    fscore = (2 * prec * rec) / (prec + rec)

    # Find the max f score and the threshold associated with it prec, rec, thresholds.
    ix = np.argmax(fscore)
    print("Best Threshold={}, F1-Score={}, Prec={}, Rec={}".format(thresholds[ix], fscore[ix], prec[ix], rec[ix]))

    # Save threshold to best_XGB_params
    best_params["threshold"] = thresholds[ix]

    with open(os.path.join(args.project_path, args.best_XGB_params), 'w') as f:
        f.write(str(best_params))

    # Save model
    with open(os.path.join(args.project_path, args.alg, "XGB_model.pkl"), 'wb') as f:
        pickle.dump(pipe, f)

def eval_model(x_val, y_val):

    # Load best parameters
    with open(os.path.join(args.project_path, args.best_XGB_params), 'r') as f:
        best_params = eval(f.read())

    threshold = best_params["threshold"]

    # Load model
    with open(os.path.join(args.project_path, args.alg, "XGB_model.pkl"), 'rb') as f:
        pipe = pickle.load(f)

    # Prediction of probability to validation set
    yhat = pipe.predict_proba(x_val)
    probs = yhat[:, 1]

    # Convert to binary prediction
    yhat = np.where(probs > threshold, 1, 0)

    # Evaluate model
    accuracy = accuracy_score(y_val, yhat)
    precision = precision_score(y_val, yhat)
    recall = recall_score(y_val, yhat)
    f1 = f1_score(y_val, yhat)

    # Print all:
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Save metrics to json
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    with open(os.path.join(args.project_path, args.alg, "XGB_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

def run(args):
    metrics_file = os.path.join(args.project_path, args.alg, "XGB_metrics.json")
    
    if not os.path.isfile(metrics_file):
        ml_folder = os.path.join(args.project_path, args.alg)
        os.makedirs(ml_folder, exist_ok=True)
        
        prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
        # Json: Uniprot ID, list of domains(PfamIDs)
        with open(prot_domains_path, 'r') as f:
            prot_domains = json.load(f)
        # Corpus: TXT file with all possible domain pairs for every PPI, token --> Pfam, Sentence--> PPI
        corpus_path = os.path.join(args.project_path, args.corpus)
        if not os.path.exists(corpus_path):
            train = pd.read_csv(os.path.join(args.project_path, args.train))
            val = pd.read_csv(os.path.join(args.project_path, args.val))
            test = pd.read_csv(os.path.join(args.project_path, args.test))
            dataset = pd.concat([train, val, test], ignore_index=True)
            buid_corpus(dataset,prot_domains,corpus_path)
        # Pfam2vec model: Takes corpus and creates vector representation of every PfamID 
        pfam2vec_path = os.path.join(args.project_path, args.pfam2vec)
        if not os.path.exists(pfam2vec_path):
            word2vec_model(corpus_path)
        # Read training dataset
        train = pd.read_csv(os.path.join(args.project_path, args.train))

        ddi_inst_path = os.path.join(args.project_path, args.ddi_inst)
        if not os.path.exists(ddi_inst_path):
            get_ddi_inst(train,prot_domains)
        
        train_vec_path = os.path.join(args.project_path, args.vec, "train_vec.npz")
        val_vec_path = os.path.join(args.project_path, args.vec, "val_vec.npz")

        if not (os.path.isfile(train_vec_path) and os.path.isfile(val_vec_path)):
            os.makedirs(os.path.join(args.project_path, args.vec), exist_ok=True)
            prot_domains_path = os.path.join(args.project_path, args.dataset_prot_domains)
        
            with open(prot_domains_path, 'r') as f:
                prot_domains = json.load(f)

            # Load model
            pfam2vec_model = Word2Vec.load(pfam2vec_path)    
            
            dic = ''
            with open(ddi_inst_path,'r') as f:
                for i in f.readlines():
                    dic=i
            ddi_inst = eval(dic)
            
            obtain_vector(prot_domains,pfam2vec_model,ddi_inst, mode="train")
            obtain_vector(prot_domains,pfam2vec_model,ddi_inst, mode="val")

        # Load vector representation
        train_vec = np.load(train_vec_path)
        x_train = train_vec["x"]
        y_train = train_vec["y"]

        print("Dimension of x_train: ", x_train.shape)

        val_vec = np.load(val_vec_path)
        x_val = val_vec["x"]
        y_val = val_vec["y"]

        print("Dimension of x_val: ", x_val.shape)
        
        # Grid search
        xgb_parameters_file = os.path.join(args.project_path, args.best_XGB_params)
        cv_results = os.path.join(args.project_path, args.cv_results)
        if not (os.path.isfile(xgb_parameters_file) and os.path.isfile(cv_results)):
            XGB_grid_search(x_train, y_train, x_val, y_val)
            dev_model(x_train, y_train, x_val, y_val)
            eval_model(x_val, y_val)
    else:
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(metrics)

def argParse():
    """
        Parse script arguments.
    """
    parser = argparse.ArgumentParser(description='Domain - Domain interaction prediction.')
    # Project path
    parser.add_argument('-pp', '--project_path', type=str, help='Project path.', default="/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA")
    # Dataset path
    parser.add_argument('-ds', '--dataset', type=str, help='Dataset to parse.', default='data/Dataset')
    # Experimental data
    parser.add_argument('-ed', '--exp_data', type=str, help='Experimental data.', default="data/ExperimentalInteractions.txt")
    # Golden standard data
    parser.add_argument('-gs', '--gold_standard', type=str, help='Golden standard data.', default="data/3did2020.csv")
    #  "prot_domains.json"
    parser.add_argument('-ds_prot_dom', '--dataset_prot_domains', type=str, help='prot_domains.json.', default="data/Dataset/dataset_prot_domains.json")
    # Train dataset
    parser.add_argument('-train', '--train', type=str, help='Train dataset.', default="data/Dataset/train.csv")
    # Validation dataset
    parser.add_argument('-val', '--val', type=str, help='Validation dataset.', default="data/Dataset/val.csv")
    # Test dataset
    parser.add_argument('-test', '--test', type=str, help='Test dataset.', default="data/Dataset/test.csv")
    # Results directory
    parser.add_argument('-res', '--results_dir', type=str, help='Results directory.', default="Results")
    # Results/alg
    parser.add_argument('-alg', '--alg', type=str, help='Algorithm.', default="Results/ml_method")
    # Corpus path
    parser.add_argument('-corpus', '--corpus', type=str, help='Corpus path.', default="Results/ml_method/corpus.txt")
    # Pfam2vec model
    parser.add_argument('-pfam2vec', '--pfam2vec', type=str, help='Pfam2vec model.', default="Results/ml_method/pfam2vec.model")
    # DDI instances
    parser.add_argument('-ddi_inst', '--ddi_inst', type=str, help='DDI instances.', default="Results/ml_method/ddi_inst.txt")
    # Vector representation directory
    parser.add_argument('-vec', '--vec', type=str, help='Vector representation directory.', default="Results/ml_method/vec")
    # best_XGB_params
    parser.add_argument('-best_XGB_params', '--best_XGB_params', type=str, help='best_XGB_params.', default="Results/ml_method/best_RXGBparams.txt")
    # CV Results
    parser.add_argument('-cv_results', '--cv_results', type=str, help='CV Results.', default="Results/ml_method/XGB_cv_results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = argParse()
    run(args)