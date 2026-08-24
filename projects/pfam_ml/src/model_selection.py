import os
import argparse
import json
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from collections import defaultdict
from itertools import product
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, precision_recall_curve, confusion_matrix

def models_evaluation():
    """
        Evaluate models.
    """
    # Results directory
    results_dir = os.path.join(args.project_path, args.results_dir)
    methods = {
        "Association": "as_method",
        "Linear Programming": "lp_method",
        "Machine Learning (RF)": "ml_method",
        #"Machine Learning (XGBoost)": "ml_method"
    }
    
    total_metrics = {}
    for m, dir_name in methods.items():
        print("Method: {}".format(m))

        if m == "Association":
            # Metrics file:
            metrics_file = os.path.join(results_dir, dir_name, "metrics.json")

            # Read metrics file
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            # Golden standard metrics
            gsMetrics = metrics["GS"]
            
            gsMetrics = pd.DataFrame.from_dict(metrics["GS"], orient="index")
            gsMetrics.columns = ["Golden Standard Accuracy"]
            print("Golden standard metrics:")
            print(gsMetrics)

            total_metrics[m] = metrics["Tuned"]
        
        elif m == "Linear Programming":
            # Metrics file:
            metrics_file = os.path.join(results_dir, dir_name, "metrics.json")

            # Read metrics file
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            total_metrics[m] = metrics
        
        elif m == "Machine Learning (RF)":
            # Grid search metrics file:
            # Metrics file:
            metrics_file = os.path.join(results_dir, dir_name, "RF_metrics.json")

            # Read metrics file
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            gs_metrics_file = os.path.join(results_dir, dir_name, "RF_cv_results.csv")
            gs_metrics = pd.read_csv(gs_metrics_file, index_col=None)
            
            columns = [
                    "mean_fit_time",
                    "mean_score_time",
                    "param_rf__max_features", 
                    "param_rf__n_estimators", 
                    "mean_test_score"
                    ]
            gs_metrics = gs_metrics[columns]
            print("Grid search metrics:")
            print(gs_metrics)
            total_metrics[m] = metrics

        elif m == "Machine Learning (XGBoost)":
            # Metrics file:
            metrics_file = os.path.join(results_dir, dir_name, "XGB_metrics.json")

            # Read metrics file
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            # Grid search metrics file:
            gs_metrics_file = os.path.join(results_dir, dir_name, "XGB_cv_results.csv")
            gs_metrics = pd.read_csv(gs_metrics_file, index_col=None)

            columns = [
                    "mean_fit_time",
                    "mean_score_time",
                    "param_xgb__max_depth", 
                    "param_xgb__subsample", 
                    "mean_test_score"
                    ]
            gs_metrics = gs_metrics[columns]
            print("Grid search metrics:")
            print(gs_metrics)
            total_metrics[m] = metrics

        print("-"*50)

    print("Total metrics:")
    total_metrics = pd.DataFrame.from_dict(total_metrics, orient="index")
    print(total_metrics)
    print("-"*50)

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
    final_model_dir = os.path.join(results_dir, "final_model")
    if not os.path.isdir(final_model_dir):
        os.mkdir(final_model_dir)

    ddi_freq_file = os.path.join(final_model_dir, "ddi_freq.txt")

    with open(ddi_freq_file, "w") as f:
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

def obtain_vector(data, prot_domains,pfam2vec_model,ddi_inst, mode="train"):
    x, y = obtain_features(pfam2vec_model, data, prot_domains, ddi_inst)
    results_dir = os.path.join(args.project_path, args.results_dir)

    models_evaluation()
    
    final_model_dir = os.path.join(results_dir, "final_model")
    print("Mode: ", mode)
    print("x shape: ", x.shape)

    np.savez(os.path.join(final_model_dir, mode + "_vec.npz"), x=x, y=y)

def tune_final_model():
    """
        Tune final model by including the whole dataset: train + validation.
        Selected model is Random Forest.
    """

    # Results directory
    results_dir = os.path.join(args.project_path, args.results_dir)

    # ml_method directory
    ml_dir = os.path.join(results_dir, "ml_method")

    # final model directory
    final_model_dir = os.path.join(results_dir, "final_model")

    train_vec_file = os.path.join(final_model_dir, "train_vec.npz")

    # Load train and validation vectors
    train_vec = np.load(train_vec_file)

    # Concatenate train and validation vectors

    x_train = train_vec["x"]
    y_train = train_vec["y"]

    # Random Forest classifier with pipeline

    # Load best parameters
    best_params_file = os.path.join(ml_dir, "best_RF_params.txt")

    with open(best_params_file, "r") as f:
        best_params = eval(f.read())

    pipe = Pipeline([('scaler', StandardScaler()),
                        ('rf', RandomForestClassifier(random_state=0,n_jobs=-1))])

    # Set parameters
    # Drop threshold key
    best_params.pop("threshold")
    pipe.set_params(**best_params)

    # Fit model
    pipe.fit(x_train, y_train)

    # Save model
    model_file = os.path.join(final_model_dir, "final_model.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(pipe, f)

def predict():
    # Results directory
    results_dir = os.path.join(args.project_path, args.results_dir)
    # Machine learning method directory
    ml_dir = os.path.join(results_dir, "ml_method")
    
    # final model directory
    final_model_dir = os.path.join(results_dir, "final_model")

    test_vec_file = os.path.join(final_model_dir, "test_vec.npz")
    test_vec = np.load(test_vec_file)

    x_test = test_vec["x"]
    y_test = test_vec["y"]

    # Load model
    model_file = os.path.join(final_model_dir, "final_model.pkl")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_hat = model.predict_proba(x_test)
    y_hat = y_hat[:, 1]

    # Load threshold
    best_params_file = os.path.join(ml_dir, "best_RF_params.txt")

    with open(best_params_file, "r") as f:
        best_params = eval(f.read())
    
    threshold = best_params["threshold"]

    # Predicted labels
    y_pred = np.where(y_hat >= threshold, 1, 0)

    # Predictions to npz
    predictions_file = os.path.join(final_model_dir, "predictions.npz")
    np.savez(predictions_file, y_pred=y_pred, y_true=y_test, posts=y_hat)

def test_final_model():
    """
        Test final model.
    """
    print("Testing final model...")
    results_dir = os.path.join(args.project_path, args.results_dir)
    # final model directory
    final_model_dir = os.path.join(results_dir, "final_model")
    # Machine learning method directory

    predictions_file = os.path.join(final_model_dir, "predictions.npz")
    if not os.path.isfile(predictions_file):
        predict()

    # Load predictions
    predictions = np.load(predictions_file)
    y_pred = predictions["y_pred"]
    y_true = predictions["y_true"]
    y_hat = predictions["posts"]
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix using pandas
    cm = pd.DataFrame(cm, index=["Negative", "Positive"], columns=["Negative", "Positive"])
    print("Confusion matrix:")
    print(cm)

    # Plot precision-recall curve using plotly
    precision, recall, thresholds = precision_recall_curve(y_true, y_hat)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines"))
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=800,
        height=600,
        font=dict(
            size=18,
        )
    )

    # Change theme to plotly_white, use grid and Times New Roman font everywhere:
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            #gridcolor="black",
            #gridwidth=1,
            title_font_family="Times New Roman",
            tickfont_family="Times New Roman",
            title_font_size=40,
            tickfont_size=20,
        ),
        yaxis=dict(
            #gridcolor="black",
            #gridwidth=1,
            title_font_family="Times New Roman",
            tickfont_family="Times New Roman",
            title_font_size=40,
            tickfont_size=20,
        ),
    )

    #fig.show()
    # Save a high quality image
    fig.write_image(os.path.join(final_model_dir, "precision_recall_curve.png"))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines"))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600,
        font=dict(
            size=18,
        )
    )

    # Change theme to plotly_white, use grid and Times New Roman font everywhere:
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            #gridcolor="black",
            #gridwidth=1,
            title_font_family="Times New Roman",
            tickfont_family="Times New Roman",
            title_font_size=40,
            tickfont_size=20,
        ),
        yaxis=dict(
            #gridcolor="black",
            #gridwidth=1,
            title_font_family="Times New Roman",
            tickfont_family="Times New Roman",
            title_font_size=40,
            tickfont_size=20,
        ),
    )

    #fig.show()
    # Save a high quality image
    fig.write_image(os.path.join(final_model_dir, "roc_curve.png"))

    # AUC
    auc_score = auc(fpr, tpr)
    print("AUC score: {}".format(auc_score))

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: {}".format(accuracy))

    # F1 score
    f1 = f1_score(y_true, y_pred)
    print("F1 score: {}".format(f1))

    # Precision
    precision = precision_score(y_true, y_pred)
    print("Precision: {}".format(precision))

    # Recall
    recall = recall_score(y_true, y_pred)
    print("Recall: {}".format(recall))

def test_gut_exp():
    to_test = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/testingdf.csv"
    to_testDF = pd.read_csv(to_test)
    to_testDF.columns = ["PPI_No", "P1", "P2"]

    to_keep = [
        2544,
        2821,
        8253,
        9308,
        9523,
        9532,
        10100,
        10108,
        10266,
        11941,
        17857,
        17887,
        20263
        ]

    # keep only the PPIs we want to test
    to_testDF = to_testDF[to_testDF["PPI_No"].isin(to_keep)]

    print("Number of PPIs to test: {}".format(len(to_testDF)))

    to_testDF["label"] = 1

    pfam2vec_path = os.path.join(args.project_path, args.pfam2vec)

    # Load model
    pfam2vec_model = Word2Vec.load(pfam2vec_path)    
        
    dic = ''
    with open(ddi_freq_file,'r') as f:
        for i in f.readlines():
            dic=i
    ddi_inst = eval(dic)

    prot_domains_file = os.path.join(args.project_path, args.dataset_prot_domains)
    with open(prot_domains_file, "r") as f:
        prot_domains = json.load(f)

    x, y = obtain_features(pfam2vec_model, to_testDF, prot_domains, ddi_inst)

    # Load model
    model_file = os.path.join(final_model_dir, "final_model.pkl")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Predict
    y_hat = model.predict_proba(x)
    y_hat = y_hat[:, 1]

    # Load threshold
    best_params_file = os.path.join("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/ml_method", "best_RF_params.txt")

    with open(best_params_file, "r") as f:
        best_params = eval(f.read())
    
    threshold = best_params["threshold"]

    # Predicted labels
    y_pred = np.where(y_hat >= threshold, 1, 0)
    
    print("Experimental interactions: ")

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy: {}".format(accuracy))

    # F1 score
    f1 = f1_score(y, y_pred)
    print("F1 score: {}".format(f1))

    # Precision
    precision = precision_score(y, y_pred)
    print("Precision: {}".format(precision))

    # Recall
    recall = recall_score(y, y_pred)
    print("Recall: {}".format(recall))

    # Drop column label
    to_testDF = to_testDF.drop("label", axis=1)
    to_testDF["Probability"] = y_hat

    # Save
    to_testDF.to_csv(to_test.replace(".csv", "_results.csv"), index=False)



def argParse():
    """
        Parse script arguments.
    """
    parser = argparse.ArgumentParser(description='Domain - Domain interaction prediction.')
    # Project path
    parser.add_argument('-pp', '--project_path', type=str, help='Project path.', default="/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA")
    # Results directory
    parser.add_argument('-res', '--results_dir', type=str, help='Results directory.', default="Results")
    # Train dataset
    parser.add_argument('-train', '--train', type=str, help='Train dataset.', default="data/Dataset/train.csv")
    # Validation dataset
    parser.add_argument('-val', '--val', type=str, help='Validation dataset.', default="data/Dataset/val.csv")
    # Test dataset
    parser.add_argument('-test', '--test', type=str, help='Test dataset.', default="data/Dataset/test.csv")
    parser.add_argument('-ds_prot_dom', '--dataset_prot_domains', type=str, help='prot_domains.json.', default="data/Dataset/dataset_prot_domains.json")
    parser.add_argument('-pfam2vec', '--pfam2vec', type=str, help='Pfam2vec model.', default="Results/ml_method/pfam2vec.model")

    return parser.parse_args()

if __name__ == "__main__":
    args = argParse()

    # Results directory
    results_dir = os.path.join(args.project_path, args.results_dir)

    models_evaluation()
    
    final_model_dir = os.path.join(results_dir, "final_model")
    if not os.path.isdir(final_model_dir):
        os.mkdir(final_model_dir)

    ddi_freq_file = os.path.join(final_model_dir, "ddi_freq.txt")
    if not os.path.isfile(ddi_freq_file):
        # Read train dataset
        train_file = os.path.join(args.project_path, args.train)
        train = pd.read_csv(train_file, index_col=None)
        # Read validation dataset
        val_file = os.path.join(args.project_path, args.val)
        val = pd.read_csv(val_file, index_col=None)

        # Concatenate train and validation datasets
        train = pd.concat([train, val], ignore_index=True)

        # Read prot_domains
        prot_domains_file = os.path.join(args.project_path, args.dataset_prot_domains)
        with open(prot_domains_file, "r") as f:
            prot_domains = json.load(f)

        get_ddi_inst(train, prot_domains)

    
    # train vectors
    train_vec_file = os.path.join(final_model_dir, "train_vec.npz")
    # test vectors
    test_vec_file = os.path.join(final_model_dir, "test_vec.npz")

    if not os.path.isfile(train_vec_file) or not os.path.isfile(test_vec_file):
        # read train dataset
        train_file = os.path.join(args.project_path, args.train)
        train = pd.read_csv(train_file, index_col=None)
        # read validation dataset
        val_file = os.path.join(args.project_path, args.val)
        val = pd.read_csv(val_file, index_col=None)

        # Concatenate train and validation datasets
        train = pd.concat([train, val], ignore_index=True)
        # read test dataset
        test_file = os.path.join(args.project_path, args.test)
        test = pd.read_csv(test_file, index_col=None)

        pfam2vec_path = os.path.join(args.project_path, args.pfam2vec)

        # Load model
        pfam2vec_model = Word2Vec.load(pfam2vec_path)    
            
        dic = ''
        with open(ddi_freq_file,'r') as f:
            for i in f.readlines():
                dic=i
        ddi_inst = eval(dic)

        prot_domains_file = os.path.join(args.project_path, args.dataset_prot_domains)
        with open(prot_domains_file, "r") as f:
            prot_domains = json.load(f)

        obtain_vector(train,prot_domains,pfam2vec_model,ddi_inst, mode="train")
        obtain_vector(test,prot_domains,pfam2vec_model,ddi_inst, mode="test")

    final_model_file  = os.path.join(final_model_dir, "final_model.pkl")
    
    if not os.path.isfile(final_model_file):
        tune_final_model()

    #test_final_model()

    test_gut_exp()