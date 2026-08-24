import json
import pandas as pd
from itertools import product

GS_PATH = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/3did2020.csv"
PRED_PPIS = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/Prediction/Healthy_Unspecified/Predicted_PPIs_Healthy_Unspecified.txt"
HUM_PROT_DONAINS = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/human_proteins_domains.json"
UN_BAC_DONAINS = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/Unspecified/Unspecified_proteins_domains.json"
H_BAC_DONAINS = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/Healthy/Healthy_proteins_domains.json"

def main():
    df = pd.read_csv(GS_PATH, header=0)
    
    # Take top 10 according to Number of PDB entries
    df = df.sort_values(by=["Number of PDB entries"], ascending=False).reset_index(drop=True)
    print(df.head(10))

    # Read Human Protein Domains
    with open(HUM_PROT_DONAINS, "r") as f:
        hum_prot_domains = json.load(f)

    # Read Unspecified Bacteria Protein Domains
    with open(UN_BAC_DONAINS, "r") as f:
        un_bac_domains = json.load(f)

    # Read Healthy Bacteria Protein Domains
    with open(H_BAC_DONAINS, "r") as f:
        h_bac_domains = json.load(f)
    
    # Merge bacteria domains
    bac_domains = {**un_bac_domains, **h_bac_domains}

    # Read Predicted PPIs txt line by line
    pairs_freq = dict()
    with open(PRED_PPIS, "r") as f:
        for line in f:
            line = line.strip()
            p1 = line.split("\t")[0]
            p2 = line.split("\t")[1]

            if p1 not in hum_prot_domains or p2 not in bac_domains:
                continue

            p1_dom = hum_prot_domains[p1]
            p2_dom = bac_domains[p2]

            pfam_pairs = {tuple(sorted(pair)) for pair in product(p1_dom, p2_dom)}
            for pair in pfam_pairs:
                if pair not in pairs_freq:
                    pairs_freq[pair] = 0
                pairs_freq[pair] += 1

    # To dataframe
    
    df = pd.DataFrame.from_dict(pairs_freq, orient="index", columns=["Frequency"])
    df = df.sort_values(by=["Frequency"], ascending=False).reset_index()
    df.columns = ["Domain Pair", "Frequency"]
    print(df.head(10))


"""
        D1       D2  Number of PDB entries
0  PF00227  PF00227                  26287
1  PF07686  PF07686                   9298
2  PF10584  PF00227                   9265
3  PF07654  PF07654                   8625
4  PF07686  PF07654                   8311
5  PF00210  PF00210                   5707
6  PF10584  PF10584                   4921
7  PF00509  PF00509                   4607
8  PF00400  PF00400                   4391
9  PF00607  PF00607                   3358
          Domain Pair  Frequency
0  (PF00069, PF00072)    4217117
1  (PF00069, PF02518)    3098496
2  (PF00069, PF00512)    2482581
3  (PF00069, PF04542)    1861189
4  (PF00069, PF00486)    1693709
5  (PF00400, PF02518)    1622188
6  (PF00069, PF08281)    1497134
7  (PF00400, PF00512)    1316813
8  (PF00400, PF01381)    1223121
9  (PF00076, PF00534)    1076266
"""


if __name__ == "__main__":
    main()
