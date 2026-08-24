import os
import argparse
import json
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import sknetwork
import igraph as ig
from thefuzz import fuzz
from itertools import product
from gensim.models import Word2Vec
from influential import ivi


def get_protein_pools():
     # Bacterial Proteins Pool
    gut_bac_prot = {}
    # Gut Mode
    gutMode = args.gut_mode

    for m in gutMode:
        print("Fetch disease association proteins", m)
        # Gut Bacterial Proteins Pfam Directory --> Get all json files
        modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
        proteinsFile = os.path.join(modeDir, "{}_proteins.json".format(m))
        
        with open(proteinsFile, 'r') as f:
            modeGutProts = json.load(f)
            print("Number of Gut Bacterial Proteins with pFam for {}: {}".format(m,len(modeGutProts)))

        # Append to gut_bac_prot
        for k,v in modeGutProts.items():
            gut_bac_prot[k] = v

    gb_hum_prot_file = os.path.join(args.project_path, args.gut_data_dir, "human_proteins.json")

    with open(gb_hum_prot_file, 'r') as f:
        gb_hum_prot = json.load(f)
               
    print("Number of Gut Bacterial Proteins with pFam:", len(gut_bac_prot), "Number of Gut and Brain Human Proteins with pFam:", len(gb_hum_prot))
    print("Potential protein pairs:", len(gut_bac_prot)*len(gb_hum_prot))
    return gut_bac_prot, gb_hum_prot

def calculate_gene_similarity(genes1, genes2, threshold=0.7):
    for g1 in genes1:
        for g2 in genes2:
            ratio = fuzz.ratio(g1, g2)
            if ratio > threshold:
                return True
    return False

def load_protein_features(prot):
    prot_file = os.path.join(args.project_path, args.protein_db, prot + ".json")
    with open(prot_file, "r") as f:
        prot_res = json.load(f)

    genes = {gene["geneName"]["value"] for gene in prot_res["to"]["genes"] if "geneName" in gene}
    synonyms = {s["value"] for gene in prot_res["to"]["genes"] if "synonyms" in gene for s in gene["synonyms"]}
    gene_list = list(genes.union(synonyms, [prot_res["to"]["uniProtkbId"].split("_")[0]]))
    #print(genes)
    pfam_list = [cr["id"] for cr in prot_res["to"]["uniProtKBCrossReferences"] if cr["database"] == "Pfam"]

    return {"genes": gene_list,"pfam": pfam_list}

def find_pfam_equality(pfam1, pfam2):
    '''for p1 in pfam1:
        for p2 in pfam2:
            if p1 == p2:
                return True
    return False'''
    # Exactly the same pfam lists:
    if pfam1 == pfam2:
        return True

def group_proteins(prot_domains,group_file):
    grouped_dict = {}
    for idx, (key, value) in enumerate(prot_domains.items()):
        # Convert the list to a tuple for use as a dictionary key
        tuple_value = tuple(value)
        
        # Add the original key to the group corresponding to the tuple_value
        if tuple_value in grouped_dict:
            grouped_dict[tuple_value].append(key)
        else:
            grouped_dict[tuple_value] = [key]

        print("Protein: ", idx, " / ", len(prot_domains))

    print("Number of groups: ", len(grouped_dict))
    
    # Write to txt file in this form: group_idx \t protein1 \t protein2 \t ...
    with open(group_file, "w") as f:
        for idx, (key, value) in enumerate(grouped_dict.items()):
            f.write("{}\t{}\n".format(idx, "\t".join(value)))

    return grouped_dict

def group_similar_proteins(prot_list, group_file):
    """
    """
    with open(group_file, "w") as f:
        cluster_idx = 0
        features = {prot: load_protein_features(prot) for prot in prot_list}
        # Copy features
        features_copy = features.copy()

        for pidx, (k1,v1) in enumerate(features.items()):
            # Check if protein is in temp file
            if k1 in features_copy:
                sims = set()
                for k2,v2 in features_copy.items():
                    if k1 == k2:
                        continue
                    if k2 in features_copy:        
                        if find_pfam_equality(v1["pfam"], v2["pfam"]):
                            sims.add(k2)

                sims.add(k1)
                if len(sims) > 0:
                    
                    # Write to file
                    # cluster_idx : set
                    f.write("{}\t{}\n".format(cluster_idx, "\t".join(sims)))
                    cluster_idx += 1
                    # delete from features_copy
                    for k in sims:
                        del features_copy[k]
                    
            print("Protein: ", pidx, " / ", len(features), " - ", len(features_copy))

def extract_interactions(human_group_file,gut_bac_group_file,resInfoDir):
    perProtResDir = os.path.join(resInfoDir,args.per_protein_results)
    os.makedirs(perProtResDir, exist_ok=True)
    # Load human groups
    with open(human_group_file, "r") as f:
        human_groups = {}
        for line in f:
            line = line.strip()
            line = line.split("\t")
            # line[0] to keys and the rest to values
            human_groups[line[0]] = line[1:]

    # Load gut bac groups
    with open(gut_bac_group_file, "r") as f:
        gut_bac_groups = {}
        for line in f:
            line = line.strip()
            line = line.split("\t")
            # line[0] to keys and the rest to values
            gut_bac_groups[line[0]] = line[1:]

    # Read human protein domains
    with open(args.human_protein_domains, "r") as f:
        hum_prot_domains = json.load(f)
        f.close()

    # Read gut bac protein domains
    gutMode = args.gut_mode
    gut_bac_domains = {}
    for m in gutMode:
        print("Fetch disease association proteins", m)
        # Gut Bacterial Proteins Pfam Directory --> Get all json files
        modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
        proteinsFile = os.path.join(modeDir, "{}_proteins_domains.json".format(m))

        with open(proteinsFile, 'r') as f:
            modeGutProts = json.load(f)

        # Append to gut_bac_prot
        for k,v in modeGutProts.items():
                gut_bac_domains[k] = v

    #Load Pfam2Vec model
    pfam2vec_path = os.path.join(args.project_path, args.pfam2vec)
    pfam2vec_model = Word2Vec.load(pfam2vec_path) 
    pfamVocab = pfam2vec_model.wv.key_to_index

    # Final model dir
    final_model_dir = os.path.join(args.project_path, "Results/final_model")
    ddi_inst_path = os.path.join(final_model_dir, args.ddi_inst)
    dic = ''
    with open(ddi_inst_path,'r') as f:
        for i in f.readlines():
            dic=i
    ddi_inst = eval(dic)

    rfModel = os.path.join(final_model_dir, "final_model.pkl")
    with open(rfModel, "rb") as f:
        model = pickle.load(f)
        model.n_jobs = -1

    # Read best parameters txt
    rf_best_params_path = os.path.join(args.project_path, "Results/ml_method/best_RF_params.txt")
    with open(rf_best_params_path, "r") as f:
        best_params = eval(f.read())
    
    threshold = best_params["threshold"]
    print("Threshold:", threshold)

    for hum_group_idx, hum_group in human_groups.items():
        print("Human group: ", hum_group_idx, " / ", len(human_groups))
        # Choose a protein from the group
        hum_prot = hum_group[0]
        x_features = list()
        to_pred = list()
        hp_pfam = hum_prot_domains[hum_prot]

        for gut_group_idx, gut_group in gut_bac_groups.items():
            # Choose a protein from the group
            gut_prot = gut_group[0]
            bp_pfam = gut_bac_domains[gut_prot]

            # Create a set of pfam pairs
            pfam_pairs = {tuple(sorted(pair)) for pair in product(hp_pfam, bp_pfam) if pair[0] in pfamVocab and pair[1] in pfamVocab}
            # Keep pairs that exist in ddi_inst
            pfam_pairs = {pair for pair in pfam_pairs if pair in ddi_inst}

            reprDict = {p: np.concatenate((pfam2vec_model.wv[p[0]], pfam2vec_model.wv[p[1]])) for p in pfam_pairs}

            if len(reprDict) > 0:
                if len(pfam_pairs) == 1:
                    ppi_repr = reprDict[list(pfam_pairs)[0]]
                else:
                    ppi_repr = np.zeros(200)
                    for p in pfam_pairs:
                        freq = ddi_inst.get(p, 0)
                        if freq > 0:
                            ppi_repr += reprDict[p] * freq
                # if not ppis_repr zeros:
                if not np.all(ppi_repr == 0):
                    x_features.append(ppi_repr)
                    to_pred.append(gut_group_idx)
            
            
        if len(x_features) > 0:
            x_features = np.array(x_features)
            print("Number of PPIs to predict:", len(x_features))

            pred_proba = model.predict_proba(x_features)[:,1]
            # Predicted PPIs
            pred = np.where(pred_proba > threshold, 1, 0)
            protInterInfo = {}
            for idx, bacGroupIdx in enumerate(to_pred):
                if pred[idx] == 1:
                    for bacP in gut_bac_groups[bacGroupIdx]:
                        protInterInfo[bacP] = pred_proba[idx]

            # Write to file as many times as the number of human proteins in the group
            for hp in hum_group:
                reFile = os.path.join(perProtResDir, hp + ".json")
                with open(reFile, "w") as f:
                    json.dump(protInterInfo, f, indent=4)
                    f.close()
            
            print("Number of interacting proteins: ", len(protInterInfo))

def aggregate_results(resInfoDir,gb_hum_prot,predPPIs):
    """
    """
    tot = 0
    perProtResDir = os.path.join(resInfoDir,args.per_protein_results)

    # Filter and write to txt file:
    with open(predPPIs, "w") as ppf:
        for idxx, (pid,_) in enumerate(gb_hum_prot.items()):
            print(idxx,"/",len(gb_hum_prot))
            if os.path.isfile((os.path.join(perProtResDir, "{}.json".format(pid)))):
                try:
                    with open (os.path.join(perProtResDir, "{}.json".format(pid)), "r") as f:
                        data = json.load(f)                                
                        for k,v in data.items():
                            # Write pid, k to file:
                            if v >= 0.99:
                                ppf.write("{}\t{}\n".format(pid, k))
                                tot += 1   
                except Exception as e:
                    print("Error with file {}: {}".format(pid,e))
                    continue

def extract_degree_info(resInfoDir,predPPIs):
    human_prots = {}
    bacterial_prots = {}

    with open(predPPIs, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            hum_prot = line[0]
            bac_prot = line[1]

            if not hum_prot in human_prots:
                human_prots[hum_prot] = 1
            else:
                human_prots[hum_prot] += 1

            if not bac_prot in bacterial_prots:
                bacterial_prots[bac_prot] = 1
            else:
                bacterial_prots[bac_prot] += 1

    # Human degree
    human_degree = pd.DataFrame.from_dict(human_prots, orient='index', columns=["degree"])

    # Bacterial degree
    bacterial_degree = pd.DataFrame.from_dict(bacterial_prots, orient='index', columns=["degree"])

    # Save to excel: Each DF a new sheet:
    with pd.ExcelWriter(os.path.join(resInfoDir,"protDegree.xlsx")) as writer:
        human_degree.to_excel(writer, sheet_name='HumanProteins')
        bacterial_degree.to_excel(writer, sheet_name='BacterialProteins')

def pickla_comparison(resInfoDir):
    # Read protDegree.xlsx
    protDegree = os.path.join(resInfoDir,"protDegree.xlsx")
    protDegree = pd.read_excel(protDegree, sheet_name="HumanProteins")
    protDegree = protDegree.rename(columns={"Unnamed: 0":"HumanProtein"})

    # Read pickla
    pickla_file = os.path.join(args.project_path, args.pikla_path)
    picklaDF = pd.read_csv(pickla_file, sep="\t")

    # Calculate degree for every protein
    pickla_degree = {}
    for k,v in picklaDF.T.to_dict().items():
        if v["InteractorA"] in pickla_degree:
            pickla_degree[v["InteractorA"]] += 1
        else:
            pickla_degree[v["InteractorA"]] = 1

        if v["InteractorB"] in pickla_degree:
            pickla_degree[v["InteractorB"]] += 1
        else:
            pickla_degree[v["InteractorB"]] = 1

    pickla_degree = pd.DataFrame.from_dict(pickla_degree, orient='index', columns=["pickla_degree"])
    pickla_degree = pickla_degree.reset_index()
    pickla_degree = pickla_degree.rename(columns={"index":"HumanProtein"})

    # Merge dataframes
    prot_degreeDF = pd.merge(protDegree, pickla_degree, left_on="HumanProtein", right_on="HumanProtein", how="inner")
    
    # Pearson correlation
    corr = prot_degreeDF[["degree","pickla_degree"]].corr(method='pearson')
    print("Pearson correlation of degree and pickla_degree: ")
    print(corr)
    
    # Scatter plot
    fig = px.scatter(
        prot_degreeDF, 
        x="degree", 
        y="pickla_degree", 
        hover_data=["HumanProtein"]
    )

    # Update layout
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="Degree",  # Add title text if needed
                font=dict(
                    family="Times New Roman",
                    size=50
                )
            ),
            tickfont=dict(
                family="Times New Roman",
                size=30
            ),
            # Uncomment grid options if desired
            # gridcolor="black",
            # gridwidth=1
        ),
        yaxis=dict(
            title=dict(
                text="Pickle Degree",  # Add title text if needed
                font=dict(
                    family="Times New Roman",
                    size=50
                )
            ),
            tickfont=dict(
                family="Times New Roman",
                size=30
            ),
            # Uncomment grid options if desired
            # gridcolor="black",
            # gridwidth=1
        )
    )

    # Save the image
    img_name = os.path.join(resInfoDir, "pickla_comparison.svg")
    fig.write_image(img_name, width=1920, height=1080, scale=1)




def extractBacterialInfo(args,resInfoDir,mode):
    """
        Extract info about bacteria of predicted interactions.
    """
    dfList = list()
    for m in mode:
        gutDataDir = os.path.join(args.project_path, args.gut_data_dir)
        gutBactInfo = os.path.join(gutDataDir,"{}/gutBac_{}.csv".format(m,m))
        gutBactInfo = pd.read_csv(gutBactInfo)
        
        # Recheck - Maybe dublicates but they are actually the same.
        gutBactInfo = gutBactInfo.set_index("Proteome ID(with higher BUSCO)")
        gutBactInfo = gutBactInfo.T.to_dict()
        
        gutProteomsDir = os.path.join(args.project_path, args.gut_data_dir, m, args.gut_proteoms_dir)
        modeDFList = list()
        for pt in os.listdir(gutProteomsDir):
            # read pickle
            ptFile = os.path.join(gutProteomsDir, pt)
            ptFile = pd.read_pickle(ptFile)
            subDF = pd.DataFrame(ptFile, columns=["Protein"])
            proteome = pt.split(".")[0]
            if proteome in gutBactInfo:
                subDF["Proteome"] = proteome
                
                v = gutBactInfo[proteome]

                for k,v in v.items():
                    subDF[k] = v
                
                modeDFList.append(subDF)

        modeDF = pd.concat(modeDFList)
        dfList.append(modeDF)

    gutProteomsDF = pd.concat(dfList)
    gutProteomsDF = gutProteomsDF.reset_index(drop=True)
    #print(gutProteomsDF)
    
    # Read interactions info
    protDegree = os.path.join(resInfoDir,"protDegree.xlsx")
    protDegree = pd.read_excel(protDegree, sheet_name="BacterialProteins")
    protDegree = protDegree.rename(columns={"Unnamed: 0":"BacProtein"})    

    # Merge
    gutIntProtDF = pd.merge(gutProteomsDF, protDegree, left_on="Protein", right_on="BacProtein", how="inner")
    
    #print(gutIntProtDF)
    # Save to csv
    

    # Group by Proteome and calculate the mean of degree. Keep abundance and name:
    if "name" in gutIntProtDF.columns:
        nameCol = "name"
    else:
        nameCol = "name_cr"
    gutIntProtDF = gutIntProtDF.groupby(["Proteome","abundance",nameCol]).agg({"degree":"mean"})
    gutIntProtDF = gutIntProtDF.reset_index()
    gutIntProtDF = gutIntProtDF.rename(columns={"degree":"mean_degree"})
    modeStr = "_".join(mode)
    gutIntProtDF.to_csv(os.path.join(resInfoDir,"Bacteria_info_{}.csv".format(modeStr)), index=False)
    print(gutIntProtDF)    
    # pearson correlation
    corr = gutIntProtDF[["mean_degree","abundance"]].corr(method='pearson')
    print("Pearson correlation of mean_degree and abundance: ")
    print(corr)

    # Scatter plot
    fig = px.scatter(
        gutIntProtDF,
        x="mean_degree",
        y="abundance",
        color=nameCol,
        size="mean_degree",
        hover_data=["Proteome"]
    )

    # Disable legend if not needed
    fig.update_layout(showlegend=False)

    # Update layout for aesthetics
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title=dict(
                text="Mean Degree",  # Add a meaningful x-axis title
                font=dict(
                    family="Times New Roman",
                    size=50
                )
            ),
            tickfont=dict(
                family="Times New Roman",
                size=30
            ),
            # Uncomment grid options if needed
            # gridcolor="black",
            # gridwidth=1
        ),
        yaxis=dict(
            title=dict(
                text="Abundance",  # Add a meaningful y-axis title
                font=dict(
                    family="Times New Roman",
                    size=50
                )
            ),
            tickfont=dict(
                family="Times New Roman",
                size=30
            ),
            # Uncomment grid options if needed
            # gridcolor="black",
            # gridwidth=1
        )
    )

    # Save a high-quality image
    img_name = os.path.join(resInfoDir, f"Bacteria_info_{modeStr}.svg")
    fig.write_image(img_name, width=1920, height=1080, scale=1)


def UniRef90ClusterInteractions(resInfoDir,mode):
    # Bacterial Proteins Pool
    gut_bac_uniref = {}
    # Gut Mode
    gutMode = args.gut_mode

    for m in gutMode:
        print("Fetch disease association proteins", m)
        # Gut Bacterial Clusters Pfam Directory --> Get all json files
        modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
        UniRef90File = os.path.join(modeDir, "{}_cluster_info.txt".format(m))
        
        with open(UniRef90File, 'r') as f:
            modeGutUniRef90 = eval(f.read())
            print("Number of Gut Bacterial Clusters for {}: {}".format(m,len(modeGutUniRef90)))

        # Append to gut_bac_uniref
        for k,v in modeGutUniRef90.items():
            gut_bac_uniref[k] = v

    gb_hum_clusters_file = os.path.join(args.project_path, args.gut_data_dir, "human_proteins_ref90.txt")

    with open(gb_hum_clusters_file, 'r') as f:
        gb_hum_clusters = eval(f.read())
               
    print("Number of Gut Bacterial Protein Clusters:", len(gut_bac_uniref), "Number of Gut and Brain Human Protein Clusters:", len(gb_hum_clusters))

    # Protein interactions dictionary of every protein
    perProtResDir = os.path.join(resInfoDir,args.per_protein_results)
    
    # Uniref90Cluster-Cluster interaction file path
    clusterClusterFile = os.path.join(resInfoDir,"Uniref90_ClusterClusterInteractions.txt")
    # Open Uniref90Cluster-Cluster interaction file to write:
    with open(clusterClusterFile, "w") as fc:
        # For every human cluster, find the interactions of representative protein with bacterial clusters
        for u,hc in gb_hum_clusters.items():
            # Representative protein
            repr_p = u
            # Find protein file for repr_p: key--> interactiong protein, value--> probability
            repr_p_file = os.path.join(perProtResDir, repr_p + ".json")
            # If file exists, open file
            if os.path.isfile(repr_p_file):
                with open(repr_p_file, "r") as f:
                    repr_p_data = json.load(f)
                    # a. Check every interaction file of repr_p, keep only interactions with prob >= 0.99 AND
                    # b. Check if interacting protein is a representative in gut_bac_uniref
                    # Make dictionary: key--> interacting protein if is representative in bacterial cluster, value--> probability
                    repr_p_data = {k:v for k,v in repr_p_data.items() if k in gut_bac_uniref and v >= 0.99}
                    f.close()
                    # If dictionary is not empty, write to txt file
                    if len(repr_p_data) > 0:
                        for k,v in repr_p_data.items():
                            fc.write("{}\t{}\n".format(repr_p,k))
            else:
                continue
def extract_uniref90_degree_info(resInfoDir):
    uniref_clusters_interaction_file = os.path.join(resInfoDir, "Uniref90_ClusterClusterInteractions.txt")

    human_clusters = {}
    bacterial_clusters = {}
    with open(uniref_clusters_interaction_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            human_cluster = line[0]
            bacterial_cluster = line[1]

            if not human_cluster in human_clusters:
                human_clusters[human_cluster] = 1
            else:
                human_clusters[human_cluster] += 1
            
            if not bacterial_cluster in bacterial_clusters:
                bacterial_clusters[bacterial_cluster] = 1
            else:
                bacterial_clusters[bacterial_cluster] += 1

    # Human degree
    human_degree = pd.DataFrame.from_dict(human_clusters, orient='index', columns=["degree"])

    # Bacterial degree
    bacterial_degree = pd.DataFrame.from_dict(bacterial_clusters, orient='index', columns=["degree"])

    # Save to excel: Each DF a new sheet:
    with pd.ExcelWriter(os.path.join(resInfoDir,"Uniref90_ClusterDegree.xlsx")) as writer:
        human_degree.to_excel(writer, sheet_name='HumanClusters')
        bacterial_degree.to_excel(writer, sheet_name='BacterialClusters')

def fuzzy_clusters_interactions(resInfoDir):
    # Read fuzzy human clusters
    fuzzy_human_clusters_file = os.path.join(args.project_path, args.gut_data_dir, "fuzzy_human_clusters.txt") 
    
    # Read clusters
    fuzzy_human_clusters = {}
    with open(fuzzy_human_clusters_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            cluster_id = line[0]
            cluster_prots = line[1:]
            fuzzy_human_clusters[cluster_id] = cluster_prots

    gutMode = args.gut_mode
    fuzzy_bac_clusters = {}
    for m in gutMode:
        print("Fetch disease association proteins", m)
        # Gut Bacterial Proteins Pfam Directory --> Get all json files
        modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
        mode_clusters_file = os.path.join(modeDir, "fuzzy_{}_clusters.txt".format(m))

        with open(mode_clusters_file, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                cluster_id = line[0]
                cluster_prots = line[1:]
                fuzzy_bac_clusters[cluster_id] = cluster_prots

    print("Number of fuzzy human clusters:", len(fuzzy_human_clusters))
    print("Number of fuzzy bacterial clusters:", len(fuzzy_bac_clusters))

    # Make cluster-cluster interaction file
    cluster_cluster_interaction_file = os.path.join(resInfoDir, "Fuzzy_ClusterClusterInteractions.txt")

    perProtResDir = os.path.join(resInfoDir,args.per_protein_results)

    with open(cluster_cluster_interaction_file, "w") as fc:
        for hcidx, (hc_repr, hcl_members) in enumerate(fuzzy_human_clusters.items(),1):
            print(f"Cluster: {hcidx}/{len(fuzzy_human_clusters)}")
            human_cluster_interactions = {}
            for p in hcl_members:
                prot_inter_file = os.path.join(perProtResDir, f"{p}.json")
                if os.path.isfile(prot_inter_file):
                    with open(prot_inter_file, "r") as f:
                        data = json.load(f)
                        # keys: the bacterial proteins that interact with p
                        bac_int_per_hprot = {k for k,v in data.items() if v > 0.99}
                        f.close()
                    
                    if len(bac_int_per_hprot) > 0:
                        interacts_with = bac_int_per_hprot
                        for bcidx, bprots in fuzzy_bac_clusters.items():
                            bprots_set = set(bprots)
                            
                            inter = interacts_with.intersection(bprots_set)

                            if inter:
                                if not bcidx in human_cluster_interactions:
                                    human_cluster_interactions[bcidx] = len(inter)
                                else:
                                    human_cluster_interactions[bcidx] += len(inter)

            for bcidx, count in human_cluster_interactions.items():
                fc.write("{}\t{}\t{}\n".format(hc_repr, bcidx, count))

def extract_fuzzy_degree_info(resInfoDir):
    fuzzy_clusters_interaction_file = os.path.join(resInfoDir, "Fuzzy_ClusterClusterInteractions.txt")

    human_clusters = {}
    bacterial_clusters = {}
    with open(fuzzy_clusters_interaction_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            human_cluster = line[0]
            bacterial_cluster = line[1]

            if not human_cluster in human_clusters:
                human_clusters[human_cluster] = 1
            else:
                human_clusters[human_cluster] += 1
            
            if not bacterial_cluster in bacterial_clusters:
                bacterial_clusters[bacterial_cluster] = 1
            else:
                bacterial_clusters[bacterial_cluster] += 1

    # Human degree
    human_degree = pd.DataFrame.from_dict(human_clusters, orient='index', columns=["degree"])

    # Bacterial degree
    bacterial_degree = pd.DataFrame.from_dict(bacterial_clusters, orient='index', columns=["degree"])

    # Save to excel: Each DF a new sheet:
    with pd.ExcelWriter(os.path.join(resInfoDir,"Fuzzy_ClusterDegree.xlsx")) as writer:
        human_degree.to_excel(writer, sheet_name='HumanClusters')
        bacterial_degree.to_excel(writer, sheet_name='BacterialClusters')

def extract_IVI(resInfoDir):
    fuzzy_clusters_interaction_file = os.path.join(resInfoDir, "Fuzzy_ClusterClusterInteractions.txt")

    # Read in dataframe
    fuzzy_clusters_interaction = pd.read_csv(fuzzy_clusters_interaction_file, sep="\t", header=None)

    # Rename columns
    fuzzy_clusters_interactionsDF = fuzzy_clusters_interaction.rename(columns={0:"human_cluster", 1:"bacterial_cluster", 2:"count"})
    fuzzy_clusters_interactionsDF = fuzzy_clusters_interactionsDF[['human_cluster', 'bacterial_cluster']]

    # Graph
    g = ig.Graph.TupleList(fuzzy_clusters_interactionsDF.itertuples(index=False), directed=False)    
    iviDF = ivi(g, vertices = None, weights = None, directed = False, mode = "all", loops = True, d = 3, scale = 'range', verbose = False)
    # save to csv
    iviDF.to_csv(os.path.join(resInfoDir,"iviDF.csv"), index=False)

def extract_network_features(network_file):
    """
        Extract network features.
    """
    '''# Read network
    network = pd.read_csv(network_file, sep="\t", header=None)
    network = network.rename(columns={0:"Human_node", 1:"Bacterial_node"})
    # Number of interactions
    num_interactions = len(network)
    # Number of nodes
    nodes = set(network["Human_node"]).union(set(network["Bacterial_node"]))
    num_nodes = len(nodes)
    # Number of human proteins
    human_proteins = set(network["Human_node"])
    num_human_proteins = len(human_proteins)
    # Number of bacterial proteins
    bacterial_proteins = set(network["Bacterial_node"])
    num_bacterial_proteins = len(bacterial_proteins)
    # Save to dictionary
    network_info = {"Number of interactions":num_interactions, "Number of Nodes":num_nodes, "Number of Human Nodes":num_human_proteins, "Number of Bacterial Nodes":num_bacterial_proteins}
    return network_info'''

    # Read network as txt file row by row
    with open(network_file, "r") as f:
        num_interactions = 0
        num_nodes = set()
        num_human_proteins = set()
        num_bacterial_proteins = set()
        for line in f:
            line = line.strip().split("\t")
            num_interactions += 1
            num_nodes.add(line[0])
            num_nodes.add(line[1])
            num_human_proteins.add(line[0])
            num_bacterial_proteins.add(line[1])
        f.close()

    # Save to dictionary
    network_info = {"Number of interactions":num_interactions, "Number of Nodes":len(num_nodes), "Number of Human Nodes":len(num_human_proteins), "Number of Bacterial Nodes":len(num_bacterial_proteins)}
    return network_info

def extend_uniref_cluster_info(cluster_id):
    # Read cluster db clusterid.json
    cluster_file = os.path.join(args.project_path, args.cluster_db, cluster_id + ".json")
    with open(cluster_file, "r") as f:
        cluster_info = json.load(f)
        uniref_m = cluster_info["to"]["members"]
        f.close()
    # Open cluster id in protein db
    repr_file = os.path.join(args.project_path, args.protein_db, cluster_id + ".json")
    with open(repr_file, "r") as f:
        repr_info = json.load(f)
        if not "recommendedName" in repr_info["to"]["proteinDescription"]:
            cluster_name = repr_info["to"]["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        else:
            cluster_name = repr_info["to"]["proteinDescription"]["recommendedName"]["fullName"]["value"]
        org = repr_info["to"]["organism"]["scientificName"]
        f.close()
    # Open every member in protein db
    members_org = []
    for m in uniref_m:
        m_file = os.path.join(args.project_path, args.protein_db, m + ".json")
        with open(m_file, "r") as f:
            m_info = json.load(f)
            m_org = m_info["to"]["organism"]["scientificName"]
            members_org.append(m_org)
            f.close()
    return cluster_name, org, members_org

def extend_fuzzy_cluster_info(cluster_id,clusters_info):
    # Read cluster db clusterid.json
    
    cl_members = clusters_info[cluster_id]
    # Open cluster id in protein db
    repr_file = os.path.join(args.project_path, args.protein_db, cluster_id.split("_")[1] + ".json")
    with open(repr_file, "r") as f:
        repr_info = json.load(f)
        if not "recommendedName" in repr_info["to"]["proteinDescription"]:
            cluster_name = repr_info["to"]["proteinDescription"]["submissionNames"][0]["fullName"]["value"]
        else:
            cluster_name = repr_info["to"]["proteinDescription"]["recommendedName"]["fullName"]["value"]
        org = repr_info["to"]["organism"]["scientificName"]
        f.close()
    # Open every member in protein db
    members_org = []
    for m in cl_members:
        m_file = os.path.join(args.project_path, args.protein_db, m + ".json")
        with open(m_file, "r") as f:
            m_info = json.load(f)
            m_org = m_info["to"]["organism"]["scientificName"]
            members_org.append(m_org)
            f.close()
    return cluster_name, org, members_org

def export_results(resInfoDir,modeStr):
    """
        Aggregate results.
        Sheet 1: Number of interactions, Number of Nodes, Number of Human Proteins, Number of Bacterial Proteins: 
        a. Full Network, b. Uniref, c. Fuzzy
        Sheet 2: Full Network--> Human protein degree (sorted)
        Sheet 3: Full Network--> Bacterial protein degree (sorted)
        Sheet 4: Uniref90 Network--> Human protein degree (sorted)
        Sheet 5: Uniref90 Network--> Bacterial protein degree (sorted)
        Sheet 6: Fuzzy Network--> Clusters degree, IVI Human(IVI-sorted)
        Sheet 7: Fuzzy Network--> Clusters degree, IVI Bacterial(IVI-sorted)
    """

    ## Sheet 1: Full Network
    networks_info = {}
    predPPIs = os.path.join(resInfoDir,args.predicted_ppis.split(".")[0] + "_" + modeStr + "." + args.predicted_ppis.split(".")[1])
    full_net = extract_network_features(predPPIs)
    networks_info["Full Network"] = full_net
    ## Sheet 1b: Uniref
    clusterClusterFile = os.path.join(resInfoDir,"Uniref90_ClusterClusterInteractions.txt")
    uniref_net = extract_network_features(clusterClusterFile)
    networks_info["Uniref Network"] = uniref_net
    ## Sheet 1c: Fuzzy
    fuzzy_cluster_cluster_file = os.path.join(resInfoDir,"Fuzzy_ClusterClusterInteractions.txt")
    fuzzy_net = extract_network_features(fuzzy_cluster_cluster_file)
    networks_info["Fuzzy Network"] = fuzzy_net
    # Make dataframe
    networks_infoDF = pd.DataFrame.from_dict(networks_info, orient='index')
    print(networks_infoDF)
    
    ## Sheet 2: Full Network--> Human protein degree (sorted)
    protein_degree_file = os.path.join(resInfoDir,"protDegree.xlsx")
    human_degree = pd.read_excel(protein_degree_file, sheet_name="HumanProteins")
    human_degree = human_degree.rename(columns={"Unnamed: 0":"HumanProtein", "degree":"Degree"})
    # Sort by Degree
    human_degree = human_degree.sort_values(by=['Degree'])

    ## Sheet 3: Full Network--> Bacterial protein degree (sorted)
    bacterial_degree = pd.read_excel(protein_degree_file, sheet_name="BacterialProteins")
    bacterial_degree = bacterial_degree.rename(columns={"Unnamed: 0":"BacterialProtein", "degree":"Degree"})
    # Sort by Degree
    bacterial_degree = bacterial_degree.sort_values(by=['Degree'])

    ## Sheet 4: Uniref90 Network--> Human protein degree (sorted)
    uniref90_degree_file = os.path.join(resInfoDir,"Uniref90_ClusterDegree.xlsx")
    uniref90_human_degree = pd.read_excel(uniref90_degree_file, sheet_name="HumanClusters")
    uniref90_human_degree = uniref90_human_degree.rename(columns={"Unnamed: 0":"HumanCluster", "degree":"Degree"})
    # Map cluster id to cluster name, org, members org
    uniref90_human_degree["ClusterName"], uniref90_human_degree["Organism"], uniref90_human_degree["MembersOrg"] = zip(*uniref90_human_degree["HumanCluster"].map(extend_uniref_cluster_info))
    uniref90_human_degree = uniref90_human_degree.sort_values(by=['Degree'])

    ## Sheet 5: Uniref90 Network--> Bacterial protein degree (sorted)
    uniref90_bacterial_degree = pd.read_excel(uniref90_degree_file, sheet_name="BacterialClusters")
    uniref90_bacterial_degree = uniref90_bacterial_degree.rename(columns={"Unnamed: 0":"BacterialCluster", "degree":"Degree"})
    # Map cluster id to cluster name, org, members org
    uniref90_bacterial_degree["ClusterName"], uniref90_bacterial_degree["Organism"], uniref90_bacterial_degree["MembersOrg"] = zip(*uniref90_bacterial_degree["BacterialCluster"].map(extend_uniref_cluster_info))
    uniref90_bacterial_degree = uniref90_bacterial_degree.sort_values(by=['Degree'])
    
    # Load fuzzy clusters
    fuzzy_human_clusters_file = os.path.join(args.project_path, args.gut_data_dir, "fuzzy_human_clusters.txt") 
    
    # Read clusters
    fuzzy_human_clusters = {}
    with open(fuzzy_human_clusters_file, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            cluster_id = line[0]
            cluster_prots = line[1:]
            fuzzy_human_clusters[cluster_id] = cluster_prots

    gutMode = args.gut_mode
    fuzzy_bac_clusters = {}
    for m in gutMode:
        print("Fetch disease association proteins", m)
        # Gut Bacterial Proteins Pfam Directory --> Get all json files
        modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
        mode_clusters_file = os.path.join(modeDir, "fuzzy_{}_clusters.txt".format(m))

        with open(mode_clusters_file, "r") as f:
            for line in f:
                line = line.strip().split("\t")
                cluster_id = line[0]
                cluster_prots = line[1:]
                fuzzy_bac_clusters[cluster_id] = cluster_prots

    ## Sheet 6: Fuzzy Network--> Clusters degree, IVI Human(IVI-sorted)
    fuzzy_degree_file = os.path.join(resInfoDir,"Fuzzy_ClusterDegree.xlsx")
    fuzzy_human_degree = pd.read_excel(fuzzy_degree_file, sheet_name="HumanClusters")
    fuzzy_human_degree = fuzzy_human_degree.rename(columns={"Unnamed: 0":"HumanCluster", "degree":"Degree"})
    ivi_file = os.path.join(resInfoDir,"iviDF.csv")
    iviDF = pd.read_csv(ivi_file)
    # See if node_name is in fuzzy_human_degree
    iviDF_h = pd.merge(iviDF, fuzzy_human_degree, left_on="Node_name", right_on="HumanCluster", how="inner")
    iviDF_h = iviDF_h.sort_values(by=['IVI'], ascending=False)
    iviDF_h = iviDF_h[["HumanCluster","IVI","Degree"]]
    # Map cluster id to cluster name, org, members org
    iviDF_h["ClusterName"], iviDF_h["Organism"], iviDF_h["MembersOrg"] = zip(*iviDF_h["HumanCluster"].map(lambda x: extend_fuzzy_cluster_info(x,fuzzy_human_clusters)))
    ## Sheet 7: Fuzzy Network--> Clusters degree, IVI Bacterial(IVI-sorted)
    fuzzy_bacterial_degree = pd.read_excel(fuzzy_degree_file, sheet_name="BacterialClusters")
    fuzzy_bacterial_degree = fuzzy_bacterial_degree.rename(columns={"Unnamed: 0":"BacterialCluster", "degree":"Degree"})
    # See if node_name is in fuzzy_bacterial_degree
    iviDF_b = pd.merge(iviDF, fuzzy_bacterial_degree, left_on="Node_name", right_on="BacterialCluster", how="inner")
    iviDF_b = iviDF_b.sort_values(by=['IVI'], ascending=False)
    iviDF_b = iviDF_b[["BacterialCluster","IVI","Degree"]]
    # Map cluster id to cluster name, org, members org
    iviDF_b["ClusterName"], iviDF_b["Organism"], iviDF_b["MembersOrg"] = zip(*iviDF_b["BacterialCluster"].map(lambda x: extend_fuzzy_cluster_info(x,fuzzy_bac_clusters)))

    print(iviDF_h)
    print(iviDF_b)

    # Save to excel
    with pd.ExcelWriter(os.path.join(resInfoDir,"Aggregated_results_{}.xlsx".format(modeStr))) as writer:
        networks_infoDF.to_excel(writer, sheet_name='NetworksInfo')
        human_degree.to_excel(writer, sheet_name='HumanProteins')
        bacterial_degree.to_excel(writer, sheet_name='BacterialProteins')
        uniref90_human_degree.to_excel(writer, sheet_name='Uniref90HumanClusters')
        uniref90_bacterial_degree.to_excel(writer, sheet_name='Uniref90BacterialClusters')
        iviDF_h.to_excel(writer, sheet_name='FuzzyHumanClusters')
        iviDF_b.to_excel(writer, sheet_name='FuzzyBacterialClusters')
        
def run():
    gutMode = args.gut_mode
    modeStr = "_".join(gutMode)
    # Print mode:
    print("Gut mode: {}".format(gutMode))
    predDir = os.path.join(args.project_path, args.prediction_dir)
    # Make dir Results/Prediction/Mode
    resInfoDir = os.path.join(predDir, modeStr)
    # Predicted_PPIs_Mode.txt
    predPPIs = os.path.join(resInfoDir,args.predicted_ppis.split(".")[0] + "_" + modeStr + "." + args.predicted_ppis.split(".")[1])

    gut_bac_prot, gb_hum_prot = get_protein_pools()

    if not os.path.isfile(predPPIs):
        os.makedirs(predDir, exist_ok=True)
        os.makedirs(resInfoDir, exist_ok=True)
        
        human_group_file = os.path.join(resInfoDir, "human_group.txt")
        gut_bac_group_file = os.path.join(resInfoDir, "gut_bac_group.txt")

        if not os.path.isfile(human_group_file):
            # Read human protein domains
            with open(args.human_protein_domains, "r") as f:
                hum_prot_domains = json.load(f)
                f.close()

            # Keep only gut brain proteins
            gb_hum_prot = {k:v for k,v in hum_prot_domains.items() if k in gb_hum_prot}
            group_proteins(gb_hum_prot, human_group_file)

        if not os.path.isfile(gut_bac_group_file):
            gut_bac_domains = {}
            for m in gutMode:
                print("Fetch disease association proteins", m)
                # Gut Bacterial Proteins Pfam Directory --> Get all json files
                modeDir = os.path.join(args.project_path, args.gut_data_dir, m)
                proteinsFile = os.path.join(modeDir, "{}_proteins_domains.json".format(m))

                with open(proteinsFile, 'r') as f:
                    modeGutProts = json.load(f)

                # Append to gut_bac_prot
                for k,v in modeGutProts.items():
                    gut_bac_domains[k] = v

            group_proteins(gut_bac_domains, gut_bac_group_file)
        
        # Predict
        extract_interactions(human_group_file,gut_bac_group_file,resInfoDir)
        aggregate_results(resInfoDir,gb_hum_prot,predPPIs)
    
    interaction_info_file = os.path.join(resInfoDir,"protDegree.xlsx")
    if not os.path.isfile(interaction_info_file):
        extract_degree_info(resInfoDir,predPPIs)

    if not os.path.isfile(os.path.join(resInfoDir,"pickla_comparison.svg")):
        pickla_comparison(resInfoDir)

    bacterial_info_file = os.path.join(resInfoDir,"Bacteria_info_{}.csv".format(modeStr))
    if not os.path.isfile(bacterial_info_file):
        extractBacterialInfo(args,resInfoDir,gutMode)

    uniref90_cluster_cluster_file = os.path.join(resInfoDir,"Uniref90_ClusterClusterInteractions.txt")
    if not os.path.isfile(uniref90_cluster_cluster_file):
        UniRef90ClusterInteractions(resInfoDir,gutMode)
    if not os.path.isfile(os.path.join(resInfoDir,"Uniref90_ClusterDegree.xlsx")):
        extract_uniref90_degree_info(resInfoDir)
    fuzzy_cluster_cluster_file = os.path.join(resInfoDir,"Fuzzy_ClusterClusterInteractions.txt")

    if not os.path.isfile(fuzzy_cluster_cluster_file):
        fuzzy_clusters_interactions(resInfoDir)

    fuzzy_degree_info_file = os.path.join(resInfoDir,"Fuzzy_ClusterDegree.xlsx")
    if not os.path.isfile(fuzzy_degree_info_file):
        extract_fuzzy_degree_info(resInfoDir)
    
    if not os.path.isfile(os.path.join(resInfoDir,"iviDF.csv")):
        extract_IVI(resInfoDir)

    if not os.path.isfile(os.path.join(resInfoDir,"Aggregated_results_{}.xlsx".format(modeStr))):
        export_results(resInfoDir,modeStr)

def list_of_strings(arg):
    return arg.split(',')

def argParse():
    """
        Parse script arguments.
    """
    parser = argparse.ArgumentParser(description='Domain - Domain interaction prediction.')
    # Project path
    parser.add_argument('-pp', '--project_path', type=str, help='Project path.', default="/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA")
    # Gut data directory
    parser.add_argument('-gut', '--gut_data_dir', type=str, help='Gut data directory.', default="data/Gut Data")
    parser.add_argument('-pikla', '--pikla_path', type=str, help='Pikla path.', default="data/Dataset/Negative Dataset/UniProtNormalizedTabular-default.txt")
    # Data during execution
    # Protein Database
    parser.add_argument('-pdb', '--protein_db', type=str, help='Protein Database.', default="data/ProteinDatabase")
    # Cluster Database 
    parser.add_argument('-clb', '--cluster_db', type=str, help='Cluster Database.', default="data/ClusterDatabase")
    # Prediction directory
    parser.add_argument('-pred', '--prediction_dir', type=str, help='Prediction directory.', default="Results/Prediction")
    # Predicted PPIs
    parser.add_argument('-ppis', '--predicted_ppis', type=str, help='Predicted PPIs.', default="Predicted_PPIs.txt")
    # Per protein results
    parser.add_argument('-ppr', '--per_protein_results', type=str, help='Per protein results.', default="PerProteinResults")
    #
    parser.add_argument('-gut_prot', '--gut_proteoms_dir', type=str, help='Gut Bac Proteins Directory.', default="BacProteins")
    #
    parser.add_argument('-hpd', '--human_protein_domains', type=str, help='Human Protein Domains.', default="data/human_proteins_domains.json")
    # DDI instances
    parser.add_argument('-ddi_inst', '--ddi_inst', type=str, help='DDI instances.', default="ddi_freq.txt")
    # Pfam2vec model
    parser.add_argument('-pfam2vec', '--pfam2vec', type=str, help='Pfam2vec model.', default="Results/ml_method/pfam2vec.model")
    # Final model
    parser.add_argument('-fm', '--final_model', type=str, help='Final model.', default="Results/final_model.pkl")
    # Prot_domains
    parser.add_argument('-prot_dom', '--prot_domains', type=str, help='prot_domains.json.', default="data/Dataset/prot_domains.json")
    # Requirements
    
    # Gut data mode: Disease association
    parser.add_argument('-gm', '--gut_mode', type=list_of_strings, help='Gut data mode: Disease association', required=True)
                        #choices=['Colon adenoma', 'Ankylosing spondylitis', 'Melanoma', 'Colorectal cancer', 'Type 1 diabetes', 'Atherosclerosis', "Crohn's disease", 'Impaired glucose tolerance', 'Type 2 diabetes', 'NAFLD', 'Vogt-Koyanagi-Harada', "Behcet's disease", 'Healthy', 'Liver cirrhosis', 'Cardiovascular disease', 'Acute diarrhea', 'Renal cancer', 'Lung cancer', "Parkinson's disease", 'Chronic fatigue syndrome', 'Ulcerative colitis', 'Unspecified'])

    return parser.parse_args()

if __name__ == "__main__":
    args = argParse()
    run()