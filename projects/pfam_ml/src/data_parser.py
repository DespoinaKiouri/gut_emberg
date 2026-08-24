import os
import shutil
import time
import argparse
import random
import requests
import pickle
import json
import numpy as np
import pandas as pd
import itertools
from utils.uniprot_mapping import uniprot_to_uniprot
from thefuzz import fuzz

class UniprotMapping:
    """
        Class to map UniprotKB IDs to UniprotKB IDs: Download proteins, clean and store results.
    """
    def __init__(self,protein_list,prot_info_path):
        self.protein_list = protein_list
        self.prot_info_path = prot_info_path
        self.temp_db_path = os.path.join(args.project_path, "temp_db")
        self.protein_db_path = os.path.join(args.project_path, args.protein_db)
        self.uniprot_mapping()
        self.clean_and_store()

        # Remove temp_db directory
        if os.path.exists(self.temp_db_path):
            shutil.rmtree(self.temp_db_path)

    def uniprot_mapping(self):
        """
            Map UniprotKB IDs to UniprotKB IDs.
        """
        # Create temp database directory
        if not os.path.exists(self.temp_db_path):
            os.makedirs(self.temp_db_path)

        batch_size = 90000
        batch_num = len(self.protein_list) // batch_size
        # Add last batch if not divisible
        if len(self.protein_list) % batch_size != 0:
            batch_num += 1

        print("Batch num: {}".format(batch_num))
        for i in range(batch_num):
            print("Batch: {}".format(i))
            # Get batch: Beggining: i*batch_size (first=0), End: (i+1)*batch_size (first=90.000)
            batch = self.protein_list[i*batch_size:(i+1)*batch_size]
            batch_result = uniprot_to_uniprot(batch)
            print("Writing batch {} to temp database".format(i))
            self.write_to_temp_db(batch_result)

    def write_to_temp_db(self,results):
        """
            Write results to temp database.
        """
        resultIdx = len(os.listdir(self.temp_db_path)) + 1
        for result in results:
            # if result is empty, continue:
            if len(result) == 0:
                print("Empty result: from {} - to {}".format(result["from"], result["to"]["primaryAccession"]))
                continue
        
            protein_file = os.path.join(self.temp_db_path, "{}.json".format(resultIdx))
        
            # Write only proteins with Pfam domains
            if "uniProtKBCrossReferences" in result["to"]:
                domains = list()
                for cr in result["to"]["uniProtKBCrossReferences"]:
                    if cr["database"] == "Pfam":
                        domains.append(cr["id"])

                if len(domains) > 0:
                    # IssueB: Remove References from ID mapping results (huge files)
                    if "References" in result["to"]:
                        del result["to"]["References"]
                    # Write protein file
                    with open(protein_file, "w") as f:
                        json.dump(result, f)
                        resultIdx += 1
                        f.close()
                else:
                    print("No Pfam domains: from {} - to {}".format(result["from"], result["to"]["primaryAccession"]))

    def clean_and_store(self):
        """
            Clean and store results in json files.
        """
        if not os.path.exists(self.protein_db_path):
            os.makedirs(self.protein_db_path)

        protein_indexing = {}

        # For every file in temp_db_path
        for file in os.listdir(self.temp_db_path):
            # Read file
            try:
                with open(os.path.join(self.temp_db_path, file), "r") as f:
                    prot_info = json.load(f)
                    if prot_info["from"] == prot_info["to"]["primaryAccession"]:
                        if prot_info["from"] in protein_indexing:
                            continue
                        else:
                            if not os.path.isfile(os.path.join(self.protein_db_path, prot_info["from"] + ".json")):
                                # Move file in protein_db_path and rename to from.json
                                os.rename(os.path.join(self.temp_db_path, file), os.path.join(self.protein_db_path, prot_info["from"] + ".json"))
                            protein_indexing[prot_info["from"]] = prot_info["from"] + ".json"
                    else:
                        print("Redundant file: {}".format(file))
                    f.close()
            except:
                print("Error reading file: {}".format(file))
        
        # Write protein indexing
        with open(self.prot_info_path, "w") as f:
            json.dump(protein_indexing, f)
            f.close()

class UnirefMapping:
    """
        Class to map UniprotKB IDs to UniRef90 Clusters: Download protein clusters, clean and store results.
    """
    def __init__(self,protein_list,cluster_info_path,prot_domains):
        self.protein_list = protein_list
        self.cluster_info_path = cluster_info_path
        self.temp_cl_db_path = os.path.join(args.project_path, "temp_cl_db")
        self.cluster_db_path = os.path.join(args.project_path, args.cluster_db)
        self.prot_domains = prot_domains
        self.uniprot_mapping()
        self.clean_and_store()

        # Remove temp_db directory
        if os.path.exists(self.temp_cl_db_path):
            shutil.rmtree(self.temp_cl_db_path)

        
    def uniprot_mapping(self):
        """
            Map UniprotKB IDs to UniprotKB IDs.
        """
        # Create temp database directory
        if not os.path.exists(self.temp_cl_db_path):
            os.makedirs(self.temp_cl_db_path)

        batch_size = 90000
        batch_num = len(self.protein_list) // batch_size
        # Add last batch if not divisible
        if len(self.protein_list) % batch_size != 0:
            batch_num += 1

        print("Batch num: {}".format(batch_num))
        for i in range(batch_num):
            print("Batch: {}".format(i))
            # Get batch: Beggining: i*batch_size (first=0), End: (i+1)*batch_size (first=90.000)
            batch = self.protein_list[i*batch_size:(i+1)*batch_size]
            batch_result = uniprot_to_uniprot(batch,from_db="UniProtKB_AC-ID", to_db="UniRef90")
            print("Writing batch {} to temp database".format(i))
            self.write_to_temp_cl_db(batch_result)

    def write_to_temp_cl_db(self,results):
        """
            Write results to temp database.
        """
        resultIdx = len(os.listdir(self.temp_cl_db_path)) + 1

        for result in results:
            # if result is empty, continue:
            if len(result) == 0:
                print("Empty result: ", result)
                continue
            cluster_file = os.path.join(self.temp_cl_db_path, "{}.json".format(resultIdx))
            with open(cluster_file, "w") as f:
                json.dump(result, f)
                resultIdx += 1
                f.close()

    def clean_and_store(self):
        """
            Clean and store results in json files.
        """
        if not os.path.exists(self.cluster_db_path):
            os.makedirs(self.cluster_db_path)

        cluster_file_indexing = {}
        
        # For every file in temp_db_path
        for file in os.listdir(self.temp_cl_db_path):
            # Read file
            with open(os.path.join(self.temp_cl_db_path, file), "r") as f:
                cluster_info = json.load(f)
            
                # Get cluster ID
                cluster_id = cluster_info["to"]["id"]
                if not cluster_id in cluster_file_indexing:
                    cluster_file_indexing[cluster_id] = file
                else:
                    print("Redundant file: {}".format(file))

                f.close()

        print("Clusters downloaded: ",len(os.listdir(self.temp_cl_db_path)))
        print("Clusters to keep: ",len(cluster_file_indexing))

        cluster_info_file = {}

        for cluster_id, file in cluster_file_indexing.items():
            with open(os.path.join(self.temp_cl_db_path, file), "r") as f:
                cluster_info = json.load(f)

                # Get cluster members
                members = cluster_info["to"]["members"]
                # Keep members that are in protList
                members = [member.split(",")[0] for member in members if member.split(",")[0] in self.protein_list]
                # Update cluster
                cluster_info["to"]["members"] = members

                prot_id = cluster_id.split("_")[1]

                if not prot_id in self.protein_list:
                # Get cluster members
                    members = cluster_info["to"]["members"]

                    if len(members) > 0:
                        # Get domain lengths
                        dom_len_list = list()
                        for m in members:
                            member_domain = self.prot_domains[m]
                            dom_len_list.append(len(member_domain))

                        # Get max domain length
                        max_dom_len = max(dom_len_list)

                        # Index of max domain length
                        max_dom_len_index = dom_len_list.index(max_dom_len)

                        prot_id = members[max_dom_len_index]

                        #print("Changing cluster ID: {} to {}".format(cluster_id, prot_id))
                    else:
                        prot_id = None

                
                if prot_id is None:
                    continue

                # Change cluster ID to prot_id
                cluster_info["to"]["id"] = prot_id

                # Write cluster file
                with open(os.path.join(self.cluster_db_path, prot_id + ".json"), "w") as f:
                    json.dump(cluster_info, f)
                    f.close()

                # Add cluster info to cluster_info_file
                cluster_info_file[prot_id] = prot_id + ".json"

        # Write cluster indexing
        with open(self.cluster_info_path, "w") as f:
            json.dump(cluster_info_file, f)
            f.close()

class ExperimentalProteins:
    """
        Class to store the proteins of the experimental dataset.
    """
    def __init__(self):
        self.exp_data_path = os.path.join(args.project_path, args.exp_data)
        self.ppidm_data_path = os.path.join(args.project_path, args.ppidm_data)

        # File with the dataset ppis: Combination of experimental dataset and PPIDM dataset
        self.dataset_ppis_path = os.path.join(args.project_path, args.dataset_ppis)
        
        # File with the experimental proteins used for database indexing
        self.exp_prots_path = os.path.join(args.project_path, args.exp_prots)

        # File with the experimental proteins domains
        self.exp_prot_domains_path = os.path.join(args.project_path, args.exp_prots_domains)

        if not (os.path.isfile(self.dataset_ppis_path) and os.path.isfile(self.exp_prots_path) and os.path.isfile(self.exp_prot_domains_path)):
            expDF = self.experimental_data_combo()
            self.get_proteins(expDF)
            self.get_proteins_domains()
            self.clean_PPIs(expDF)
        else:
            print("Experimental proteins and domains already stored:")

            # Load experimental proteins
            with open(self.exp_prots_path, "r") as f:
                exp_prots = json.load(f)
                f.close()
            print("Experimental proteins: {}".format(len(exp_prots)))

            # Load PPIs
            expDF = pd.read_csv(self.dataset_ppis_path, sep="\t", header=None)
            print("Experimental PPIS: {}".format(expDF.shape[0]))

    def experimental_data_combo(self):
        """
            Combine experimental and PPIdomainminer data to create a unique PPI dataset.
        """
        expDF = pd.read_csv(self.exp_data_path, sep="\t", header=0)
        expDF.columns = ["P1","P2"]
        ppimDF = pd.read_csv(self.ppidm_data_path, sep="\t", header=None)
        ppimDF.columns = ["P1","P2"]

        # Concatenate dataframes
        expDF = pd.concat([expDF, ppimDF], ignore_index=True)

        # Sort PPIs and add to dictionary
        expDict = {}
        idx = 0
        for k, v in expDF.T.to_dict().items():
            pair = tuple(sorted([v["P1"],v["P2"]]))
            expDict[idx] = {"P1":pair[0], "P2":pair[1]}
            idx += 1

        # Convert dict to dataframe
        expDF = pd.DataFrame.from_dict(expDict).T

        # Remove duplicates
        expDF = expDF.groupby(["P1","P2"]).size().reset_index(name="Count")
        expDF = expDF[['P1','P2']]

        print("Combined PPIS: Experimental + PPIdomainminer: {}".format(expDF.shape[0]))
        return expDF

    def get_proteins(self,expDF):
        """
            Export unique proteins from expDF Search this list in Uniprot ID mapping (API call).
        """
        # Get unique proteins
        expProts = sorted(list(set(expDF["P1"].unique()).union(set(expDF["P2"].unique()))))
        UniprotMapping(expProts,self.exp_prots_path)

    def get_proteins_domains(self):
        """
            Get protein domains from protein database.
            Generates a json file with uniprot id as keys and a list of domains as the corresponging values.
        """
        # Read protein indexing file
        with open(self.exp_prots_path, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Get protein domains
        prot_domains = {}
        for prot, file in prot_indexing.items():
            protFile = os.path.join(args.project_path, args.protein_db, file)
            with open(protFile, "r") as f:
                protInfo = json.load(f)
                domains = list()
                for cr in protInfo["to"]["uniProtKBCrossReferences"]:
                    if cr["database"] == "Pfam":
                        domains.append(cr["id"])

                if len(domains) > 0:
                    prot_domains[prot] = domains

                f.close()
        
        # Write protein domains
        with open(self.exp_prot_domains_path, "w") as f:
            json.dump(prot_domains, f)
            f.close()

    def clean_PPIs(self,expDF):
        """
            Clean PPIs to remove proteins without information.
        """

        # Read protein indexing file
        with open(self.exp_prots_path, "r") as f:
            protein_info = json.load(f)
            f.close()

        # Keep only the proteins that have domain information from initial dataset (expDF)
        expDF = expDF[expDF["P1"].isin(protein_info.keys())]
        expDF = expDF[expDF["P2"].isin(protein_info.keys())]

        print("Cleaned Experimental PPIS: {}".format(expDF.shape[0]))

        # Write to file
        expDF.to_csv(self.dataset_ppis_path, sep="\t", index=False, header=False)

class HumanProteins:
    """
        Class to store the human proteins.
    """
    def __init__(self):
        self.proteomes_path = os.path.join(args.project_path, args.human_proteome)
        self.hum_prots_path = os.path.join(args.project_path, args.human_protein_info)
        if not (self.hum_prots_path and os.path.isfile(os.path.join(args.project_path,args.human_protein_domains))):
            human_proteins = self.readHumanProteome()
            self.get_proteins(human_proteins)
            self.get_proteins_domains()
        else:
            print("Human proteins and domains already stored:")

            # Load human proteins
            with open(self.hum_prots_path, "r") as f:
                hum_prots = json.load(f)
                f.close()
            print("Human proteins: {}".format(len(hum_prots)))
    
    def readHumanProteome(self):
        """
            Read human proteome and store UniprotIDs in list.
        """
        self.human_proteins = pd.read_excel(self.proteomes_path)
        self.human_proteins = self.human_proteins["Entry"].tolist()
        return self.human_proteins

    def get_proteins(self,human_proteins):
        """
            Export unique proteins from expDF Search this list in Uniprot ID mapping (API call).
        """
        UniprotMapping(human_proteins,self.hum_prots_path)

    def get_proteins_domains(self):
        """
            Get protein domains from protein database.
            Generates a json file with uniprot id as keys and a list of domains as the corresponging values.
        """
        # Read protein indexing file
        with open(self.hum_prots_path, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Get protein domains
        prot_domains = {}
        for prot, file in prot_indexing.items():
            protFile = os.path.join(args.project_path, args.protein_db, file)
            with open(protFile, "r") as f:
                protInfo = json.load(f)
                domains = list()
                for cr in protInfo["to"]["uniProtKBCrossReferences"]:
                    if cr["database"] == "Pfam":
                        domains.append(cr["id"])

                if len(domains) > 0:
                    prot_domains[prot] = domains

                f.close()
        
        # Write protein domains
        with open(os.path.join(args.project_path,args.human_protein_domains), "w") as f:
            json.dump(prot_domains, f)
            f.close()

class DatasetParser():
    """
    Class to parse the dataset.
    """
    def __init__(self):
        if not (os.path.isfile(os.path.join(args.project_path, args.dataset, "train.csv"))
            and os.path.isfile(os.path.join(args.project_path, args.dataset, "val.csv"))
            and os.path.isfile(os.path.join(args.project_path, args.dataset, "test.csv"))
            and os.path.isfile(os.path.join(args.project_path, args.dataset_prot_domains))):

            self.prot_db_path = os.path.join(args.project_path, args.protein_db)
            self.exp_prots_path = os.path.join(args.project_path, args.exp_prots)
            
            self.exp_prots_path = os.path.join(args.project_path, args.exp_prots)
            self.datasetPPIs = os.path.join(args.project_path, args.dataset_ppis)
            self.pikla_path = os.path.join(args.project_path, args.pikla_path)
            self.normal_tissue_path = os.path.join(args.project_path, args.normal_tissue_path)
            self.ppis = pd.read_csv(self.datasetPPIs, sep="\t", header=None)
            self.ppis.columns = ["P1","P2"]

            print("Number of PPIs", len(self.ppis))

            self.negatives = self.parse_negatives()

            print("Number of all unique proteins (Experimental and Negative datasets)", len(self.prot_domains))
            self.dataset_split()

            with open(os.path.join(args.project_path, args.dataset_prot_domains), 'w') as fp:
                json.dump(self.prot_domains, fp, indent=4)
        else:
            # Report
            with open(os.path.join(args.project_path, args.dataset_prot_domains), 'r') as fp:
                self.prot_domains = json.load(fp)
                print("Number of all unique proteins (Experimental and Negative datasets)", len(self.prot_domains))
                fp.close()

            # Train, Validation, Test report
            train = pd.read_csv(os.path.join(args.project_path, args.dataset, "train.csv"))
            val = pd.read_csv(os.path.join(args.project_path, args.dataset, "val.csv"))
            test = pd.read_csv(os.path.join(args.project_path, args.dataset, "test.csv"))

            print("Number of train PPIs: ", len(train[train["label"] == 1]))
            print("Number of train negative samples: ", len(train[train["label"] == 0]))
            print("Total training samples: ", len(train))
            print("Number of validation PPIs: ", len(val[val["label"] == 1]))
            print("Number of validation negative samples: ", len(val[val["label"] == 0]))
            print("Total validation samples: ", len(val))
            print("Number of test PPIs: ", len(test[test["label"] == 1]))
            print("Number of test negative samples: ", len(test[test["label"] == 0]))
            print("Total test samples: ", len(test))

    def get_prot_organ(self, prot_organ_file):
        """
            Using normal tissue data, map proteins to organs.
        """
        
        df1 =pd.read_csv(os.path.join(args.project_path, args.normal_tissue_path))
        df1.columns=["Gene", "Gene name", "Tissue", "Cell type", "Level", "Reliability"]

        # Filtering Level and Reliability
        df1 = df1[df1.Level.isin(["Low", "Medium", "High", "Ascending", "Descending"])]
        df1 = df1[df1.Reliability.isin(["Approved", "Enhanced", "Supported"])]
        
        # Filtering unique tissues
        df1 = df1.groupby(["Gene name","Tissue"]).size().reset_index()
        df1 = df1[["Gene name","Tissue"]]

        organs_dict = {"Heart": "heart muscle", "Adrenal Glands": "adrenal gland", "Appendix": "appendix", "Bone": "bone marrow",
            "Breast": ["breast", "lactating breast"], "Lungs": ["bronchus", "lung"], "Brain": ["caudate", "cerebellum", "cerebral cortex", "hippocampus", "hypothalamus", "retina", "choroid plexus", "dorsal raphe", "substantia nigra"], "Cervix": "cervix",
            "Large Intestine": ["colon", "rectum"], "Small Intestine": ["duodenum", "small intestine"],  "Uterus": ["endometrium 1", "endometrium 2", "endometrium"],
            "Testis": ["epididymis", "testis"], "Esophagus": "esophagus", "Fallopian Tubes": "fallopian tube", "Gallbladder": "gallbladder",
            "Kidneys": "kidney", "Liver": "liver", "Lymphatic System": ["lymph node", "spleen", "thymus", "tonsil"], "Nasopharynx": "nasopharynx",
            "Mouth": "oral mucosa",  "Ovaries": "ovary",  "Pancreas": "pancreas",  "Parathyroid": "parathyroid gland",  "Placenta": "placenta",
            "Prostate": "prostate", "Salivary Glands": "salivary gland", "Seminal Vesicles": "seminal vesicle", "Muscles": ["skeletal muscle", "smooth muscle", "soft tissue 1", "soft tissue 2"],
            "Skin": ["skin 1", "skin 2", "skin", "sole of foot"], "Stomach": ["stomach 1", "stomach 2"], "Thyroid": "thyroid gland",
            "Bladder": "urinary bladder", "Vagina": "vagina", "Hair": "hair",  "Eyes": ["eye", "retina"],  "Pituitary": "pituitary gland", "Cartilage": "cartilage"}

        # Map tissues to organs according to organs_dict (key:organ, value:tissues):
        organ_tissues_dict = {}
        for organ, tissues in organs_dict.items():
            if isinstance(tissues, list):
                for tissue in tissues:
                    organ_tissues_dict[tissue] = organ
            else:
                organ_tissues_dict[tissues] = organ
        
        # Map organs to tissues according to organ_tissues_dict (key:tissue, value:organ):
        df1["Organ"] = df1["Tissue"].map(organ_tissues_dict)
        
        df1 = df1.groupby(["Gene name","Organ"]).size().reset_index()
        df1 = df1[["Gene name","Organ"]]
        df1 = df1.dropna(subset=["Organ"])
        
         # Read protein db, brain and gut proteins
        human_proteins_path = os.path.join(args.project_path,args.human_protein_info)
        protein_db_path = os.path.join(args.project_path, args.protein_db)
        with open(human_proteins_path, "r") as f:
            human_proteins = json.load(f)
        prot_organs = {}
        for k, v in human_proteins.items():
            filename = os.path.join(protein_db_path, v)
            with open(filename, "r") as f:
                prot = json.load(f)
                prot_info = prot["to"]
                f.close()
            # If gene exists in protein info, make a list with genes and synonyms
            if "genes" in prot_info:
                genes = prot_info["genes"]
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                for gene in genes:
                    gene_list = list()
                    if "geneName" in gene:
                        gene_list.append(gene["geneName"]["value"])
                    if "synonyms" in gene:
                        for s in gene["synonyms"]:
                            gene_list.append(s["value"])
                # After the gene list is ready, add entry name
                gene_list = list(set(gene_list + [entry_name]))
            else:
                # If gene does not exist in protein info, gene name is entry name
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                gene_list = [entry_name]
    
        # Asignment of tissue(df1) to proteins (human_dict)
            gene1 = df1[df1["Gene name"].isin(gene_list)]
            if len(gene1) > 0:
                prot_organs[k] = {"Organs":gene1["Organ"].unique().tolist()}
        
        # Write dict to json
        with open(prot_organ_file, "w") as f:
            json.dump(prot_organs, f)

    def generate_negatives(self, prot_organ_file, negFile):
        """
            Random generation of PPIs and filtering based on organ information, pickle, goldem standard and experimental PPIs
        """
        print("Generate negatives...")
        
        # Load prot organ:
        with open(prot_organ_file, "r") as f:
            prot_organ = json.load(f)

        # Combine pikla and experimental data:
        expInters = set()
        for k, v in self.ppis.T.to_dict().items():
            expInters.add(tuple(sorted((v["P1"], v["P2"]))))

        pikla = pd.read_csv(os.path.join(args.project_path, args.pikla_path), sep="\t", header=0)
        for k, v in pikla.T.to_dict().items():
            expInters.add(tuple(sorted((v["InteractorA"], v["InteractorB"]))))

        gs_data = pd.read_csv(os.path.join(args.project_path, args.gold_standard))
        gsInt = set()
        for k, v in gs_data.T.to_dict().items():
            gsInt.add(tuple(sorted((v["D1"], v["D2"]))))

        print("Number of experimental and pikla interactions: ",len(expInters))
        print("Generate negatives...")
        start_time = time.time()

        # Open the file in write mode
        with open(negFile, "w") as f:
            # Generate protein pairs
            prot_pairs = itertools.combinations(prot_organ.keys(), 2)
            i = 0
            for pair in prot_pairs:
                pair = tuple(sorted(pair))
                inter_organ = set(prot_organ[pair[0]]["Organs"]).intersection(set(prot_organ[pair[1]]["Organs"]))

                # Check if the proteins are in different organs
                if len(inter_organ) == 0:
                    # Check if pair or its reverse is not already in piklaInters
                    if not pair in expInters:
                        # Get potential domain interactions:
                        if pair[0] in self.prot_domains and pair[1] in self.prot_domains: 
                            p1_domains = self.prot_domains[pair[0]]
                            p2_domains = self.prot_domains[pair[1]]

                            dom_pairs = itertools.product(p1_domains, p2_domains)
                            inGS = False
                            for dom_pair in dom_pairs:
                                dom_pair = tuple(sorted(dom_pair))
                                if dom_pair in gsInt:
                                    inGS = True
                                    break

                            if not inGS:
                                i += 1
                                # Write the negative pair to the text file
                                f.write(f"{pair[0]}\t{pair[1]}\n")
                            else:
                                print("Found pair in Golden Standard: {}".format(pair))
                    else:
                        print("Found pair in Experimental: {}".format(pair))

        end_time = time.time()

        print(f"Needed {(end_time - start_time) / 60:.2f} minutes to generate all possible protein pairs that are not in the same organ...")
        print("Number of potential negative PPIS: ",i)

    def parse_negatives(self):
        """
            Calls generate negative, and does random sampling of the generated negatives to be equal to PPIs
        """        
        # Proteins to organs mapping.
        prot_organ_file = os.path.join(args.project_path, args.prot_organ_path)

        if not os.path.isfile(prot_organ_file):
            self.get_prot_organ(prot_organ_file)

        self.prot_domains = {}
        # Combine exp prot domains and human prot domains
        with open(os.path.join(args.project_path, args.exp_prots_domains), "r") as f:
            exp_prot_domains = json.load(f)
            f.close()
        
        with open(os.path.join(args.project_path, args.human_protein_domains), "r") as f:
            human_prot_domains = json.load(f)
            f.close()
        
        self.prot_domains.update(exp_prot_domains)
        self.prot_domains.update(human_prot_domains)

        negFile = os.path.join(args.project_path, args.negative_dataset_file)

        if not os.path.isfile(negFile):
            self.generate_negatives(prot_organ_file, negFile)
        
        negDF = pd.read_csv(negFile, sep="\t", header=None)
        negDF.columns = ["P1", "P2"]

        negDF = negDF.sample(n=len(self.ppis), random_state=13).reset_index(drop=True)

        return negDF

    def perform_split(self, dataset):
        test = {}
        train = {}
        train_idx = 0
        test_idx = 0
        potential_ddis_train = set()
        potential_ddis_test = set()

        # Protein pairs that must be in the test set
        to_test = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/testingdf.csv"
        to_testDF = pd.read_csv(to_test)
        predefined_test_pairs = list(zip(to_testDF["Bacteria (UniprotID)"], to_testDF["Host Protein (UniprotID)"]))
        predefined_test_set = set(predefined_test_pairs)

        def add_to_set(data_dict, idx, p1, p2):
            """Helper function to add a protein pair to a dataset."""
            data_dict[idx] = {"P1": p1, "P2": p2}
            return idx + 1

        # Force pre-defined pairs into the test set
        for p1, p2 in predefined_test_set:
            domains_protein1 = self.prot_domains.get(p1, [])
            domains_protein2 = self.prot_domains.get(p2, [])
            pfamPairs = [frozenset(pair) for pair in itertools.product(domains_protein1, domains_protein2)]
            
            # Add to test set, bypassing domain constraint
            test_idx = add_to_set(test, test_idx, p1, p2)
            potential_ddis_test.update(pfamPairs)

        # Process the rest of the dataset
        for idx, row in dataset.sample(frac=1).iterrows():
            p1, p2 = row["P1"], row["P2"]

            # Skip predefined pairs already processed
            if (p1, p2) in predefined_test_set or (p2, p1) in predefined_test_set:
                continue

            domains_protein1 = self.prot_domains.get(p1, [])
            domains_protein2 = self.prot_domains.get(p2, [])
            pfamPairs = [frozenset(pair) for pair in itertools.product(domains_protein1, domains_protein2)]

            # Check domain constraints for the remaining dataset
            if all(pfam_pair in potential_ddis_train for pfam_pair in pfamPairs) and random.random() < 0.3:
                # Add to test set
                test_idx = add_to_set(test, test_idx, p1, p2)
                potential_ddis_test.update(pfamPairs)
            else:
                # Add to training set
                train_idx = add_to_set(train, train_idx, p1, p2)
                potential_ddis_train.update(pfamPairs)

        return train, test

    def dataset_split(self):
        
        # Split datasets:
        print("Split datasets...")
   
        train_ppis, test_ppis = self.perform_split(self.ppis)
        train_negatives, test_negatives = self.perform_split(self.negatives)
        train_ppis, val_ppis = self.perform_split(pd.DataFrame.from_dict(train_ppis, orient='index'))
        train_negatives, val_negatives = self.perform_split(pd.DataFrame.from_dict(train_negatives, orient='index'))
        print("Number of train PPIs: ", len(train_ppis))
        print("Number of train negative samples: ", len(train_negatives))
        print("Number of validation PPIs: ", len(val_ppis))
        print("Number of validation negative samples: ", len(val_negatives))
        print("Number of test PPIs: ", len(test_ppis))
        print("Number of test negative samples: ", len(test_negatives))

        # Combine datasets, assign labels (0 non interaction, 1 interaction) and shuffle:
        train_ppis = pd.DataFrame.from_dict(train_ppis, orient='index')
        train_negatives = pd.DataFrame.from_dict(train_negatives, orient='index')
        train_ppis["label"] = 1
        train_negatives["label"] = 0
        train = pd.concat([train_ppis, train_negatives], ignore_index=True)
        train = train.sample(frac=1).reset_index(drop=True)

        val_ppis = pd.DataFrame.from_dict(val_ppis, orient='index')
        val_negatives = pd.DataFrame.from_dict(val_negatives, orient='index')
        val_ppis["label"] = 1
        val_negatives["label"] = 0
        val = pd.concat([val_ppis, val_negatives], ignore_index=True)
        val = val.sample(frac=1).reset_index(drop=True)

        test_ppis = pd.DataFrame.from_dict(test_ppis, orient='index')
        test_negatives = pd.DataFrame.from_dict(test_negatives, orient='index')
        test_ppis["label"] = 1
        test_negatives["label"] = 0
        test = pd.concat([test_ppis, test_negatives], ignore_index=True)
        test = test.sample(frac=1).reset_index(drop=True)

        # Save datasets:
        train.to_csv(os.path.join(args.project_path, args.dataset, "train.csv"), index=False)
        val.to_csv(os.path.join(args.project_path, args.dataset, "val.csv"), index=False)
        test.to_csv(os.path.join(args.project_path, args.dataset, "test.csv"), index=False) 

class GutDataParser():
    """
        Gut data handler class.
    """
    def __init__(self):
        self.mode = args.gut_mode

        self.gutBacMap = os.path.join(args.project_path, args.gut_bac_map)
        self.gutAtlas = os.path.join(args.project_path, args.gut_microbiome_atlas)
        self.vectAtlas = os.path.join(args.project_path, args.vect_atlas)

        self.modeDir = os.path.join(args.project_path, args.gut_data_dir, self.mode)
        os.makedirs(self.modeDir,exist_ok=True)

        self.gutBactInfo = os.path.join(self.modeDir,"gutBac_{}.csv".format(self.mode))
        self.gutProteomsDir = os.path.join(self.modeDir, args.gut_proteoms_dir)
        os.makedirs(self.gutProteomsDir,exist_ok=True)

        self.prot_db_path = os.path.join(args.project_path, args.protein_db)
        self.modeProteinsFile = os.path.join(self.modeDir, "{}_proteins.json".format(self.mode))
        self.modeProteinsDomainsFile = os.path.join(self.modeDir, "{}_proteins_domains.json".format(self.mode))
        self.modeProteinsRef90File =  os.path.join(self.modeDir, "{}_cluster_info.txt".format(self.mode))
        self.fuzzyModeClusters = os.path.join(self.modeDir, "fuzzy_{}_clusters.txt".format(self.mode))
        self.hum_prot_brain = os.path.join(args.project_path, args.brain_human_proteins)
        self.hum_prot_gut = os.path.join(args.project_path, args.gut_human_proteins)    
        self.humanProteinsFile = os.path.join(args.project_path, args.gut_data_dir, "human_proteins.json")
        self.humanProteinRef90File = os.path.join(args.project_path, args.gut_data_dir, "human_proteins_ref90.txt")
        self.fuzzyHumanClusters = os.path.join(args.project_path, args.gut_data_dir, "fuzzy_human_clusters.txt")

        print("MODE: {}".format(self.mode))

        if not os.path.isfile(self.gutBactInfo):
            self.parse_bac_protein_data()

        gutBacDF = pd.read_csv(self.gutBactInfo)
        print("Found data for {} bacteria.".format(len(gutBacDF)))
        
        if not os.path.isfile(self.modeProteinsFile):
            self.get_bac_proteins(gutBacDF)

        if not os.path.isfile(self.modeProteinsDomainsFile):
            self.get_bac_proteins_domains()

        # Number of mode proteins:
        with open(self.modeProteinsFile, "r") as f:
            modeProteins = json.load(f)
            f.close()

        print("Number of {} proteins: {}".format(self.mode, len(modeProteins)))

        if not os.path.isfile(self.humanProteinsFile):
            self.parse_human_protein_data()

        # Number of human proteins:
        with open(self.humanProteinsFile, "r") as f:
            humanProteins = json.load(f)
            f.close()

        print("Number of human proteins: {}".format(len(humanProteins)))

        if not os.path.isfile(self.modeProteinsRef90File):
            self.get_bac_proteins_ref90()
        
        # Number of mode proteins in ref90:
        with open(self.modeProteinsRef90File, "r") as f:
            modeProteinsRef90 = json.load(f)
            f.close()
        
        print("Number of {} proteins in ref90: {}".format(self.mode, len(modeProteinsRef90)))

        if not os.path.isfile(self.humanProteinRef90File):
            self.get_human_proteins_ref90()
        
        # Bacterial Clusters
        if not os.path.isfile(self.fuzzyModeClusters):
            self.get_fuzzy_clusters(modeProteins, self.fuzzyModeClusters)
        # Human Clusters
        if not os.path.isfile(self.fuzzyHumanClusters):
            self.get_fuzzy_clusters(humanProteins, self.fuzzyHumanClusters)

    def parse_bac_protein_data(self):
        """
            Map Gut Bacterial to proteins.
        """
        gutAtlasDF = pd.ExcelFile(self.gutAtlas)
        gutBacMapDF = pd.ExcelFile(self.gutBacMap)

        dfList = list()
        for sheet in gutBacMapDF.sheet_names:
            atlasF = gutAtlasDF.parse(sheet)
            bacF = gutBacMapDF.parse(sheet)
            
            if "1805478" or 1805478 in atlasF.columns:
                atlasF = atlasF.rename(columns={"1805478":"MSP"})
                atlasF = atlasF.rename(columns={1805478:"MSP"})

            atlasF['MSP'] = atlasF['MSP'].fillna(method='ffill')

            # Group by MSP, keep other columns and add Disease association to list:
            atlasF = atlasF.rename(columns={"Species name":"name"})

            mergedDF = pd.merge(bacF, atlasF, on="name")
            dfList.append(mergedDF)

        gutBacDF = pd.concat(dfList)
        gutBacDF = gutBacDF.dropna(subset=["MSP"])
        gutBacDF["MSP"] = gutBacDF["MSP"].str.lower()

        vect_atlasDF = pd.read_csv(self.vectAtlas,index_col=0)
        vect_atlasDF = vect_atlasDF.reset_index()
        vect_atlasDF.rename(columns={"index":"MSP"}, inplace=True)
        gutBacDF = pd.merge(gutBacDF, vect_atlasDF, on="MSP")
        gutBacDF = gutBacDF.dropna(subset=["MSP"])

        cols = list()
        for col in gutBacDF.columns:
            if not "Unnamed" in str(col):
                cols.append(col)
        gutBacDF = gutBacDF[cols]

        if not self.mode == "Unspecified":
            gutBacDF = gutBacDF.dropna(subset=["Disease association"])
            gutBacDF = gutBacDF[gutBacDF["Disease association"].str.contains(self.mode)]
                        
            gutBacDF["abundance"] = gutBacDF.select_dtypes(include=['float64']).mean(axis=1)
            
            gutBacDF = gutBacDF[["name","taxid","Genus","Disease association","Proteome ID(with higher BUSCO)","MSP","abundance"]]
        
        else:
            # Select all rows if disease association is NaN:
            gutBacDF = gutBacDF[gutBacDF["Disease association"].isnull()]
            # Disease association to Unspecified:
            gutBacDF["Disease association"] = "Unspecified"
            gutBacDF["abundance"] = gutBacDF.select_dtypes(include=['float64']).mean(axis=1) 
            gutBacDF = gutBacDF[["name","taxid","Genus","Disease association","Proteome ID(with higher BUSCO)","MSP","abundance"]]
           
        gutBacDF = gutBacDF.reset_index(drop=True)

        for k, v in gutBacDF.T.to_dict().items():
            protFile = self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])
                    
            if not os.path.isfile(protFile):
                print("Proteome {} of {}".format(k+1,len(gutBacDF)))
                reviewed = list()

                req = "https://www.ebi.ac.uk/proteins/api/proteomes/proteins/{}?reviewed=True".format(v["Proteome ID(with higher BUSCO)"])
                r = requests.get(req)
                res = r.json()

                if "component" in res and len(res["component"]) > 0:
                    if "protein" in res["component"][0]:
                        for p in res["component"][0]["protein"]:
                            reviewed.append(p["accession"])

                unreviewed = list()

                req = "https://www.ebi.ac.uk/proteins/api/proteomes/proteins/{}?reviewed=False".format(v["Proteome ID(with higher BUSCO)"])
                r = requests.get(req)
                res = r.json()
                if "component" in res and len(res["component"]) > 0:
                    if "protein" in res["component"][0]:
                        for p in res["component"][0]["protein"]:
                            unreviewed.append(p["accession"])
                
                protList = reviewed + unreviewed
                protList = list(set(protList))

                with open(protFile, "wb") as fp:   
                    pickle.dump(protList, fp)

        # Save gutBacDF:
        gutBacDF.to_csv(self.gutBactInfo, index=False)

    def get_bac_proteins(self,gutBacDF):
        protList = list()

        for k, v in gutBacDF.T.to_dict().items():
            if os.path.isfile(self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])):
                protFile = self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])
                with open(protFile, "rb") as fp:   
                    protList.extend(pickle.load(fp))
        # Unique proteins
        protList = sorted(list(set(protList)))

        print("Number of proteins: {}".format(len(protList)))

        # Uniprot mapping
        UniprotMapping(protList,self.modeProteinsFile)

    def get_bac_proteins_domains(self):
        """
            Get protein domains from protein database.
            Generates a json file with uniprot id as keys and a list of domains as the corresponging values.
        """
        # Read protein indexing file
        with open(self.modeProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Get protein domains
        prot_domains = {}
        for prot, file in prot_indexing.items():
            protFile = os.path.join(args.project_path, args.protein_db, file)
            with open(protFile, "r") as f:
                protInfo = json.load(f)
                domains = list()
                for cr in protInfo["to"]["uniProtKBCrossReferences"]:
                    if cr["database"] == "Pfam":
                        domains.append(cr["id"])

                if len(domains) > 0:
                    prot_domains[prot] = domains

                f.close()
        
        # Write protein domains
        with open(self.modeProteinsDomainsFile, "w") as f:
            json.dump(prot_domains, f)
            f.close()

    def parse_human_protein_data(self):
        """
        Filter protein database to get only human proteins present in gut and brain.
        """
        # Read protein db, brain and gut proteins
        human_proteins_path = os.path.join(args.project_path,args.human_protein_info)
        protein_db_path = os.path.join(args.project_path, args.protein_db)
        with open(human_proteins_path, "r") as f:
            human_proteins = json.load(f)
        brainDF = pd.read_csv(self.hum_prot_brain,index_col=0)
        gutDF = pd.read_csv(self.hum_prot_gut,index_col=0)
        # Make a list with brain and gut genes
        brain_genes = brainDF["Gene name"].unique().tolist()
        gut_genes = gutDF["Gene name"].unique().tolist()
        # Make gene list for every human protein in protein database (from genes + entry name (without_HUMAN))
        gut_brain_proteins = {}
        for k, v in human_proteins.items():
            filename = os.path.join(protein_db_path, v)
            with open(filename, "r") as f:
                prot = json.load(f)
                prot_info = prot["to"]
                f.close()
            # If gene exists in protein info, make a list with genes and synonyms
            if "genes" in prot_info:
                genes = prot_info["genes"]
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                for gene in genes:
                    gene_list = list()
                    if "geneName" in gene:
                        gene_list.append(gene["geneName"]["value"])
                    if "synonyms" in gene:
                        for s in gene["synonyms"]:
                            gene_list.append(s["value"])
                # After the gene list is ready, add entry name
                gene_list = list(set(gene_list + [entry_name]))
            else:
                # If gene does not exist in protein info, gene name is entry name
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                gene_list = [entry_name]
        
            # Check if gene list has any brain or gut genes
            if len(set(gene_list).intersection(brain_genes)) > 0 and len(set(gene_list).intersection(gut_genes)) > 0:
                # k: UniProtID, v: protein file name
                gut_brain_proteins[k] = v

        # Write dict to json
        with open(self.humanProteinsFile, "w") as f:
            json.dump(gut_brain_proteins, f)

    def get_bac_proteins_ref90(self):
        """
            Get protein ref90 from protein database.
            Generates a json file with uniprot id as keys and a list of ref90 as the corresponging values.
        """
        # Read protein indexing file
        with open(self.modeProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Read domains
        with open(self.modeProteinsDomainsFile, "r") as f:
            prot_domains = json.load(f)
            f.close()

        UnirefMapping(list(prot_indexing.keys()),self.modeProteinsRef90File,prot_domains)

    def get_human_proteins_ref90(self):
        """
            Get protein ref90 from protein database.
            Generates a json file with uniprot id as keys and a list of ref90 as the corresponging values.
        """
        # Read protein indexing file
        with open(self.humanProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Read human_proteins_domains
        with open(os.path.join(args.project_path, args.human_protein_domains), "r") as f:
            prot_domains = json.load(f)
            f.close()

        UnirefMapping(list(prot_indexing.keys()),self.humanProteinRef90File,prot_domains)
    
    def calculate_gene_similarity(self, genes1, genes2, threshold=0.7):
        for g1 in genes1:
            for g2 in genes2:
                ratio = fuzz.ratio(g1, g2)
                if ratio > threshold:
                    return True
        return False

    def load_protein_features(self,prot):
        prot_file = os.path.join(args.project_path, args.protein_db, prot + ".json")
        with open(prot_file, "r") as f:
            prot_res = json.load(f)

        genes = {gene["geneName"]["value"] for gene in prot_res["to"]["genes"] if "geneName" in gene}
        synonyms = {s["value"] for gene in prot_res["to"]["genes"] if "synonyms" in gene for s in gene["synonyms"]}
        gene_list = list(genes.union(synonyms, [prot_res["to"]["uniProtkbId"].split("_")[0]]))
        #print(genes)
        pfam_list = [cr["id"] for cr in prot_res["to"]["uniProtKBCrossReferences"] if cr["database"] == "Pfam"]

        return {"genes": gene_list,"pfam": pfam_list}

    def find_pfam_equality(self,pfam1, pfam2):
        for p1 in pfam1:
            for p2 in pfam2:
                if p1 == p2:
                    return True
        return False

    def get_fuzzy_clusters(self, prot_list, fuzzy_clusters_file):
        """
            Get fuzzy clusters from protein database.
            Criteria: > 0.7 Gene similarity, >= 1 same PfamIDs
        """
        # Features dicitonary: key --> UniProtID, values--> dict with genes and pfamIDs
        features = {prot: self.load_protein_features(prot) for prot in prot_list}

        # Copy features
        features_copy = features.copy()

        with open(fuzzy_clusters_file, "w") as f:
            cluster_idx = 0

            for pidx, (k1,v1) in enumerate(features.items()):
                # Check if protein is in temp file
                if k1 in features_copy:
                    sims = set()
                    for k2,v2 in features_copy.items():
                        if k1 == k2:
                            continue
                        # Calculate gene similarity
                        if self.calculate_gene_similarity(v1["genes"], v2["genes"]):
                            if self.find_pfam_equality(v1["pfam"], v2["pfam"]):
                                sims.add(k2)
                    sims.add(k1)
                    if len(sims) > 0:
                        
                        # Write to file
                        # cluster_idx : set

                        # Assign representative protein: the one with the most pfamIDs
                        max_pfam = 0
                        rep_prot = ""
                        for prot in sims:
                            if len(features[prot]["pfam"]) > max_pfam:
                                max_pfam = len(features[prot]["pfam"])
                                rep_prot = prot

                        f.write("{}\t{}\n".format("fc_{}".format(rep_prot), "\t".join(sims)))
                        cluster_idx += 1
                        # delete from features_copy
                        for k in sims:
                            del features_copy[k]
                        
                print("Protein: ", pidx, " / ", len(features), " - ", len(features_copy))

class CreteGutDataParser():
    """
        Gut data handler class.
    """
    def __init__(self):
        self.mode = args.gut_mode

        self.gutBacMap = os.path.join(args.project_path, args.gut_bac_map)
        self.gutAtlas = os.path.join(args.project_path, args.gut_microbiome_atlas)
        self.vectAtlas = os.path.join(args.project_path, args.vect_atlas)

        self.modeDir = os.path.join(args.project_path, args.gut_data_dir, self.mode)
        os.makedirs(self.modeDir,exist_ok=True)

        self.gutBactInfo = os.path.join(self.modeDir,"gutBac_{}.csv".format(self.mode))
        self.gutProteomsDir = os.path.join(self.modeDir, args.gut_proteoms_dir)
        os.makedirs(self.gutProteomsDir,exist_ok=True)

        self.prot_db_path = os.path.join(args.project_path, args.protein_db)
        self.modeProteinsFile = os.path.join(self.modeDir, "{}_proteins.json".format(self.mode))
        self.modeProteinsDomainsFile = os.path.join(self.modeDir, "{}_proteins_domains.json".format(self.mode))
        self.modeProteinsRef90File =  os.path.join(self.modeDir, "{}_cluster_info.txt".format(self.mode))
        self.fuzzyModeClusters = os.path.join(self.modeDir, "fuzzy_{}_clusters.txt".format(self.mode))
        self.hum_prot_brain = os.path.join(args.project_path, args.brain_human_proteins)
        self.hum_prot_gut = os.path.join(args.project_path, args.gut_human_proteins)    
        self.humanProteinsFile = os.path.join(args.project_path, args.gut_data_dir, "human_proteins.json")
        self.humanProteinRef90File = os.path.join(args.project_path, args.gut_data_dir, "human_proteins_ref90.txt")
        self.fuzzyHumanClusters = os.path.join(args.project_path, args.gut_data_dir, "fuzzy_human_clusters.txt")

        print("MODE: {}".format(self.mode))

        if not os.path.isfile(self.gutBactInfo):
            self.parse_bac_protein_data()

        gutBacDF = pd.read_csv(self.gutBactInfo)
        print("Found data for {} bacteria.".format(len(gutBacDF)))
        
        if not os.path.isfile(self.modeProteinsFile):
            self.get_bac_proteins(gutBacDF)

        if not os.path.isfile(self.modeProteinsDomainsFile):
            self.get_bac_proteins_domains()

        # Number of mode proteins:
        with open(self.modeProteinsFile, "r") as f:
            modeProteins = json.load(f)
            f.close()

        print("Number of {} proteins: {}".format(self.mode, len(modeProteins)))

        if not os.path.isfile(self.humanProteinsFile):
            self.parse_human_protein_data()

        # Number of human proteins:
        with open(self.humanProteinsFile, "r") as f:
            humanProteins = json.load(f)
            f.close()

        print("Number of human proteins: {}".format(len(humanProteins)))

        if not os.path.isfile(self.modeProteinsRef90File):
            self.get_bac_proteins_ref90()
        
        # Number of mode proteins in ref90:
        with open(self.modeProteinsRef90File, "r") as f:
            modeProteinsRef90 = json.load(f)
            f.close()
        
        print("Number of {} proteins in ref90: {}".format(self.mode, len(modeProteinsRef90)))

        if not os.path.isfile(self.humanProteinRef90File):
            self.get_human_proteins_ref90()
        
        # Bacterial Clusters
        if not os.path.isfile(self.fuzzyModeClusters):
            self.get_fuzzy_clusters(modeProteins, self.fuzzyModeClusters)
        # Human Clusters
        if not os.path.isfile(self.fuzzyHumanClusters):
            self.get_fuzzy_clusters(humanProteins, self.fuzzyHumanClusters)

    def parse_bac_protein_data(self):
        """
            Map Gut Bacterial to proteins.
        """
        gutAtlasDF = pd.ExcelFile(self.gutAtlas)
        gutBacMapDF = pd.ExcelFile(self.gutBacMap)

        dfList = list()
        for sheet in gutBacMapDF.sheet_names:
            atlasF = gutAtlasDF.parse(sheet)
            bacF = gutBacMapDF.parse(sheet)
            
            if "1805478" or 1805478 in atlasF.columns:
                atlasF = atlasF.rename(columns={"1805478":"MSP"})
                atlasF = atlasF.rename(columns={1805478:"MSP"})

            atlasF['MSP'] = atlasF['MSP'].fillna(method='ffill')

            # Group by MSP, keep other columns and add Disease association to list:
            atlasF = atlasF.rename(columns={"Species name":"name"})

            mergedDF = pd.merge(bacF, atlasF, on="name")
            dfList.append(mergedDF)

        gutBacDF = pd.concat(dfList)
        gutBacDF = gutBacDF.dropna(subset=["MSP"])
        gutBacDF["MSP"] = gutBacDF["MSP"].str.lower()

        vect_atlasDF = pd.read_csv(self.vectAtlas,index_col=0)
        vect_atlasDF = vect_atlasDF.reset_index()
        vect_atlasDF.rename(columns={"index":"MSP"}, inplace=True)
        gutBacDF = pd.merge(gutBacDF, vect_atlasDF, on="MSP")
        gutBacDF = gutBacDF.dropna(subset=["MSP"])

        cols = list()
        for col in gutBacDF.columns:
            if not "Unnamed" in str(col):
                cols.append(col)
        gutBacDF = gutBacDF[cols]
        
        if self.mode == "Crete_Patients":
            filteredPath = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/Crete/FILTERED_S_Patients.xlsx"
            sheet_names = ["hgma", "hand_mapped"]
        else:
            filteredPath = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/Gut Data/Crete/FILTERED_S_Healthy.xlsx"

        # Read excel and keep only sheets in sheet_names:
        filteredDF = pd.ExcelFile(filteredPath)

        dfList = list()
        for sheet in sheet_names:
            filteredF = filteredDF.parse(sheet)
            # If a column startswith "UP", column names dont exist and first row is header:
            for col in filteredF.columns:
                if col.startswith("UP"):
                    first_row_data = filteredF.iloc[0].tolist()        
                    filteredF.columns = ["name","Proteome ID(with higher BUSCO)"]
                    # Add first row as data:
                    filteredF = pd.concat([filteredF,pd.Series(first_row_data, index=filteredF.columns)], ignore_index=True)
                    break

            filteredF = filteredF[["name","Proteome ID(with higher BUSCO)"]]
            dfList.append(filteredF)

        filteredF = pd.concat(dfList, ignore_index=True)
        filteredF = filteredF.dropna().reset_index(drop=True)
        gutBacDF = gutBacDF.reset_index(drop=True)
 
        # Merge on Proteome ID(with higher BUSCO). Prefixes for same columns: db_ and cr_
        gutBacDF = pd.merge(gutBacDF, filteredF, on="Proteome ID(with higher BUSCO)", how="right", suffixes=("_db","_cr"))
        
        gutBacDF["abundance"] = gutBacDF.select_dtypes(include=['float64']).mean(axis=1)
            
        gutBacDF = gutBacDF[["name_db","name_cr","taxid","Genus","Disease association","Proteome ID(with higher BUSCO)","MSP","abundance"]]

        gutBacDF = gutBacDF.reset_index(drop=True)

        # Save gutBacDF:
        gutBacDF.to_csv(self.gutBactInfo, index=False)
 
        for k, v in gutBacDF.T.to_dict().items():
            protFile = self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])
                    
            if not os.path.isfile(protFile):
                print("Proteome {} of {}".format(k+1,len(gutBacDF)))
                reviewed = list()

                req = "https://www.ebi.ac.uk/proteins/api/proteomes/proteins/{}?reviewed=True".format(v["Proteome ID(with higher BUSCO)"])
                r = requests.get(req)
                res = r.json()

                if "component" in res and len(res["component"]) > 0:
                    if "protein" in res["component"][0]:
                        for p in res["component"][0]["protein"]:
                            reviewed.append(p["accession"])

                unreviewed = list()

                req = "https://www.ebi.ac.uk/proteins/api/proteomes/proteins/{}?reviewed=False".format(v["Proteome ID(with higher BUSCO)"])
                r = requests.get(req)
                res = r.json()
                if "component" in res and len(res["component"]) > 0:
                    if "protein" in res["component"][0]:
                        for p in res["component"][0]["protein"]:
                            unreviewed.append(p["accession"])
                
                protList = reviewed + unreviewed
                protList = list(set(protList))

                with open(protFile, "wb") as fp:   
                    pickle.dump(protList, fp)

        # Save gutBacDF:
        gutBacDF.to_csv(self.gutBactInfo, index=False)

    def get_bac_proteins(self,gutBacDF):
        protList = list()

        for k, v in gutBacDF.T.to_dict().items():
            if os.path.isfile(self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])):
                protFile = self.gutProteomsDir + "/" + "{}.pkl".format(v["Proteome ID(with higher BUSCO)"])
                with open(protFile, "rb") as fp:   
                    protList.extend(pickle.load(fp))
        # Unique proteins
        protList = sorted(list(set(protList)))

        print("Number of proteins: {}".format(len(protList)))

        # Uniprot mapping
        UniprotMapping(protList,self.modeProteinsFile)

    def get_bac_proteins_domains(self):
        """
            Get protein domains from protein database.
            Generates a json file with uniprot id as keys and a list of domains as the corresponging values.
        """
        # Read protein indexing file
        with open(self.modeProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Get protein domains
        prot_domains = {}
        for prot, file in prot_indexing.items():
            protFile = os.path.join(args.project_path, args.protein_db, file)
            with open(protFile, "r") as f:
                protInfo = json.load(f)
                domains = list()
                for cr in protInfo["to"]["uniProtKBCrossReferences"]:
                    if cr["database"] == "Pfam":
                        domains.append(cr["id"])

                if len(domains) > 0:
                    prot_domains[prot] = domains

                f.close()
        
        # Write protein domains
        with open(self.modeProteinsDomainsFile, "w") as f:
            json.dump(prot_domains, f)
            f.close()

    def parse_human_protein_data(self):
        """
        Filter protein database to get only human proteins present in gut and brain.
        """
        # Read protein db, brain and gut proteins
        human_proteins_path = os.path.join(args.project_path,args.human_protein_info)
        protein_db_path = os.path.join(args.project_path, args.protein_db)
        with open(human_proteins_path, "r") as f:
            human_proteins = json.load(f)
        brainDF = pd.read_csv(self.hum_prot_brain,index_col=0)
        gutDF = pd.read_csv(self.hum_prot_gut,index_col=0)
        # Make a list with brain and gut genes
        brain_genes = brainDF["Gene name"].unique().tolist()
        gut_genes = gutDF["Gene name"].unique().tolist()
        # Make gene list for every human protein in protein database (from genes + entry name (without_HUMAN))
        gut_brain_proteins = {}
        for k, v in human_proteins.items():
            filename = os.path.join(protein_db_path, v)
            with open(filename, "r") as f:
                prot = json.load(f)
                prot_info = prot["to"]
                f.close()
            # If gene exists in protein info, make a list with genes and synonyms
            if "genes" in prot_info:
                genes = prot_info["genes"]
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                for gene in genes:
                    gene_list = list()
                    if "geneName" in gene:
                        gene_list.append(gene["geneName"]["value"])
                    if "synonyms" in gene:
                        for s in gene["synonyms"]:
                            gene_list.append(s["value"])
                # After the gene list is ready, add entry name
                gene_list = list(set(gene_list + [entry_name]))
            else:
                # If gene does not exist in protein info, gene name is entry name
                entry_name = prot_info["uniProtkbId"].split("_")[0]
                gene_list = [entry_name]
        
            # Check if gene list has any brain or gut genes
            if len(set(gene_list).intersection(brain_genes)) > 0 and len(set(gene_list).intersection(gut_genes)) > 0:
                # k: UniProtID, v: protein file name
                gut_brain_proteins[k] = v

        # Write dict to json
        with open(self.humanProteinsFile, "w") as f:
            json.dump(gut_brain_proteins, f)

    def get_bac_proteins_ref90(self):
        """
            Get protein ref90 from protein database.
            Generates a json file with uniprot id as keys and a list of ref90 as the corresponging values.
        """
        # Read protein indexing file
        with open(self.modeProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Read domains
        with open(self.modeProteinsDomainsFile, "r") as f:
            prot_domains = json.load(f)
            f.close()

        UnirefMapping(list(prot_indexing.keys()),self.modeProteinsRef90File,prot_domains)

    def get_human_proteins_ref90(self):
        """
            Get protein ref90 from protein database.
            Generates a json file with uniprot id as keys and a list of ref90 as the corresponging values.
        """
        # Read protein indexing file
        with open(self.humanProteinsFile, "r") as f:
            prot_indexing = json.load(f)
            f.close()

        # Read human_proteins_domains
        with open(os.path.join(args.project_path, args.human_protein_domains), "r") as f:
            prot_domains = json.load(f)
            f.close()

        UnirefMapping(list(prot_indexing.keys()),self.humanProteinRef90File,prot_domains)
    
    def calculate_gene_similarity(self, genes1, genes2, threshold=0.7):
        for g1 in genes1:
            for g2 in genes2:
                ratio = fuzz.ratio(g1, g2)
                if ratio > threshold:
                    return True
        return False

    def load_protein_features(self,prot):
        prot_file = os.path.join(args.project_path, args.protein_db, prot + ".json")
        with open(prot_file, "r") as f:
            prot_res = json.load(f)

        genes = {gene["geneName"]["value"] for gene in prot_res["to"]["genes"] if "geneName" in gene}
        synonyms = {s["value"] for gene in prot_res["to"]["genes"] if "synonyms" in gene for s in gene["synonyms"]}
        gene_list = list(genes.union(synonyms, [prot_res["to"]["uniProtkbId"].split("_")[0]]))
        #print(genes)
        pfam_list = [cr["id"] for cr in prot_res["to"]["uniProtKBCrossReferences"] if cr["database"] == "Pfam"]

        return {"genes": gene_list,"pfam": pfam_list}

    def find_pfam_equality(self,pfam1, pfam2):
        for p1 in pfam1:
            for p2 in pfam2:
                if p1 == p2:
                    return True
        return False

    def get_fuzzy_clusters(self, prot_list, fuzzy_clusters_file):
        """
            Get fuzzy clusters from protein database.
            Criteria: > 0.7 Gene similarity, >= 1 same PfamIDs
        """
        # Features dicitonary: key --> UniProtID, values--> dict with genes and pfamIDs
        features = {prot: self.load_protein_features(prot) for prot in prot_list}

        # Copy features
        features_copy = features.copy()

        with open(fuzzy_clusters_file, "w") as f:
            cluster_idx = 0

            for pidx, (k1,v1) in enumerate(features.items()):
                # Check if protein is in temp file
                if k1 in features_copy:
                    sims = set()
                    for k2,v2 in features_copy.items():
                        if k1 == k2:
                            continue
                        # Calculate gene similarity
                        if self.calculate_gene_similarity(v1["genes"], v2["genes"]):
                            if self.find_pfam_equality(v1["pfam"], v2["pfam"]):
                                sims.add(k2)
                    sims.add(k1)
                    if len(sims) > 0:
                        
                        # Write to file
                        # cluster_idx : set

                        # Assign representative protein: the one with the most pfamIDs
                        max_pfam = 0
                        rep_prot = ""
                        for prot in sims:
                            if len(features[prot]["pfam"]) > max_pfam:
                                max_pfam = len(features[prot]["pfam"])
                                rep_prot = prot

                        f.write("{}\t{}\n".format("fc_{}".format(rep_prot), "\t".join(sims)))
                        cluster_idx += 1
                        # delete from features_copy
                        for k in sims:
                            del features_copy[k]
                        
                print("Protein: ", pidx, " / ", len(features), " - ", len(features_copy))

#   ----------------------------------------------------------------------------    #
#   Functions                                                                       #
#   ----------------------------------------------------------------------------    #

def argParse():
    """
        Parse script arguments.
    """
    parser = argparse.ArgumentParser(description='Domain - Domain interaction prediction.')
    # Project path
    parser.add_argument('-pp', '--project_path', type=str, help='Project path.', default="/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA")
    
    # Initial data

    # Experimental data
    parser.add_argument('-ed', '--exp_data', type=str, help='Experimental data.', default="data/ExperimentalInteractions.txt")
    # PPIdomainminer dataset
    parser.add_argument('-ppidm', '--ppidm_data', type=str, help='PPIdomainminer dataset.', default="data/PPIdomainminerinput.txt")
    # Human proteome
    parser.add_argument('-hpdb', '--human_proteome', type=str, help='Human proteins UniProtIDs.', default="data/HumanProteomeUniprotIDs.xlsx")
    # Gut data directory
    parser.add_argument('-gut', '--gut_data_dir', type=str, help='Gut data directory.', default="data/Gut Data")
    # Gut bacteria mapping
    parser.add_argument('-gbm', '--gut_bac_map', type=str, help='GM bacteria taxid and proteome ids mapping.', default="data/Gut Data/GM_bacteria_taxid_and_proteome_ids_mapping.xlsx")
    # GM bacteria Gut Microbiome Atlas
    parser.add_argument('-gma', '--gut_microbiome_atlas', type=str, help='GM_bacteria_Gut_Microbiome_Atlas.', default="data/Gut Data/GM_bacteria_Gut_Microbiome_Atlas.xlsx")
    # vect_atlas
    parser.add_argument('-va', '--vect_atlas', type=str, help='vect_atlas.', default="data/Gut Data/vect_atlas.csv")
    # Gut Human Proteins
    parser.add_argument('-ghp', '--gut_human_proteins', type=str, help='Gut Human Proteins.', default="data/Gut Data/Human_proteins_gut_f.csv")
    # Brain Human Proteins
    parser.add_argument('-brhp', '--brain_human_proteins', type=str, help='Brain Human Proteins.', default="data/Gut Data/Human_proteins_brain_f.csv")
    # Golden standard data
    parser.add_argument('-gs', '--gold_standard', type=str, help='Golden standard data.', default="data/3did2020.csv")
    # Dataset path
    parser.add_argument('-ds', '--dataset', type=str, help='Dataset to parse.', default='data/Dataset')
    # Negative dataset
    parser.add_argument('-nds', '--negative_dataset', type=str, help='Negative dataset.', default="data/Dataset/Negative Dataset")
    # Pikla path
    parser.add_argument('-pikla', '--pikla_path', type=str, help='Pikla path.', default="data/Dataset/Negative Dataset/UniProtNormalizedTabular-default.txt")
    # Normal tissue path
    parser.add_argument('-nt', '--normal_tissue_path', type=str, help='Normal tissue path.', default="data/Dataset/Negative Dataset/normal_tissue.csv")
    
    # Data during execution

    # Protein Database 
    parser.add_argument('-pdb', '--protein_db', type=str, help='Protein Database.', default="data/ProteinDatabase")
    # Cluster Database 
    parser.add_argument('-clb', '--cluster_db', type=str, help='Cluster Database.', default="data/ClusterDatabase")
    # Human Protein Info
    parser.add_argument('-hpi', '--human_protein_info', type=str, help='Human Protein Info.', default="data/human_proteins.json")
    # Human Protein Domains
    parser.add_argument('-hpd', '--human_protein_domains', type=str, help='Human Protein Domains.', default="data/human_proteins_domains.json")
    # Experimental Proteins
    parser.add_argument('-exp_prot', '--exp_prots', type=str, help='Experimental Proteins.', default="data/ExperimentalProteins.json")
    # Experimental Proteins Domains
    parser.add_argument('-exp_prot_dom', '--exp_prots_domains', type=str, help='Experimental Proteins Domains.', default="data/ExperimentalProteinsDomains.json")
    # Dataset PPIs: Combined PPIS of Experimental + PPIdomainminer
    parser.add_argument('-ds_ppis', '--dataset_ppis', type=str, help='Dataset PPIs.', default="data/Dataset/dataset_ppis.txt")
    # Gut Bac Proteins Directory: It will go to Disease/BacProteins
    parser.add_argument('-gut_prot', '--gut_proteoms_dir', type=str, help='Gut Bac Proteins Directory.', default="BacProteins")
    #  Json file to save the protein domains of the dataset
    parser.add_argument('-ds_prot_dom', '--dataset_prot_domains', type=str, help='prot_domains.json.', default="data/Dataset/dataset_prot_domains.json")
    # Protein organs path
    parser.add_argument('-po', '--prot_organ_path', type=str, help='prot_organ_path.', default="data/Dataset/Negative Dataset/prot_organ.json")
    # Negative pairs file
    parser.add_argument('-negf', '--negative_dataset_file', type=str, help='Negative dataset.', default="data/Dataset/Negative Dataset/negative_pairs.txt")
    
    # Requirements

    # Gut data mode: Disease association
    parser.add_argument('-gm', '--gut_mode', type=str, help='Gut data mode: Disease association', default="Healthy", choices=['Colon adenoma', 'Ankylosing spondylitis', 'Melanoma', 'Colorectal cancer', 'Type 1 diabetes', 'Atherosclerosis', "Crohn's disease", 'Impaired glucose tolerance', 'Type 2 diabetes', 'NAFLD', 'Vogt-Koyanagi-Harada', "Behcet's disease", 'Healthy', 'Liver cirrhosis', 'Cardiovascular disease', 'Acute diarrhea', 'Renal cancer', 'Lung cancer', "Parkinson's disease", 'Chronic fatigue syndrome', 'Ulcerative colitis', 'Unspecified', 'Crete_Patients'])
    # Data type (Experimental, PPIdomainminer, Gut, All) to parse
    parser.add_argument('-dt', '--data_type', type=str, help='Data type (Experimental, PPIdomainminer, Gut, Dataset or All) to parse.', default="Dataset")

    return parser.parse_args()

if __name__ == "__main__":
    args = argParse()
    ExperimentalProteins()
    HumanProteins()
    DatasetParser()
    if "Crete" in args.gut_mode:
        CreteGutDataParser()
    else:
        GutDataParser()