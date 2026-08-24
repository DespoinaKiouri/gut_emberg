import os
import json
import pandas as pd

prot_db = "data/ProteinDatabase"
hum_prot_info_file = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/data/human_proteins.json"
agg_res_file = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/Prediction/Healthy_Unspecified/Aggregated_results_Healthy_Unspecified.xlsx"

# Read HumanProteins sheet
hum_df = pd.read_excel(agg_res_file, sheet_name="FuzzyHumanClusters", header=0, index_col=0)

# Sort by Degree
hum_df = hum_df.sort_values(by=["Degree"], ascending=False).reset_index(drop=True)

# Rename HumanProtein to UniprotID
hum_df = hum_df.rename(columns={"HumanCluster": "UniprotID"})
# split to fc_
hum_df["UniprotID"] = hum_df["UniprotID"].str.split("_").str[1]
print(hum_df.head())


# Convert to dictionary
hum_dict = hum_df.to_dict(orient="index")

# Read hum_prot_info_file
with open(hum_prot_info_file, "r") as f:
    hum_prot_info = json.load(f)

res_dict = {}

for k, v in hum_dict.items():
    un_id = v["UniprotID"]
    
    # Check if un_id in hum_prot_info
    if un_id in hum_prot_info:
        # Map to filename
        prot_file = hum_prot_info[un_id]
        prot_file = os.path.join(prot_db, prot_file)

        # Open file and read
        with open(prot_file, "r") as f:
            prot_data = json.load(f)

        # Find protein name
        prot_data = prot_data["to"]
        if "recommendedName" in prot_data["proteinDescription"]:
            prot_name = prot_data["proteinDescription"]["recommendedName"]["fullName"]["value"]
        else:
            prot_name = prot_data["proteinDescription"]["submissionNames"][0]["fullName"]["value"]

        # Find GO terms
        bp_list = list()
        mf_list = list()

        for db in prot_data["uniProtKBCrossReferences"]:
            if db["database"] == "GO":
                GOid = db["id"]
                for p in db["properties"]:
                    if p["key"] == "GoTerm":
                        goterm = p["value"]
                        # Biological Process GO list
                        if goterm.startswith("P:"):
                            goterm = goterm.replace("P:", "")
                            term_info = {
                                "GO Term": goterm, 
                                "GO id" : GOid
                            }
                            bp_list.append(goterm)
                        # Molecular Function GO list
                        elif goterm.startswith("F:"):
                            goterm = goterm.replace("F:", "")
                            term_info = {
                                "GO Term": goterm, 
                                "GO id" : GOid
                            }
                            mf_list.append(goterm)

        # Add to res_dict
        res_dict[un_id] = {
            "Protein Name": prot_name,
            "Biological Process": bp_list,
            "Molecular Function": mf_list
        }

# To dataframe
res_df = pd.DataFrame.from_dict(res_dict, orient="index")

# Save to file
res_df.to_excel("human_prot_GO.xlsx")
                            