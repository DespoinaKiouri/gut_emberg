import os
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio

NET_INFO_FILE = "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/Results/Prediction/Healthy_Unspecified/Aggregated_results_Healthy_Unspecified.xlsx"

# Set plotly theme to white
pio.templates.default = "plotly_white"

PLOTS_DIR = "./figs"

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def mobiDB(acc):
    url = f'https://mobidb.org/api/download?format=json&acc={acc}'
   
    headers = {
        'accept': 'text/plain'
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def calc_disorder(mobiDB_data):
    curated = 0
    predicted = 0

    if "prediction-disorder-priority" in mobiDB_data:
        predicted = mobiDB_data["prediction-disorder-priority"]["content_count"]

    if "curated-disorder-priority" in mobiDB_data:
        curated = mobiDB_data["curated-disorder-priority"]["content_count"]
    
    length = mobiDB_data["length"]

    disorder = (curated + predicted) / length

    return disorder

def human_clusters_disorder():
    hum_clustersDF = pd.read_excel(NET_INFO_FILE, sheet_name = "FuzzyHumanClusters", index_col = 0, engine='openpyxl')
    
    # HumanCluster split with fc_ and keep only the accession number
    hum_clustersDF["Protein"] = hum_clustersDF["HumanCluster"].str.split("fc_").str[1]

    hum_prots = hum_clustersDF["Protein"].unique()
    
    hum_prot_res = {}
    for idx, prot in enumerate(hum_prots):
        print(f"Processing {idx+1}/{len(hum_prots)}: {prot}")
        mobiDB_data = mobiDB(prot)
        if mobiDB_data:
            disorder = calc_disorder(mobiDB_data)
        else:
            disorder = 0
            print(f"Error processing {prot}")
        print(f"Disorder: {disorder}")
        hum_prot_res[prot] = disorder

    hum_prot_resDF = pd.DataFrame.from_dict(hum_prot_res, orient='index', columns=['Disorder'])
    hum_prot_resDF.index.name = 'Protein'
    hum_prot_resDF.to_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/hum_prot_disorder_clusters.csv")

def human_protein_disorder():
    hum_proteins_DF = pd.read_excel(NET_INFO_FILE, sheet_name = "HumanProteins", index_col = 0, engine='openpyxl')
    hum_prots = hum_proteins_DF["HumanProtein"].unique()
    
    hum_prot_res = {}
    for idx, prot in enumerate(hum_prots):
        print(f"Processing {idx+1}/{len(hum_prots)}: {prot}")
        mobiDB_data = mobiDB(prot)
        if mobiDB_data:
            disorder = calc_disorder(mobiDB_data)
        else:
            disorder = 0
            print(f"Error processing {prot}")
        print(f"Disorder: {disorder}")
        hum_prot_res[prot] = disorder

    hum_prot_resDF = pd.DataFrame.from_dict(hum_prot_res, orient='index', columns=['Disorder'])
    hum_prot_resDF.index.name = 'Protein'
    hum_prot_resDF.to_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/hum_prot_disorder.csv")

def bacteria_clusters_disorder():
    bac_clustersDF = pd.read_excel(NET_INFO_FILE, sheet_name = "FuzzyBacterialClusters", index_col = 0, engine='openpyxl')
    
    # BacteriaCluster split with fc_ and keep only the accession number
    bac_clustersDF["Protein"] = bac_clustersDF["BacterialCluster"].str.split("fc_").str[1]

    bac_prots = bac_clustersDF["Protein"].unique()
    
    bac_prot_res = {}
    for idx, prot in enumerate(bac_prots):
        print(f"Processing {idx+1}/{len(bac_prots)}: {prot}")
        mobiDB_data = mobiDB(prot)
        if mobiDB_data:
            disorder = calc_disorder(mobiDB_data)
        else:
            disorder = 0
            print(f"Error processing {prot}")
        print(f"Disorder: {disorder}")
        bac_prot_res[prot] = disorder

    bac_prot_resDF = pd.DataFrame.from_dict(bac_prot_res, orient='index', columns=['Disorder'])
    bac_prot_resDF.index.name = 'Protein'
    bac_prot_resDF.to_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bac_prot_disorder_clusters.csv")

def bacteria_protein_disorder():
    bac_proteinsDF = pd.read_excel(NET_INFO_FILE, sheet_name = "BacterialProteins", index_col = 0, engine='openpyxl')
    
    bac_prots = bac_proteinsDF["BacterialProtein"].unique()
    
    bac_prot_res = {}
    for idx, prot in enumerate(bac_prots):
        print(f"Processing {idx+1}/{len(bac_prots)}: {prot}")
        mobiDB_data = mobiDB(prot)
        if mobiDB_data:
            disorder = calc_disorder(mobiDB_data)
        else:
            disorder = 0
            print(f"Error processing {prot}")
        print(f"Disorder: {disorder}")
        bac_prot_res[prot] = disorder

    bac_prot_resDF = pd.DataFrame.from_dict(bac_prot_res, orient='index', columns=['Disorder'])
    bac_prot_resDF.index.name = 'Protein'
    bac_prot_resDF.to_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bac_prot_disorder.csv")

def histogram_cluster_disorder():
    hum_prot_resDF = pd.read_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/hum_prot_disorder_clusters.csv", index_col=0)

    # Disorder column to 0-100:
    hum_prot_resDF["Disorder"] = hum_prot_resDF["Disorder"] * 100

    fig1 = px.histogram(hum_prot_resDF, x="Disorder", nbins=30,
                       color_discrete_sequence=["#EB89B5"],  range_x=[0, 100], histnorm='percent')
    
    fig1.update_layout(
        xaxis_title="Percentage of disordered amino acids in protein sequence",
        yaxis_title="Percentage of proteins",
        bargap=0.2,
        font = dict(
            family="Times New Roman",
            size=20,
            color="black"
        ))

    fig1.update_yaxes(dtick=5)
    fig1.update_xaxes(dtick=10)

    
    # Save
    pio.write_image(fig1, os.path.join(PLOTS_DIR, "human_protein_disorder_clusters.svg"), width=800, height=700, scale=1)

    '''bac_prot_resDF = pd.read_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bac_prot_disorder_clusters.csv", index_col=0)

    # Disorder column to 0-100:
    bac_prot_resDF["Disorder"] = bac_prot_resDF["Disorder"] * 100

    fig2 = px.histogram(bac_prot_resDF, x="Disorder", nbins=30,
                       color_discrete_sequence=["#330C73"],  range_x=[0, 100], histnorm='percent')
    
    fig2.update_layout(title="Bacterial Protein Disorder",
                      xaxis_title="Percentage of disordered amino acids in protein sequence",
                      yaxis_title="Percentage of proteins",
                      bargap=0.2
                      )
    
    fig2.update_yaxes(dtick=5)
    fig2.update_xaxes(dtick=5)

    # Save
    pio.write_image(fig2, "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bacteria_cluster_disorder.png", width=800, height=600, scale=2)'''

def histogram_disorder():
    hum_prot_resDF = pd.read_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/hum_prot_disorder.csv", index_col=0)

    # Disorder column to 0-100:
    hum_prot_resDF["Disorder"] = hum_prot_resDF["Disorder"] * 100

    fig1 = px.histogram(
        hum_prot_resDF, 
        x="Disorder", 
        nbins=30,
        color_discrete_sequence=["#EB89B5"],  
        range_x=[-5, 105],  # Extend the range slightly beyond the data
        histnorm='percent'
    )

    # Add numbers to the bars
    fig1.update_traces(
        texttemplate='%{y:.1f}', 
        textposition='outside', 
        textfont_size=20
    )

    # Layout adjustments
    fig1.update_layout(
        bargap=0.2,  # Control gap between bars
        xaxis_title="Percentage of disordered amino acids in protein sequence",
        yaxis_title="Percentage of proteins",
        font=dict(
            family="Times New Roman",
            size=20,
            color="black"
        )
    )

    # Adjust axis ticks
    fig1.update_yaxes(dtick=5)
    fig1.update_xaxes(dtick=10)

    
    # Save
    pio.write_image(fig1, os.path.join(PLOTS_DIR, "human_protein_disorder.svg"), width=800, height=600, scale=1)

    '''bac_prot_resDF = pd.read_csv("/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bac_prot_disorder.csv", index_col=0)

    # Disorder column to 0-100:
    bac_prot_resDF["Disorder"] = bac_prot_resDF["Disorder"] * 100

    fig2 = px.histogram(bac_prot_resDF, x="Disorder", nbins=30,
                       color_discrete_sequence=["#330C73"],  range_x=[0, 100], histnorm='percent')
    
    fig2.update_layout(title="Bacterial Protein Disorder",
                      xaxis_title="Percentage of disordered amino acids in protein sequence",
                      yaxis_title="Percentage of proteins",
                      bargap=0.2
                      )
    
    fig2.update_yaxes(dtick=5)
    fig2.update_xaxes(dtick=5)

    # Save
    pio.write_image(fig2, "/home/c3biolab/c3biolab_projects/Gut_MB_Project/DDA/bacterial_protein_disorder.png", width=800, height=600, scale=2)'''




    
def main():
    #human_clusters_disorder()
    #bacteria_clusters_disorder()
    #histogram_cluster_disorder()
    #human_protein_disorder()
    #bacteria_protein_disorder()
    histogram_disorder()

if __name__ == '__main__':
    main()