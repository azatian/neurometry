# %%
import pandas as pd
from datetime import datetime
from neurometry import viz, cave

set_one = pd.read_csv("outputs/set_one_synapses_11_02_2022_14:20:45.csv")
set_two = pd.read_csv("outputs/set_two_synapses_11_02_2022_14:20:45.csv")
set_three = pd.read_csv("outputs/set_three_synapses_11_02_2022_14:20:45.csv")



# %%
#viz.nt_heatmap(pd.concat([set_one, set_two, set_three]), 9).write_image("figures/nt_heatmap_600_synapses_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")
#fig, sort = viz.nt_heatmap(pd.concat([set_one, set_two, set_three]), 9)
# %%
total = pd.concat([set_one, set_two, set_three])

# %%
client = cave.set_up()

# %%
list_synapse_ids = list(total["id"].unique())
synapse_neuropil= cave.neuropil_query_by_id(client, list_synapse_ids)

# %%
synapse_neuropil["neuropil"].value_counts()

# %%
neuropils = {"AL_R" : "Antennal Lobe Right",
            "AL_L" : "Antennal Lobe Left",
            "GNG" : "Gnathal Ganglia",
            "ME_R" : "Medulla Right",
            "LO_L" : "Lobula Left",
            "FB" : "Fan Shaped Body",
            "LH_L" : "Lateral Horn Left",
            "ME_L" : "Medulla Left",
            "SMP_R" : "Superior Medial Protocerebrum Right",
            "SLP_L" : "Superior Lateral Protocerebrum Left",
            "SLP_R" : "Superior Lateral Protocerebrum Right",
            "LH_R" : "Lateral Horn Right",
            "LO_R" : "Lobula Right",
            "AVLP_L" : "Anterior Ventrolateral Protocerebrum Left",
            "AVLP_R" : "Anterior Ventrolateral Protocerebrum Right",
            "SPS_R" : "Superior Posterior Slope Right",
            "PLP_R" : "Posteriolateral Protocerebrum Right",
            "IPS_R" : "Inferior Posterior Slope Right",
            "PLP_L" : "Posteriolateral Protocerebrum Left",
            "MB_ML_R" : "Mushroom Body Medial Lobe Right",
            "SAD" : "Saddle",
            "PRW" : "Prow",
            "FLA_L" : "Flange Left",
            "SMP_L" : "Superior Medial Protocerebrum Left",
            "LAL_R" : "Lateral Accessory Lobe Right",
            "VES_L" : "Vest Left",
            "MB_ML_L" : "Mushroom Body Medial Lobe Left",
            "LOP_R" : "Lobula Plate Right",
            "MB_CA_R" : "Mushroom Body Calyx Right",
            "IPS_L" : "Inferior Posterior Slope Left",
            "WED_L" : "Wedge Left",
            "SCL_R" : "Superior Clamp Right",
            "SIP_R" : "Superior Intermediate Protocerebrum Right",
            "ICL_L" : "Inferior Clamp Left",
            "PVLP_L" : "Posterior Ventrolateral Protocerebrum Left",
            "VES_R" : "Vest Right",
            "LAL_L" : "Lateral Accessory Lobe Left",
            "SPS_L" : "Superior Posterior Slope Left",
            "EB" : "Ellipsoid Body",
            "MB_CA_L" : "Mushroom Body Calyx Left",
            "PVLP_R" : "Posterior Ventrolateral Protocerebrum Right",
            "CRE_L" : "Crepine Left",
            "CRE_R" : "Crepine Right",
            "GOR_R" : "Gorget Right",
            "ICL_R" : "Inferior Clamp Right",
            "SIP_L" : "Superior Intermediate Protocerebrum Left",
            "MB_VL_R" : "Mushroom Body Vertical Lobe Right",
            "AOTU_L" : "Anterior Optic Tubercle Left",
            "LOP_L" : "Lobula Plate Left",
            "MB_PED_R" : "Mushroom Body Pedunculus Right",
            "WED_R" : "Wedge Right",
            "FLA_R" : "Flange Right",
            "PB" : "Protocerebral Bridge",
            "MB_VL_L" : "Mushroom Body Vertical Lobe Left",
            "ATL_R" : "Antler Right",
            "CAN_L" : "Cantle Left",
            "GOR_L" : "Gorget Left",
            "ATL_L" : "Antler Left"}

regions=    {"Antennal Lobe Right" : "Antennal Lobe",
            "Antennal Lobe Left" : "Antennal Lobe",
            "Gnathal Ganglia" : "Gnathal Ganglia",
            "Medulla Right" : "Optic Lobe",
            "Lobula Left" : "Optic Lobe",
            "Fan Shaped Body" : "Central Complex",
            "Lateral Horn Left" : "Lateral Horn",
            "Medulla Left" : "Optic Lobe",
            "Superior Medial Protocerebrum Right" : "Superior Neuropils",
            "Superior Lateral Protocerebrum Left" : "Superior Neuropils",
            "Superior Lateral Protocerebrum Right" : "Superior Neuropils",
            "Lateral Horn Right" : "Lateral Horn",
            "Lobula Right" : "Optic Lobe",
            "Anterior Ventrolateral Protocerebrum Left" : "Ventrolateral Neuropils",
            "Anterior Ventrolateral Protocerebrum Right" : "Ventrolateral Neuropils",
            "Superior Posterior Slope Right" : "Ventromedial Neuropils",
            "Posteriolateral Protocerebrum Right" : "Ventrolateral Neuropils",
            "Inferior Posterior Slope Right" : "Ventromedial Neuropils",
            "Posteriolateral Protocerebrum Left" : "Ventrolateral Neuropils",
            "Mushroom Body Medial Lobe Right" : "Mushroom Body", 
            "Saddle" : "Periesophageal Neuropils",
            "Prow" : "Periesophageal Neuropils",
            "Flange Left" : "Periesophageal Neuropils",
            "Superior Medial Protocerebrum Left" : "Superior Neuropils",
            "Lateral Accessory Lobe Right" : "Lateral Complex",
            "Vest Left" : "Ventromedial Neuropils",
            "Mushroom Body Medial Lobe Left" : "Mushroom Body",
            "Lobula Plate Right" : "Optic Lobe",
            "Mushroom Body Calyx Right" : "Mushroom Body",
            "Inferior Posterior Slope Left" : "Ventromedial Neuropils",
            "Wedge Left" : "Ventrolateral Neuropils",
            "Superior Clamp Right" : "Inferior Neuropils",
            "Superior Intermediate Protocerebrum Right" : "Superior Neuropils",
            "Inferior Clamp Left" : "Inferior Neuropils",
            "Posterior Ventrolateral Protocerebrum Left" : "Ventrolateral Neuropils",
            "Vest Right" : "Ventromedial Neuropils",
            "Lateral Accessory Lobe Left" : "Lateral Complex",
            "Superior Posterior Slope Left" : "Ventromedial Neuropils",
            "Ellipsoid Body" : "Central Complex",
            "Mushroom Body Calyx Left" : "Mushroom Body",
             "Posterior Ventrolateral Protocerebrum Right" : "Ventrolateral Neuropils",
            "Crepine Left" : "Inferior Neuropils",
            "Crepine Right" : "Inferior Neuropils",
            "Gorget Right" : "Ventromedial Neuropils",
            "Inferior Clamp Right": "Inferior Neuropils",
            "Superior Intermediate Protocerebrum Left" : "Superior Neuropils",
            "Mushroom Body Vertical Lobe Right" : "Mushroom Body",
            "Anterior Optic Tubercle Left" : "Ventrolateral Neuropils",
            "Lobula Plate Left" : "Optic Lobe",
            "Mushroom Body Pedunculus Right" : "Mushroom Body",
            "Wedge Right" : "Ventrolateral Neuropils",
            "Flange Right" : "Periesophageal Neuropils",
            "Protocerebral Bridge" : "Central Complex",
           "Mushroom Body Vertical Lobe Left" : "Mushroom Body",
           "Antler Right" : "Inferior Neuropils",
            "Cantle Left" : "Periesophageal Neuropils",
            "Gorget Left" : "Ventromedial Neuropils",
             "Antler Left" : "Inferior Neuropils"}

# %%
regions_df = pd.DataFrame.from_dict(regions, orient="index", columns=["region"]).reset_index()
# %%
neuropil_df = pd.DataFrame.from_dict(neuropils, orient="index", columns=["neuropil"]).reset_index()
# %%
merged = neuropil_df.merge(regions_df, how="inner", left_on="neuropil", right_on="index")

# %%
# %%
final_df = synapse_neuropil.merge(merged, how="left", left_on="neuropil", right_on="index_x")
# %%
region_group = final_df.groupby(["region"])["id"].agg(["count"]).reset_index().sort_values(by="count", ascending=False)
neuropil_group = final_df.groupby(["neuropil_y"])["id"].agg(["count"]).reset_index().sort_values(by="count", ascending=False)

# %%
import plotly.graph_objects as go
def bar_chart_example(x, y, title):
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
    fig.update_layout(title_text=title, xaxis_tickangle=-45,
    plot_bgcolor = "rgba(0, 0, 0, 0)",
paper_bgcolor = "rgba(0, 0, 0, 0)")
    return fig



# %%
bar_chart_example(region_group["region"], region_group["count"], "Region Representation").write_image("figures/region_representation_bar_chart_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
bar_chart_example(neuropil_group["neuropil_y"], neuropil_group["count"], "Neuropil Representation").write_image("figures/neuropil_representation_bar_chart_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
