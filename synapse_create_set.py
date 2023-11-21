# %% 
import pandas as pd
from neurometry import viz, cave
from datetime import datetime

al_bodies = pd.read_csv("outputs/sample_synapse_al_bodies_11_01_2022_01:13:16.csv")
ann = pd.read_csv("outputs/sample_synapse_annotated_neurons_11_01_2022_01:13:17.csv")
spn = pd.read_csv("outputs/sample_synapse_spn_11_01_2022_14:43:22.csv")



# %%
grp_al_bodies = al_bodies.groupby("pre_pt_root_id").agg({"id":["count"], "cleft_score" : ["mean", "median"], "connection_score" : ["mean", "median"]}).reset_index()

# %%
grp_ann = ann.groupby("pre_pt_root_id").agg({"id":["count"], "cleft_score" : ["mean", "median"], "connection_score" : ["mean", "median"]}).reset_index()

# %%
alb_neurons = list(al_bodies["pre_pt_root_id"].unique())
ann_neurons = list(ann["pre_pt_root_id"].unique())


# %%
set_one = []
set_two = []
synapse_ids = set()
#set three will just be the same as spn

for x in alb_neurons:
    _syn = al_bodies[al_bodies["pre_pt_root_id"] == x]
    _syn = _syn.sample(1)
    synapse_ids.add(_syn["id"].values[0])
    set_one.append(_syn)

for x in ann_neurons:
    _syn = ann[ann["pre_pt_root_id"] == x]
    _syn = _syn.sample(1)
    synapse_ids.add(_syn["id"].values[0])
    set_two.append(_syn)

while len(set_one) != 200:
    to_add = al_bodies.sample(1)
    if to_add["id"].values[0] not in synapse_ids:
        set_one.append(to_add)

while len(set_two) != 200:
    to_add = ann.sample(1)
    if to_add["id"].values[0] not in synapse_ids:
        set_two.append(to_add)

set_one_df = pd.concat(set_one)
set_two_df = pd.concat(set_two)
set_three_df = spn

# %%
viz.histogram(set_one_df["cleft_score"], "Set I - Cleft Score Distribution").write_image("figures/set_I_cleft_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
viz.histogram(set_two_df["cleft_score"], "Set II - Cleft Score Distribution").write_image("figures/set_II_cleft_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")
# %%
viz.histogram(set_three_df["cleft_score"], "Set III - Cleft Score Distribution").write_image("figures/set_III_cleft_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%

# %%
viz.histogram(set_one_df["connection_score"], "Set I - Connection Score Distribution").write_image("figures/set_I_connection_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")
# %%
viz.histogram(set_two_df["connection_score"], "Set II - Connection Score Distribution").write_image("figures/set_II_connection_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")
# %%
viz.histogram(set_three_df["connection_score"], "Set III - Connection Score Distribution").write_image("figures/set_III_connection_scores_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
#from datetime import datetime
#al_bodies["cell_class"].value_counts().values.tolist(), "Cell Class").write_image("figures/cell_class_distribution_flyaldump_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")


# %%
set_one_df.to_csv("outputs/set_one_synapses_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)
set_two_df.to_csv("outputs/set_two_synapses_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)
set_three_df.to_csv("outputs/set_three_synapses_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
