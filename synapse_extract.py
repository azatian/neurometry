# %%
from neurometry import viz, cave
import pandas as pd
from datetime import datetime



# %%
al_bodies_neurons = pd.read_csv("outputs/al_bodies_200_neurons_10_31_2022_21:46:17.csv")
sample_annotations = pd.read_csv("outputs/sample_ann_200_neurons_10_31_2022_21:48:15.csv")

# %%
al_bodies_neurons_list = list(al_bodies_neurons["root_id"])
sample_annotations_list = list(set(list(sample_annotations["pt_root_id"])))

# %%
client = cave.set_up()

# %%
al_bodies_synapse = []
al_bodies_not_included = []
for x in al_bodies_neurons_list:
    syn_df = cave.synapse_query(client, [x])
    syn_df = cave.synapse_strict_filters(syn_df)
    if len(syn_df) < 10:
        al_bodies_not_included.append(x)
    else:
        al_bodies_synapse.append(syn_df.sample(10))
    


 # %%
al_bodies_synapses_all = pd.concat(al_bodies_synapse)

# %%
annotations_neurons = []
annotations_neurons_not_included = []
for x in sample_annotations_list:
    syn_df = cave.synapse_query(client, [x])
    syn_df = cave.synapse_strict_filters(syn_df)
    if len(syn_df) < 10:
        annotations_neurons_not_included.append(x)
    else:
        annotations_neurons.append(syn_df.sample(10))



# %%
annotations_synapses_all = pd.concat(annotations_neurons)

# %%
#s_bag_of_words_df_alphabetical.to_csv("outputs/sample_ann_200_neurons_annotations_tags_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)
al_bodies_synapses_all.to_csv("outputs/sample_synapse_al_bodies_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)
annotations_synapses_all.to_csv("outputs/sample_synapse_annotated_neurons_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
# From Stephan: 244358226 links
import random
spn = []
counter = 0
while len(spn) < 200:
    #choose a random number between 0 and 244,358,226
    _id = random.randint(0, 244358226)
    #call the cave function
    syn_df = cave.synapse_query_by_id(client, [_id])
    #call the standard filtering
    syn_df = cave.synapse_strict_filters(syn_df)
    #check if not empty
    if len(syn_df) != 0:
        spn.append(syn_df)
    
    counter += 1
    #add to spn

# %%
spn_synapses = pd.concat(spn)

# %%
spn_synapses.to_csv("outputs/sample_synapse_spn_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
