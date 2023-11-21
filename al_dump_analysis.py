#from neurometry import viz

#viz.scatter()

# %%
from neurometry import viz, cave
import pandas as pd

al_bodies = pd.read_csv("data/AL_dump_Oct_24_2022.csv")

# %%
from datetime import datetime
# %%
#viz.donut(al_bodies["cell_class"].value_counts().index.tolist(), 
#al_bodies["cell_class"].value_counts().values.tolist(), "Cell Class").write_image("figures/cell_class_distribution_flyaldump_"+datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".png")

# %%
al_bodies["class+type"] = al_bodies["cell_class"] + " " + al_bodies["cell_type"]

# %%
client = cave.set_up()

# %%
meta = cave.get_cellinfo_meta(client)

# %%
al_ids = list(al_bodies["root_id"].unique())
annotations = cave.cellidentity_query(client, al_ids)

# %%
# %%
annotations_grouped = annotations.groupby(["pt_root_id", "tag"])["pt_root_id"].agg(["count"]).sort_values(by="count", ascending=False)
annotations_grouped = annotations_grouped.reset_index()

# %%
ann_dump = client.materialize.query_table('neuron_information_v2')

# %%
tags_grouped = ann_dump.groupby(["tag"])["id"].agg(["count"]).sort_values(by="count", ascending=False)
tags_grouped = tags_grouped.reset_index()

# %%
annotations_merged = al_bodies.merge(annotations_grouped, how="left", left_on="root_id", right_on="pt_root_id")

# %%
#annotations_merged.to_csv("outputs/annotations_merged_al_bodied_flywire_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
import re
def separation(row):
    text = row["tag"]
    pattern = r"[^a-zA-Z']"
    text = re.sub(pattern, ' ', text.lower())
    #text = text.lower()
    text = text.split()
    return text



# %%
ann_dump

# %%
ann_dump["tag_processed"] = ann_dump.apply(separation, axis=1)

# %%
bag_of_words = {}
for index, row in ann_dump.iterrows():
    _tags = row["tag_processed"]
    for tag in _tags:
        if tag in bag_of_words:
            bag_of_words[tag] += 1
        else:
            bag_of_words[tag] = 1

# %%
bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient='index', columns=["frequency"]).reset_index()

# %%
bag_of_words_df_sorted = bag_of_words_df.sort_values(by="frequency", ascending=False)

# %%
bag_of_words_df_alphabetical = bag_of_words_df.sort_values(by="index")

# %%
#bag_of_words_df_alphabetical.to_csv("outputs/annotations_tags_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
ann_dump[ann_dump["tag_processed"].apply(lambda x: 'mbp' in x)]

# %%
annotation_tags = pd.read_csv("data/flywire_annotation_tags.csv")

# %%
important_tags = annotation_tags[annotation_tags["important?"]==1].reset_index()[["index","frequency","important?"]]

# %%
keywords = list(important_tags["index"])

# %%
remove_list = ["alln", "alpn", "antennal", "sensory"]
for x in remove_list:
    keywords.remove(x)

# %%
len(keywords)

# %%
# Sample 200 synapses from AL_bodies
sampke_al_bodies = al_bodies.sample(n=200)

# %%
sampke_al_bodies.to_csv("outputs/al_bodies_200_neurons_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
chosen = set()
while len(chosen) < 200:
    the_word = pd.Series(keywords).sample().values[0]
    the_sample = ann_dump[ann_dump["tag_processed"].apply(lambda x: the_word in x)]
    the_sample = the_sample.sample(1)
    root_id = the_sample["pt_root_id"].values[0]
    chosen.add(root_id)

# %%
sample_ann_dump = ann_dump[ann_dump["pt_root_id"].isin(chosen)]

# %%
sample_ann_dump.to_csv("outputs/sample_ann_200_neurons_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)
# %%
s_bag_of_words = {}
for index, row in sample_ann_dump.iterrows():
    _tags = row["tag_processed"]
    for tag in _tags:
        if tag in s_bag_of_words:
            s_bag_of_words[tag] += 1
        else:
            s_bag_of_words[tag] = 1

# %%
s_bag_of_words_df = pd.DataFrame.from_dict(s_bag_of_words, orient='index', columns=["frequency"]).reset_index()

# %%
s_bag_of_words_df_sorted = s_bag_of_words_df.sort_values(by="frequency", ascending=False)

# %%
s_bag_of_words_df_alphabetical = s_bag_of_words_df.sort_values(by="index")

# %%
s_bag_of_words_df_alphabetical.to_csv("outputs/sample_ann_200_neurons_annotations_tags_" + datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+".csv", index=False)

# %%
