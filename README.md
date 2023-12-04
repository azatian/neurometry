# neurometry
auxiliary package to interface with navis and fafbseg

## Main Operations

**neurometry/cave.py** 
  
  - How to interact with the CAVEClient to:
      - extract cutouts from FAFB (Flywire)
      - access predicted synapse table
      - access predicted neurotransmitter per synapse
      - access cell identity table
      - filter said tables

**neurometry/ic.py**

  - How to interact with the imageryclient to:
      - access segmentation layer
      - access raw image
      - create masks
      - plot accessed data and overlays

**neurometry/ml.py**

  - How to create an example CNN architecture
  - How to use a image augmentation library

**neutrometry/viz.py**
  
  - Helper functions to plot pie charts, scattergrams, bar charts, histograms, heatmaps


