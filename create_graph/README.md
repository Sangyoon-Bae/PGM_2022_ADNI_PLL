## data folder
```
- adni_connectome_aparc_count.mat: edge data (connectivity)
- ADNI-179_subjects-label-mor: subject list, label (diagnosis result), brain morphometry
- DTI - sMRI.csv: util file for generating nodes
```

## generate graph in pickle form
```
- code: [create_graph]01.ADNI_build_structural_graph.ipynb
- result: /ADNI_structural_graph_count_threshold_20
```

## generate graph in OGB form
*(especially, .csv.gz file in the 'raw' folder)*
```
- code: [create_graph]02.ADNI_build_csv_for_graphormer.ipynb
- result: ../graphormer_v2/dataset/adni_struct_count/raw
```
