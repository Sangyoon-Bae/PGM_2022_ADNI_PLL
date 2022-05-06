# PGM_2022_ADNI_PLL
Final project for PGM 2022 (SNU)
This repository belongs to Stella Sangyoon Bae (stellasybae@snu.ac.kr)

## Environmental Setting for graphormer
conda create --name graphormer2 python=3.7
cd graphormer_v2
bash install.sh

## Run graphormer
conda activate graphormer2
cd graphormer_v2
bash adni-train-valid.sh
