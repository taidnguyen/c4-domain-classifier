#!/bin/bash

# HF_TOKEN=your_hf_token

experiments=()
experiments+=("google/embeddinggemma-300m|5e-4|32|--focal")
experiments+=("google/embeddinggemma-300m|5e-4|32|")
experiments+=("google/embeddinggemma-300m|2e-4|32|--focal")
experiments+=("google/embeddinggemma-300m|2e-4|32|")
experiments+=("google/embeddinggemma-300m|2e-4|64|--focal")
experiments+=("google/embeddinggemma-300m|5e-4|64|--focal")
experiments+=("all-mpnet-base-v2|2e-4|32|--focal")
experiments+=("all-mpnet-base-v2|2e-4|32|")

for experiment in "${experiments[@]}"; do
    IFS='|' read -r model lr bs focal <<< "$experiment"
    echo "Running experiment $exp_num: $model (lr=$lr, bs=$bs, focal=$focal)"
    HF_TOKEN=${HF_TOKEN} python3 trainer.py --embedding_model "$model" --lr "$lr" --bs "$bs" $focal
    
done

echo "All experiments completed"