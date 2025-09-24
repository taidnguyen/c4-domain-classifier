# Build a News Classifier with Text Embeddings

We are tasked with training a text classifier to predict news documents. The classifier can be potentially used for filtering pretraining data for LLMs, and the test set is imbalanced (90-10% split) to mimic a realistic setup.

With the goal of correctly curating News data, we want to optimize for metrics that prioritize true positives, or at the very least samples that might look very similar to news. This is a priority over false positives, which are low-quality non-news samples. Thus, metrics such as precision, average_precision, and F0.5 are used in this exercise.

Our best results use Google's `gemma-300m` embeddings, which significantly outperformed other small-scale (less than 500M) embedding models tried. The classifier is a simple 3-layer MLP neural network with dropouts.

The best experiment uses the Focal Loss, a modification of the cross-entropy loss that downweighs the contribution of "easy" examples and up-weights the contribution of "hard" examples. Hard examples might come from many samples in C4, where the text can look extremely similar to News (and could even be considered news from personal examination).

## Results

| Model | Parameter | Focal Loss | News Precision | F1 | F0.5 | F0.5 Optimized | Link |
|-------|-----------|------------|----------------|----|----- |----------------|------|
| all-mpnet-base-v2 | 109m | False | 0.519 | 0.645 | 0.563 | 0.824 | [link](https://wandb.ai/taidnguyen/ccnews-classification/runs/j58r8esr?nw=nwusertaidnguyen) |
| all-mpnet-base-v2 | 109m | True | 0.522 | 0.652 | 0.567 | 0.83 | [link](https://wandb.ai/taidnguyen/ccnews-classification/runs/awm16q4z?nw=nwusertaidnguyen) |
| google/embeddinggemma | 300m | False | 0.814 | 0.87 | 0.836 | 0.932 | [link](https://wandb.ai/taidnguyen/ccnews-classification/runs/qmqol3pz?nw=nwusertaidnguyen) |
| google/embeddinggemma | 300m | True | 0.829 | 0.876 | 0.847 | 0.97 | [link](https://wandb.ai/taidnguyen/ccnews-classification/runs/x6t8regc?nw=nwusertaidnguyen) |

### Notes 
- Results are reported on the test set.
- Train: n=6000 (balanced)
- Validation: n=3588 (30 CCNews - 70 C4)
- Test: n=3588 (10 CCNews - 90 C4)
- The validation set is used to tune hyperparameters and decide an "optimal" decision threshold for the binary classifier.
- Compared to F1, F0.5 assigns more importance to precision.
- F0.5 Optimized is calculated with the tuned decision threshold.

### Classification report for last row

Binary threshold = 0.9524

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Non-News | 0.97 | 1.00 | 0.98 | 2700 |
| News | 0.97 | 0.80 | 0.88 | 370 |
| Macro Avg | 0.97 | 0.90 | 0.93 | 3070 |
| Weighted Avg | 0.97 | 0.97 | 0.97 | 3070 |

### Findings
- **google/embeddinggemma-300m with Focal Loss** achieved the highest F0.5 optimized score of **0.97**.
- This model also has decent precision (0.829) and F1 (0.876) scores
- EmbeddingGemma-300m significantly outperformed all-mpnet-base-v2 across all metrics
- Focal loss generally improved performance, especially for EmbeddingGemma
- EmbeddingGemma models achieved >0.92 F0.5 optimizeds, indicating excellent precision when the binary threshold is tuned.

## Resources
We used Pytorch Lightning to train and WandB to log all results.

All training code lives in `trainer.py`. Experiments can be replicated by running:
```
bash launch.sh
```

Full Wandb training log: https://wandb.ai/taidnguyen/ccnews-classification?nw=nwusertaidnguyen

## Limitations + Future improvements
- Our classifier requires very high confidence (95.24%) to classify something as News. While it is fine to avoid the false positives, we could also potentially lose out on some valuable documents. Using ensemble methods and agreement metrics between many classifiers can potentially improve the calibration.
- We work with the frozen embedding models out of the box in this exercise, but finetuning them for this specific task can potentially yield performance gains.
