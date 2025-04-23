# MAGLLM
## Overview
This repository contains the code for **MAGLLM**, our proposed approach for personalized review recommendation as described in the paper **Graph Meets LLM for Review Personalization Based on User Votes** which available [here]([https://dl.acm.org/doi/pdf/10.1145/3696410.3714691]). MAGLLM leverages heterogeneous graphs to model relationships among users, reviews, and products, enriched with LLM to generate user profiles. Unlike traditional approaches that rely on authored reviews, we use **review votes** to enhance personalization.

Our experiments demonstrate that the voting signal significantly improves recommendation performance compared to authorship-based approaches across various recommendation models and e-commerce domains.

## Installation
Clone the repository:

`git clone [https://github.com/your-repo/MAGLLM.git](https://github.com/lilahz/MAGLLM.git)`

`cd MAGLLM`

Install dependencies:

`pip install -r requirements.txt`

## Running Experiments
To train MAGLLM, run:
`python main.py --category 'Beauty' --signal 'vote'`  
In case of first run, need to pre-process the data, therefore `--preprocess True` needs to be added.  
Other wise the preprocess can be run separetly:
`python preprocess.py --category 'Beauty' --signal 'vote'`

Other options:
--category: DVDs, Beauty, Food & Drink, Internet, Games
--signal: vote, write, both

## Dataset
We use the Ciao dataset. The data include two files:
1. ciao_reviews.tsv - all the reviews in the dataset after preprocessing
2. votes_4core.tsv - the review votes of users after the 4-core filtering, including the split to train, validation and test

Data can be downloaded from: https://drive.google.com/drive/u/1/folders/1hGNzK0abv45O-dOF29NzXYkMhEUfYqc5 

## Citation
If you use this code, please cite our paper:

```
@inproceedings{hirsch2025graph,
  author    = {Sharon Hirsch and Lilach Zitnitski and Slava Novgorodov and Ido Guy and Bracha Shapira},
  title     = {Graph Meets LLM for Review Personalization Based on User Votes},
  booktitle = {Proceedings of the ACM Web Conference 2025 (WWW '25)},
  year      = {2025}
}
```
