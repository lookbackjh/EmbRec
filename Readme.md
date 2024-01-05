# Code for Simple and effective recommendations using implicit feedback-aware factorization machines

## 0. Overview

- Our paper mainly proposes **SVD embedding** for factorization machines (FM/DeepFM)
- Followed  Most of the DeepFM/FM code structure from [https://github.com/rixwew/pytorch-fm](https://github.com/rixwew/pytorch-fm)

## 1. Basic Requirements.

- Code Tested on cuda 11.8 and python 3.10.11~12
- pip Install requirements.txt

## 2. How to Use

**run `new_test.py`**

- To try SVD embedding, make `args.embedding_type="SVD"`
- To change model options(FM/DeepFM) change `args.embedding_type`
- Various hyperparamters  to test can be changed based on your preference.
- datasets: other more datasets were used but here we upload ml100k and ml1m

## 3. Algorithms

- SVD Embedding applied FM/DeepFM can be seen in `model/SVD_emb` folder
- How we implemented negative sampling can be seen in `util/negativesampler.py`
- Our  frequency-based negative sampling (abbreviated FNS in paper, ) can be used by making `args.isuniform=False`