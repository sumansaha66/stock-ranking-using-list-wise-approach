# stock-ranking-using-list-wise-approach
This repository contains necessary code for the paper "Stock Ranking Prediction Using List-Wise Approach and Node Embedding Technique". The paper was published in [IEEE Access](https://ieeexplore.ieee.org/document/9461199).

We have built our codes on top of the [Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) repository. That was our baseline. However, that repository was created using Tensorflow 1.3 version. We have updated the code for Tensorflow >2.0 versions.

# Environment
Python > 3.5 & Tensorflow > 2.0

# Procedure
There are two main files here. The file "stock_ranking_list_wise_tgc.py" should be used to generate the performance using temporal graph convolution (TGC) approach. Please change the loss function or market names to get different results.
