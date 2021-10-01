# stock-ranking-using-list-wise-approach
This repository contains necessary code for the paper "Stock Ranking Prediction Using List-Wise Approach and Node Embedding Technique". The paper was published in [IEEE Access](https://ieeexplore.ieee.org/document/9461199).

We have built our codes on top of the [Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking) repository. That was our baseline. However, that repository was created using Tensorflow 1.3 version. We have updated the code for Tensorflow >2.0 versions.

# Environment
Python > 3.5, Tensorflow > 2.0, [Networkx 2.5](https://networkx.org/), [stellargraph 1.2.1](https://stellargraph.readthedocs.io/en/stable/)

# Data

Please collect the raw historical data from the [Temporal_Relational_Stock_Ranking/data/2013-01-01/](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data) folder. Please use the [pretrain](https://drive.google.com/file/d/1fyNCZ62pEItTQYEBzLwsZ9ehX_-Ai3qT/view) data to get the pretrained sequential embedding. All other relevant files can be found in the "data" folder of this repository. Please unzip the "relation.tar.gz" file to get the relation data.

# Procedure
There are two main files here. The file "stock_ranking_list_wise_tgc.py" should be used to generate the performance using temporal graph convolution (TGC) approach. The file "stock_ranking_list_wise_node2vec.py" should be used to generate the performance using Node2vec approach. Please tune the hyperparameters of Node2vec in the "graph_embedding.py" file, if required. Please change the loss function or market names to get different results.

We highly encourage to use GPU based system to run the code with full scale data.

# Contact:
If you have any query, please contact via my linkedin profile.
https://www.linkedin.com/in/suman-saha-09873a3a/
