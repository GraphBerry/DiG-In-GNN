## DiG-In-GNN

The official implemantation of AAAI-24 paper **DiG-In-GNN: Discriminative Feature Guided GNN-based Fraud Detector against Inconsistencies in Multi-Relation Fraud Graph**. [Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/28785)

The folder `Code4DiGInGNN` in this repo contains the code version we submitted to AAAI as the supplementary file, and there is a `README.md` included in it. This version runs slowly on T-Finance dataset because it uses many `for` loops in the neighbor selector, instead of faster tensor operations that can be accelerated by parallel computing on cuda devices. We have written a faster version and will share it after the author is through with their busy graduation season.

The codes are based on [PC-GCN](https://github.com/PonderLY/PC-GNN).

If you have any problems or trouble, you can write an issue to let us know.
