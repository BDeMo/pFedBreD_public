# Overview

The code of pFedBreD accepted by NeurIPS 2023. We provide 3 implementations of pFedBreD under Jaynes's Rule, pFedBreD_ns
and 3 baselines in the main Table of comparative experiments.

## Abstract

Classical federated learning (FL) enables training machine learning models without sharing data for privacy
preservation, but heterogeneous data characteristic degrades the performance of the localized model. Personalized FL (
PFL) addresses this by synthesizing personalized models from a global model via training on local data. Such a global
model may overlook the specific information that the clients have been sampled. In this paper, we propose a novel scheme
to inject personalized prior knowledge into the global model in each client, which attempts to mitigate the introduced
incomplete information problem in PFL. At the heart of our proposed approach is a framework, the PFL with Bregman
Divergence (pFedBreD), decoupling the personalized prior from the local objective function regularized by Bregman
divergence for greater adaptability in personalized scenarios. We also relax the mirror descent (RMD) to extract the
prior explicitly to provide optional strategies...

# Preprint

[Arxiv Preprint](https://arxiv.org/pdf/2310.09183.pdf)

[Openreview]()(Not available now)

![image](https://pic1.zhimg.com/80/v2-9b97060db9eb6db321312074d6a81ad4_720w.webp)

# Others About Paper

Poster:

![Poster](https://pic2.zhimg.com/v2-501463aa4d958506f46d1c9a10091045_r.jpg)

[Zhihu](https://zhuanlan.zhihu.com/p/661506638/edit)

# Tutorial

## Requirements and Dataset

To install requirements:

> pip install -r requirements.txt

1. Run data/*/generate_... first, if you have no data yet.
2. Then try the default hyperparameter setting on what ever --totalepoch you want by running main_fl.

More scripts will be released soon.

PS: if there's any wired characters (e.g., fo/mfo/mg) are respectively the old name and unrelated (or additional)
methods(e.g., of lg/meg/mh).

## Optional Baselines

[FedAvg](http://proceedings.mlr.press/v54/mcmahan17a.html)
[FedProx](https://arxiv.org/abs/1812.06127)
[pFedMe](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf)
[Per-FedAvg](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html)
[FedEM](https://arxiv.org/abs/2108.10252)
[FedAMP](https://ojs.aaai.org/index.php/AAAI/article/view/16960)
[pFedBayes](https://proceedings.mlr.press/v162/zhang22o.html)
[Ditto](https://proceedings.mlr.press/v139/li21h.html)
[Fedfomo](https://openreview.net/forum?id=ehJqJQk9cw)
[FedHN](http://proceedings.mlr.press/v139/shamsian21a.html)
[FedPAC](https://arxiv.org/abs/2306.11867)

## Citation

If you use our code or wish to refer to our results, please use the following BibTex entry:

@inproceedings{shi2023prior, title={PRIOR: Personalized Prior for Reactivating the Information Overlooked in Federated
Learning.}, author = {Mingjia Shi, Yuhao Zhou, Kai Wang, Huaizheng Zhang, Shudong Huang, Qing Ye, Jiancheng Lv},
boottitle = {Proceedings of the 37th NeurIPS}, year = {2023} }