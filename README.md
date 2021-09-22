

### deepHE

MIL task model training scripts based on: https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019


1. aws ec2 setup: ec2-setup-sos-workflow
2. whole slide preprocessing and tile-splitting: wsi-preprocessing, wsi-preprocessing-sos-workflow,. wsi-tile-cleanup
3. model training: cnn-model-training, rnn-model-training


## citation

paper: https://doi.org/10.3390/cancers12123687

```
@Article{cancers12123687,
AUTHOR = {Valieris, Renan and Amaro, Lucas and Osório, Cynthia Aparecida Bueno de Toledo and Bueno, Adriana Passos and Rosales Mitrowsky, Rafael Andres and Carraro, Dirce Maria and Nunes, Diana Noronha and Dias-Neto, Emmanuel and Silva, Israel Tojal da},
TITLE = {Deep Learning Predicts Underlying Features on Pathology Images with Therapeutic Relevance for Breast and Gastric Cancer},
JOURNAL = {Cancers},
VOLUME = {12},
YEAR = {2020},
NUMBER = {12},
ARTICLE-NUMBER = {3687},
URL = {https://www.mdpi.com/2072-6694/12/12/3687},
ISSN = {2072-6694},
ABSTRACT = {DNA repair deficiency (DRD) is an important driver of carcinogenesis and an efficient target for anti-tumor therapies to improve patient survival. Thus, detection of DRD in tumors is paramount. Currently, determination of DRD in tumors is dependent on wet-lab assays. Here we describe an efficient machine learning algorithm which can predict DRD from histopathological images. The utility of this algorithm is demonstrated with data obtained from 1445 cancer patients. Our method performs rather well when trained on breast cancer specimens with homologous recombination deficiency (HRD), AUC (area under curve) = 0.80. Results for an independent breast cancer cohort achieved an AUC = 0.70. The utility of our method was further shown by considering the detection of mismatch repair deficiency (MMRD) in gastric cancer, yielding an AUC = 0.81. Our results demonstrate the capacity of our learning-base system as a low-cost tool for DRD detection.},
DOI = {10.3390/cancers12123687}
}
```
