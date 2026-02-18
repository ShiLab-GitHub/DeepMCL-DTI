Paper: ["DeepMCL-DTI: predicting drug-target interactions using multi-channel deep learning with attention mechanism"] (https://doi.org/10.1007/s11030-025-11402-4)  published in Molecular Diversity: https://link.springer.com/article/10.1007/s11030-025-11402-4

# DeepMCL-DTI
A novel multi-channel deep learning DTI prediction model named DeepMCL-DTI based on attention. We firstly perform four channels to extract drug and protein features. A novel interact-attention module is used to model the semantic interdependence of each drug-target pair in spatial and channel dimensions. DeepMCL-DTI exhibits improved performance compared with several state-of-the-art methods. The case study on COVID-19 verifies the practical potential of DeepMCL-DTI.



## The environment of DeepMCL-DTI
    Linux OS
    python 3.8.12 
    pytorch                1.10.2 

## Run the DeepMCL-DTI model for DTI prediction
### Preprocess the data.
    $ python DTI_data.py
    $ python main.py


## Acknowledgments
1. We sincerely acknowledge Mehdi et al. for making the source code of AttentionSiteDTI publicly available[link](https://github.com/yazdanimehdi/AttentionSiteDTI). The AttentionSiteDTI open-source resource was invaluable for our drug data preprocessing workflow, laying a critical foundation for this study.

2. We also express our gratitude to Yunan Zhao and colleagues for sharing the dataset used in their work: Zhao Q, Zhao H, Zheng K, et al. HyperAttentionDTI: improving drug–protein interaction prediction by sequence-based deep learning with attention mechanism. Bioinformatics 2022; 38:655–662. This dataset provided essential support for our experimental design and validation.
 
3. The code developed in this study partially references the code employed in previous research by [Mehdi et al.](https://github.com/yazdanimehdi/AttentionSiteDTI)
  and [Zhao et al.](https://github.com/zhaoqichang/HpyerAttentionDTI).

We will continue to update the data and code associated with this research project and make them publicly available to facilitate reproducibility and further advancements in the field.
