# DeepMCL-DTI
 DeepMCL-DTI: predicting drug-target interactions using multi-channel deep learning with attention mechanism

	A novel multi-channel deep learning DTI prediction model named DeepMCL-DTI based on attention. 
	We firstly perform four channels to extract drug and protein features. 
	A novel interact-attention module is used to model the semantic interdependence of each drug-target pair in spatial and channel dimensions.
	DeepMCL-DTI exhibits improved performance compared with several state-of-the-art methods.
	The case study on COVID-19 verifies the practical potential of DeepMCL-DTI.



## The environment of DeepMCL-DTI
    Linux OS
    python 3.8.12 
    pytorch                1.10.2 

## Run the DeepMCL-DTI model for DTI prediction
### Preprocess the data.
The expermental data can be found in this [link](https://github.com/zhaoqichang/HpyerAttentionDTI).


    $ python DrugBank_data.py
    $ python Davis_data.py


## Acknowledgments
1. We really thank Mehdi et al. open the source code of AttentionSiteDTI at this [link](https://github.com/yazdanimehdi/AttentionSiteDTI). The AttentionSiteDTI help us to preprocess drug data.

2. We really thank Yunan Zhao et al. open the dataset in this papaer "Zhao Q, Zhao H, Zheng K, et al. HyperAttentionDTI: improving drug–protein interaction prediction by sequence-based deep learning with attention mechanism. Bioinformatics 2022; 38:655–662"

We will Keep updating the data and code of this research project.
  The code of this study is refer to the previous research results of [Mehdi et al.](https://github.com/yazdanimehdi/AttentionSiteDTI)
  and [Zhao et al.](https://github.com/zhaoqichang/HpyerAttentionDTI)
