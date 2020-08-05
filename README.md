# GCN-HIV
code for JAMI paper "Network context matters: graph convolutional network model over social networks improves the detection of unknown HIV infections among young men who have sex with men"

## Introduction
See the JAMIA paper for details, to detect risk individuals in a social network.  

## Data
Need a social network stored using adjacency matrix and a matrix of individual variables  
    social network: refer_social_sex  
    individual variables: venue_attribute  

## Method
- gcn: src_gnn  
- gat: src_gnn  
- lr/rf: src_convent  
- gcn/gat + rf: src_ensemble  

## Implementation
### Packages
Python3.6+, Tensorflow1.14+ and corresponding versions of scipy, scikit-learn, numpy, 

## Parameters
- Under each folder, there are 10 json format config files (in folder configs/) corresponding to the 10 folds in cross-validation, make sure to adjust parameters simultaneously for all the 10 files (manually at present but you can change to a generator or define arguments or define them in the main python file).  
- In patient_loader.py under each folder (in folder data_loader/), you can also set REDUCE_GRAPH_FEATURES as True or False to enable hiv positive rate and syphilis positive rate of neighborhood as features.

## Run
The script will automatically run all the folds. (Make sure to make the results/ folder first)
sh SCRIPT.sh -> e.g. sh run_chicago_sex.sh
The results will be generated in the results/ folder in about 10-20 mins for different models.

## Get mean and std of AUC
The running script will output the optimal value for each fold at the final line.
python summarize_results.py PATH_OF_RESULTS -> e.g. python summarize_results.py results/chicago_gat_sex/

## Notes
For running models on the sex or venue network, for example in src_gnn/models/gat.py, you have to switch the definition of logits to the corresponding one (e.g. bias_mat=self.bias_in_sex or self.bias_in_venue). You need also change the output directory in the run_XX.sh file to specify the output directory.  

## Cite
@article{xiang2019network,  
  title={Network context matters: graph convolutional network model over social networks improves the detection of unknown HIV infections among young men who have sex with men},  
  author={Xiang, Yang and Fujimoto, Kayo and Schneider, John and Jia, Yuxi and Zhi, Degui and Tao, Cui},  
  journal={Journal of the American Medical Informatics Association},  
  volume={26},  
  number={11},  
  pages={1263--1271},  
  year={2019},  
  publisher={Oxford University Press}  
}  
