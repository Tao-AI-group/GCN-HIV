# GCN-HIV
code for JAMI paper "Network context matters: graph convolutional network model over social networks improves the detection of unknown HIV infections among young men who have sex with men"

* Introduction
All gnn models for prediction HIV (or syphilis) status in Houston and Chicago's networks (both social(sex) and venue).

* Projects
See my jamia and aids paper.  
combine gnn and statistics for syphilis/hiv prediction, by re-calculating exposure variables.  
transfer learning and domain adaptation.  

* Data
hiv_pure_data/202007  
    social (sex) network: refer_social_sex  
    venue network: venue_attribute  
    individual variables: venue_attribute  

* Method
- gcn: src_gnn  
- gat: src_gnn  
- lr/rf: src_convent  
- gcn/gat + rf: src_ensemble  
- gcn/gat + lr + rf: src_ensemble    
- gat fusion1: src_gat_fusion1  
- gat fusion2: src_gat_fusion1

* Implementation
** Packages
scipy, scikit-learn, numpy, tensorflow1

** Parameters
- You can change parameters and definitely you have to change parameters to get the optimal results.  
- You can modify the parameter adjustment by wrapping it up with Bayesian optimization.  
- Under each folder, there are 10 json format config files (in folder configs/) corresponding to the 10 folds in cross-validation, make sure to adjust parameters simultaneously for all the 10 files (manually at present but you can change to a generator or define arguments).  
- For example, I defined an argument (-t/--threshold) as a demo, i.e. you can specify the venue threshold by using -t 20.  
- In patient_loader.py under each folder (in folder data_loader/), you can also set REDUCE_GRAPH_FEATURES as True or False to enable hiv positive rate and syphilis positive rate of neighborhood as features.

** Run
The script will automatically run all the folds.
sh SCRIPT.sh -> e.g. sh run_chicago_sex.sh
The results will be generated in the results/ folder in about 10-40 mins for different models.

** Get mean and std of AUC
The running script will output the optimal value for each fold at the final line.
python summarize_results.py PATH_OF_RESULTS -> e.g. python summarize_results.py results/chicago_gat_sex/

* Tips
For running models on the sex or venue network, for example in src_gnn/models/gat.py, you have to switch the definition of logits to the corresponding one (e.g. bias_mat=self.bias_in_sex or self.bias_in_venue). You need also change the output directory in the run_XX.sh file to specify the output directory.  

