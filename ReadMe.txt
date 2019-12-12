CPSC-8580 Fall 2019 

Yuxin Cui

Team Project: LEMNA: Explaining Deep Learning based Security Applications

My Part: Apply LEMNA to Explain IDS

Contents:
    It's a Python2.7 program

Two sample models and datasets are given. covtype dataset, MNIST dataset and their natural and robust models.

1. Enviroment: 
		Clemson Palmetto, 24+ cores, 100+ gb mem.

              	Module: Anaconda3/5.1.0

               	Python: 3.7.4

               	Packages needed: xgboost, pandas, numpy, scipy, sklearn, mlxtend, subprocess, time, pickle, xlrd, adversarial-robustness-toolbox
               	
		In folder 'env', we give .sh files to help setting up enviroment. 
		May need other packages, please install as required.



2. Copy py file folder_create.py to the folder created for this project. 
		Run the file to create home folder 'cai_cui' and all needed folders.



3. cd cai_cui



4. Install robust_xgboost package from Github using Unix command line:

               	git clone --recursive https://github.com/chenhongge/RobustTrees.git

               	cd RobustTrees
               
		./build.sh
    (it will build multi-thread xgboost)           
		make



5. Copy all code files in submitted folder Code to /cai_cui/codes.
               


6. Download datasets, save them in the folder /cai_cui/data      
	From LIBSVM Data:Classification (Binary Class) we can download datasets: 
		breast-cancer, cod-rna, diabetes, ijcnn1, and webspam.
                  
		link  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.
 
             
	Down load Sensorless dataset from UCI. 
		https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis.
              
	
	Download 4 data files of MNIST dataset and save in folder '/cai_cui/data/MNIST_orig' for dataset MNIST
		from http://yann.lecun.com/exdb/mnist/.
              
	Download 2 .csv data files of Fashion-MNIST dataset and save in folder '/cai_cui/data/Fashion_MNIST_orig' for dataset Fashion-MNIST 
		from Kaggle. https://www.kaggle.com/zalando-research/fashionmnist.
              
	           
	We include the dataset diabetes as the sample dataset in submitted folder 'sample datasets'.



7. Normalize data and split them into train set and test set:
              
		cd /cai_cui/codes
              
		python norm_split_driver.py
              
		python MNIST_norm.py
              
		python webspam_norm.py
              
		python ijcnn_norm_split.py
              
	After they are all completed, we will find all training and test sets are ready in folder /data. 
	The format is pickle.



8. Natual xgboost models:
              
	Stay in folder /codes
              
		python xgb_model_driver.py
              
	After it completes, we will find all xgb models are ready in folder models/xgb. Models are saved in .model format.

	To check the test set accuracy, run
              
		python xgb_accu.py
              
	The accuracies are stored in folder /test_accu/xgb_test_accu.txt. 
	Also, the time used to build models are stored in it too.
              
	This program also selects and stores samples for attack. The sample set is stored in folder /chosen_sample.



9. Robust xgboost models:
              
	Stay in folder /codes
              
		python rxgb_prep_driver.py
              
	After it completes, we will find all robust xgb models are ready in folder models/rxgb/dataset_name. 
	Models are saved in .model format.
              
	This program also outputs a file rxgb_used_time.txt, which records the time used to tune models. It is folder /test_accu.
              
	To check the test set accuracy, run
              
		python rxgb_test_sampling.py
              
	The accuracies are stored in folder /test_accu/rxgb_test_accu.txt. 
	The chosen sample sets are in folder /chosen_sample



10. Cheng's attack:
              
	Attack natual xgboost models:
                 
		python cheng_attack_driver.py
                 
	For each dataset, we have a file datasetname_cheng_attack_xgb.txt, which has the distance, original points and points after perturbation. 
                 
	We also output the average distance to file datasetname_cheng_xgb_ave.txt
              
	
	Attack robust xgboost models:
                 
		python cheng_attack_rxgb_driver.py
                 
	For each dataset, we have a file datasetname_cheng_attack_rxgb.txt, which has the distance, original points and points after perturbation. 
	We also output the average distance to file datasetname_cheng_rxgb_ave.txt

	The attack commands for all datasets are in run_Cattack.sh



11. Kantchelian's attack:
              
	For example, attack MNIST2_6 xgb model:
              
		python xgbKantchelianAttack.py -dn=mnist2_6 -mn=rxgb -d../../data/binary_mnist0.t -m=../../report3_models/rxgb/binary_mnist_robust/1000.model -c=2

	 -dn means dataset name, -mn means model name(xgb or rxgb), -d means sample data path, -m means model path, -c means number of class, we write result to path ../Kattack_result/mnist2_6_Kan_xgb_ave.txt

	The attack commands for all datasets are in run_Kattack.sh
             

12. HopSkipJump attack:
	For example, attack MNIST xgb model:

		python HSJ_attack.py mnist xgb ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784

	Attack MNIST rxgb model:

		python HSJ_attack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784

	The first parameter means dataset name(mnist), the second parameter means model name(xgb ot rxgb), the third parameter means test data path, the forth means model path, the fifth means number of class(10), the last means number of features(784). We write result fo path ../Hattack_result/mnist_HSJ_xgb_ave.txt

	The attack commands for all datasets are in run_Hattack.sh

13. Chen's attack:
	For example, attack breast_cancer xgb model:

		./treeVerify example.json > ../paper_result/treeVer_result/xgb/breast_cancer_xgb_treeVer_log.txt

	In example.json, we set:
	{   
    		"inputs":       "../data/breast_cancer_scale0.test", 
    		"model":        "../paper_models/xgb/breast_cancer_unrobust/0004.json",
    		"start_idx":    0,
    		"num_attack":   200,
   		 "eps_init":     0.3,
    		"max_clique":   2,
    		"max_search":   10,
    		"max_level":    1,
    		"num_classes":  2
	}

	"inputs" means test data path, "model" means model path, "start_idx" means index of the first point to evaluate, "num_attack" number of point to be evaluated, "eps_init" means the first epsilon in the binary search, "max_clique" means maximum number of nodes in a clique,  "max_level" means maximum number of levels of clique search, "num_classes" means number of classes.

	The example.json and attack commands for all datasets are in run_treeVerify.sh
	
14. Boundary attack:
	For example, attack MNIST xgb model:

		python Boundary_attack.py mnist xgb ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784

	Attack MNIST rxgb model:

		python Boundary_attack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784
	
	The first parameter means dataset name(mnist), the second parameter means model name(xgb ot rxgb), the third parameter means test data path, the forth means model path, the fifth means number of class(10), the last means number of features(784). We write result fo path ../Battack_result/mnist_Boundary_xgb_ave.txt

	The attack commands for all datasets are in run_Battack.sh

15. ZOO attack:
	For example, attack MNIST xgb model:

		python zoo_attack.py mnist xgb ../../data/ori_mnist.test0 ../../report3_models/xgb/ori_mnist_unrobust_new/0200.model 10 784

	Attack MNIST rxgb model:

		python zoo_attack.py mnist rxgb ../../data/ori_mnist.test0 ../../report3_models/rxgb/ori_mnist_robust_new/0200.model 10 784

	The first parameter means dataset name(mnist), the second parameter means model name(xgb ot rxgb), the third parameter means test data path, the forth means model path, the fifth means number of class(10), the last means number of features(784). 