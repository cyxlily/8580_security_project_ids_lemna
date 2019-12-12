# 8580_security_project_ids_lemna
use lemna to explain ids rnn model


  
CPSC-8580 Fall 2019 

Yuxin Cui

Team Project: LEMNA: Explaining Deep Learning based Security Applications

My Contribution: Apply LEMNA to Explain IDS

## Table of Contents

- [Contents](#contents)
- [Environment](#environment)
- [Usage](#usage)



## Contents:
It's a Python2.7 program. It can select key features of a IDS RNN model using LEMNA explaination algorithm.


## Environment: 
Clemson Palmetto, 1 gpu, 50 gb mem.

Module: Anaconda3/5.1.0 cuDNN/10.0v7.4.2 cuda-toolkit/10.0.130 openblas/0.3.5

Python: 2.7.15

Packages needed: tensorflow-gpu, keras, theano, pygpu, rpy2, r-genlasso, r-gsubfn, pandas, pydot, lxml, gensim, scikit-learn, matplotlib, jupyter

In folder 'env', we give creat_env.sh files to help setting up enviroment. 
    
May need other packages, just run "pip install package_name" if required.



## Usage:
* download the project using Unix command line:
       
        $ git clone https://github.com/cyxlily/8580_security_project_ids_lemna.git
       
        $ cd 8580_security_project_ids_lemna


* The processed ids dataset is in the folder 'data', including X_test_binary.csv ,Y_test_binary.csv and unique_tokens.csv. The trained model is in the folder 'model',  named binary_protobytes_dirty.h5. The codes is in the folder 'code'.


* You can run LEMNA using Unix command line:
       
        $ cd code
        
        $ python lemna_replication.py --f=5


* To run codes automatically, you can using command line:
    
        $ qsub lemnaids5.pbs
        
Remember to change the root file path.


* You can train the IDS model yourself. IDS related codes are in the folder 'IDS'. You can train the IDS model using command line:
        
        $ cd IDS
        
        $ python train_binary.py

The trained model is in the folder 'models',named binary_protobytes_dirty.h5


* You can test the IDS model using command line:
        
        $ python test_binary.py
       
The evaluation results is report.txt, report.csv, val.png


* You can also process the datasets yourself. First download the raw datasets in the folder 'data'. The share link is 
    
        https://drive.google.com/open?id=1pi3SluBlec_g1SexyYDPM7HMm5Z6y4uh
    
Unzip the raw dataset using command line:
        
        $ unzip labeled_flows_xml.zip
        
Process the raw datasets by usign command line:
       
        $ python generate_binary.py
        
The datasets are X_train_binary.csv, X_test_binary.csv, Y_train_binary.csv, Y_test_binary.csv, unique_tokens.csv
