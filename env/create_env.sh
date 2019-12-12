qsub -I -l select=1:ncpus=4:mem=50gb:ngpus=1:gpu_model=k40,walltime=72:00:00
module load anaconda/5.1.0 cuDNN/10.0v7.4.2 cuda-toolkit/10.0.130  openblas/0.3.5
conda create -n lemna_py27 python=2.7
source activate lemna_py27
pip install tensorflow-gpu
pip install keras
conda install theano pygpu
conda install -c conda-forge rpy2
conda install -c conda-forge r-genlasso
conda install -c conda-forge r-gsubfn
pip install pandas
pip install pydot
pip install lxml
pip install gensim
pip install scikit-learn
pip install matplotlib
conda install jupyter
python -m ipykernel install --user --name yuxinc --display-name "lemna_py27"