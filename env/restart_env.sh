qsub -I -l select=1:ncpus=4:mem=8gb:ngpus=1:gpu_model=k40,walltime=72:00:00
module purge
module load anaconda/5.1.0 cuDNN/10.0v7.4.2 cuda-toolkit/10.0.130  openblas/0.3.5
source activate tf14gpu_py27
source activate lime_py37


