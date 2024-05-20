conda create -n tfp-gpu -y
source activate tfp-gpu
conda install -c conda-forge cudatoolkit=11.5.0 cudnn=8.2.1.32 python=3.8.2 -y
conda install -c conda-forge tensorflow-probability=0.14.0 -y
pip install tensorflow-gpu==2.6.0 keras==2.6.0
conda install matplotlib jupyter seaborn scikit-learn -y
# These are due to versioning issues:
pip install protobuf==3.20.1
pip install numpy==1.23.4
pip install chardet
