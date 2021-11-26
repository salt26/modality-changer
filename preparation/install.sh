sudo apt-get -y install build-essential libasound2-dev libjack-dev

export CUDA=cu101 # may change depending on your cuda version # check it by nvcc --version
pip install torch==1.5.0+{CUDA} torchvision==0.6.0+{CUDA} -f https://download.pytorch.org/whl/torch_stable.html
pip install numba==0.48.0
pip install gast==0.3.3
pip install tensorboard==2.3.0
pip install tensorflow
pip install pypianoroll==0.5.3
pip install sklearn tqdm music21 magenta mido tslearn matplotlib

# add PYTHONPATH to your bashrc (or zshrc, ...)
