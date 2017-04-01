# **Setting up your Python Machine Learning system**
*Set up a Python 3 virtual environment on an Ubuntu system for doing Machine Learning / Deep Learning*

These notes assume that the reader is has just installed Linux, preferably an Ubuntu variant like Ubuntu-MATE, which is the best :) If you insist on using Windows, then these notes are not for you; try installing Anaconda for example, which seems to exist mostly to help Windows users do scientific/engineering coding in Python. If you are using Linux, however, there is no need for anything like Anaconda. Python 3 provides a native way to set up a virtual environment called 'venv', and for those stuck using Python 2.7, there is a package named 'virtualenv'. Below we will build Python 3.6 on our system in the /opt directory where it will be out of the way and we won't have to think about it anymore. Then we will install a virtual envirionment in our home directory which links to the /opt version we installed. Then we will use pip to install any package that we need into that virtual environment. Easy-peasy and we didn't have to install extra stuff like Anaconda to do hand-holding for us.

# _Section 1_ - Installing CUDA & cuDNN for your GPU
If you have an NVIDIA graphics card that can support CUDA 8, then you're all set. I recently bought a relatively inexpensive card, the GTX 1050 Ti. Training neural networks on this thing is considerably faster than on my GPU. When I think that for just a couple hundred $ more I could have gotten the GTX 1070....no, I will not think about that now. If you do not have a GPU in your system, then no problem, you can still do deep learning. It's true that the extremely long times required to train on some larger datasets without a GPU will limit what projects you are willing to undertake, but you will be able to learn to do deep learning on your nonetheless with smaller datasets. So, if you don't have a GPU, skip ahead to the next section, but if you do, then follow the instructions below: 

+ 
Download CUDA from:  
https://developer.nvidia.com/cuda-downloads 

+ 
Install CUDA 8 per instructions at:    
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4ZXLKBcAN  

Add this to your .bashrc file:  
`#CUDA`    
`export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}`  
`export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

Restart the system.
+ 
Download cudnn from:  
https://developer.nvidia.com/rdp/cudnn-download  

Uncompress and copy the cudnn files to appropriate directories in /usr/local/cuda-8.0/  


# _Section 2_ - Download and Build Python 3.6.1 on Your System  

before installing Python, install the following dependencies  


# _Section 3_ - Set up the Python Virtual Environment  

ksdjfkla
