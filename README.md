# **Setting up your Python Machine Learning system**
*Set up a Python 3 virtual environment on an Ubuntu system for doing Machine Learning / Deep Learning*

These notes assume that the reader has installed Linux, preferably an Ubuntu variant like Ubuntu-MATE, which is the best :) If you insist on using Windows, then these notes are not for you; try installing Anaconda for example, which seems to exist mostly to help Windows users do scientific/engineering coding in Python. If you are using Linux, however, there is no need for anything like Anaconda. Python 3 provides a native way to set up a virtual environment called 'venv', and for those stuck using Python 2.7, there is a package named 'virtualenv'. Below we will build Python 3.6 on our system in the /opt directory. Then we will install a virtual environment in our home directory which links to the /opt version we installed. Then we will use pip to install any package that we need into that virtual environment. Easy-peasy and we didn't have to install extra stuff like Anaconda to do hand-holding for us.  

If you are unsatisfied with this virtual environment for some reason, or you would like to replace it with an updated version, you only can simply delete the directory and no trace will remain. You could even erase the version of Python in the /opt directory. 

Below, Section 1 is only for those with a GPU; skip if you don't have one in your system and complete steps 2 through 5. The remaining sections 6 through 11 are optional. The Conclusion is simply to remind you of the ways you can invoke your new virtual Python ML setup.

## _Section 1_ - Installing CUDA & cuDNN for your GPU
If you have an NVIDIA graphics card that can support CUDA 8, then you're all set. I recently bought a relatively inexpensive card, the GTX 1050 Ti. Training neural networks on this thing is considerably faster than on my CPU. When I think that for just a couple hundred $ more I could have gotten the GTX 1070....no, I will not think about that now. If you do not have a GPU in your system, then no problem, you can still do deep learning. It's true that the extremely long times required to train on some larger datasets without a GPU will limit what projects you are willing to undertake, but you will be able to learn to do deep learning nonetheless with smaller datasets. So, if you don't have a GPU, skip ahead to the next section, but if you do, then follow the instructions below: 

Download CUDA from:  
https://developer.nvidia.com/cuda-downloads 

Install CUDA 8 per instructions at:    
http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4ZXLKBcAN  

Add this to your .bashrc file:  
`#CUDA`    
`export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}`  
`export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

Restart the system.

Download cudnn from:  
https://developer.nvidia.com/rdp/cudnn-download  

Uncompress and copy the cudnn files to appropriate directories in /usr/local/cuda-8.0/  


## _Section 2_ - Download and Build Python 3.6.1 on Your System  

Before installing Python, install the following dependencies (some of these may be installed already, in which case apt-get will inform you).

`$ sudo apt-get update`  
`$ sudo apt-get upgrade`  
`$ sudo apt-get install libbz2-dev liblzma-dev libsqlite3-dev libncurses5-dev libgdbm-dev zlib1g-dev libreadline-dev libssl-dev libssl-dev make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev cmake unzip git pkg-config gdb`  
`$ sudo apt-get install tcl-dev tk-dev python-tk python3-tk python3-tksnack`  
`$ sudo apt-get install libopenblas-dev liblapack-dev`  
`$ sudo apt-get install libgtk-3-dev`  

Make a directory in /opt to build our new Python  
`$ sudo mkdir /opt/python3.6`  
`$ sudo mkdir /opt/python3.6/lib`  

Download and uncompress Python  
 $ tar xvf Python-$RELEASE.tar.xz
 $ cd Python-$RELEASE
(NOTE: we must run configure with the --enable-share flag in order to support theano)

Configure the install with our settings  
`$ ./configure --prefix=/opt/python3.6 --enable-shared LDFLAGS="-Wl,-rpath /opt/python3.6/lib"`  
`$ make`  
Optional. Running "make test" takes a while to complete, but you might want to try it once.  
`$ make test`  
Careful on this next step. Do NOT "make install"; it will overwrite your existing Python used by your system  
`$ sudo make altinstall`    

Finished with this directory, cd back out and delete it  
`$ cd ~`  
`$ sudo rm -rf Python3.6.1`  


## _Section 3_ - Adding some aliases to .bashrc  

Add the block below to your .bashrc file. Replace '/home/username' in the block below with the path to your home directory.  
```
# PYTHON 3.6.1  
alias py36='/home/username/.ml36/bin/python3.6' # to use this Python
alias pip36='/home/username/.ml36/bin/pip3.6' # to install packages within our venv
alias jupyter-notebook_36='/home/username/.ml36/bin/jupyter-notebook' # Jupyter Notebook
alias jupyter-themer='/home/username/.ml36/bin/jupyter-themer' # to change Jupyter theme
# activate the py36 virtual environment
act_ml36 () {
  . /home/username/.ml36/bin/./activate
}

# deactivate the virtual environment
alias deact='deactivate'
# the Atom editor
alias atom36='act_ml36; atom; deactivate'

reload the .bashrc file to activate our changes  
$ source ~/.bashrc
```

## _Section 4_ - Set up the Python Virtual Environment

Next we will run our new Python from the command line with the -m flag to create the new virtual environment. I decided to call my virtual environment **.ml36**, because I do Machine Learning (ML), but call yours whatever you like. I preceded the name with a . to make it a hidden directory.  

`$ /opt/python3.6/bin/python3.6 -m venv .ml36 # to make the virtual envirionment`  

Check for yourself that the new directory is there.

`ls -la`

Was the new .ml36 directory listed?

Now let's use the first alias we created _py36_. Running this command starts our new Python 3.6.1 interpreter. This Python is used by our virtual environment. If you check in your .ml36 directory you will find a symbolic link to /opt/python3.6/bin/python3.6  

Run the py36 command and note the date and version number.  

`$ py36`  
Python 3.6.1 (default, Mar 28 2017, 22:45:00)  
[GCC 5.4.0 20160609] on linux  
Type "help", "copyright", "credits" or "license" for more information.  
\>>>   

now do this in the interpreter:

\>>> import this  
The Zen of Python, by Tim Peters  

Beautiful is better than ugly.  
Explicit is better than implicit.  
Simple is better than complex.  
Complex is better than complicated.  
Flat is better than nested.  
Sparse is better than dense.  
Readability counts.  
Special cases aren't special enough to break the rules.  
Although practicality beats purity.  
Errors should never pass silently.  
Unless explicitly silenced.  
In the face of ambiguity, refuse the temptation to guess.  
There should be one-- and preferably only one --obvious way to do it.  
Although that way may not be obvious at first unless you're Dutch.  
Now is better than never.  
Although never is often better than *right* now.  
If the implementation is hard to explain, it's a bad idea.  
If the implementation is easy to explain, it may be a good idea.  
Namespaces are one honking great idea -- let's do more of those!  
\>>>  

It works! Type Ctrl-D to exit the interpreter and go back to the shell.

 
## _Section 4_ - Installing modules we need for machine learning  

Use one of the aliases you created in your .bashrc file to activate the virtual environment.  
`$ act_ml36`  

Do you see that now each command line begins with (.ml36)? Now the virtual environment is activated. We need to do this before we install packages so that they will be installed in the right place, and therefore be usable by our virtual Python.  

Install some necessary / useful packages  
`$ pip36 install numpy scipy`  
`$ pip36 install scikit-learn`  
`$ pip36 install pillow`  
`$ pip36 install h5py`  
`$ pip36 install cython`  
`$ pip36 install numpydoc`  
`$ pip36 install scikit-image`  
`$ pip36 install nose`  
`$ pip36 install pandas`  
`$ pip36 install seaborn`  
`$ pip36 install jupyter`  
`$ pip36 install jupyter-themer`    

Install TensorFlow  
If you don't have a GPU:    
`$ pip36 install --upgrade tensorflow`  
If you have a GPU:    
`$ pip36 install --upgrade tensorflow-gpu`  

Install Keras  
`pip36 install keras`  

Now that all is installed, deactivate the virtual environment.  
`$ deact`  

Test the install  
`$ py36`  

\>>> import keras  

Is the tensorflow backend used?  

Ctrl+D to exit.   


## _Section 5_ - Get Bayesian with PyMC3
PyMC3 is a Python library for probabilistic programming. I'm working through Osvaldo Martin's book *Bayesian Analysis with Python* and am really beginning to respect the power of this Bayesian framework. Thomas Wiecki, Lead Data Scientist at Quantopian Inc., and one of the authors of PyMC3, has a blog wherein he has shown the feasibility and indeed the utility of "Bayesian Deep Learning"; it can be found here: http://twiecki.github.io.

PyMC3 uses Theano, which requires a BLAS library. First, then, install OpenBLAS like so:  
`$ cd ~/`  
`$ git clone https://github.com/xianyi/OpenBLAS`  
`$ cd OpenBLAS`  
`$ make FC=gfortran`  
`$ make PREFIX=/home/telemaque/.ml36/lib/python3.6/site-packages/OpenBLAS install`  

Now install the latest, bleeding-edge, development version of Theano with:  
`$ pip36 install --upgrade --no-deps git+git://github.com/Theano/Theano.git`  

Finally, install the development version of PyMC3:  
`$ pip36 install git+https://github.com/pymc-devs/pymc3`  


## _Section 6_ - Install Atom
In case you don't already know, Atom is a hackable text editor. When I say hackable, I mean that you can configure just about anything you want configured. It can be used to code almost any language you can think of. I'm using it now to create this markdown document.  

Download the .deb from https://atom.io/  
You can simply right-click to install it.  
Did you notice that earlier we put an alias in our .bashrc for Atom? We did this so that Atom would open using our new version of Python instead of the one installed by default on our system.  
` $ atom36`  


## _Section 7_ - XGBoost
XGBoost is a gradient boosting library. Chances are that if you stick with ML for a while, you will find yourself wanting it. Install it this way:  

`$ git clone --recursive https://github.com/dmlc/xgboost`  
`$ cd xgboost`  
`$ make`  
`$ cd python-package`  
`$ act_ml36`  
`$ py36 setup.py install`  
`$ deact` 


## _Section 8_ - Jupyter  
We installed Jupyter above. We then installed Jupyter-themer. I prefer dark themes for when I code so I use the color 'midnignt'. Go to the Jupyter-themer page and see what you might like best.  

Change the Jupyter notebook colors if you wish (OPTIONAL). I like 'midnight'  
`$ jupyter-themer_36 -c 'midnight'`  

Then start Jupyter to check that the change took effect.  
`$ jupyter-notebook_36`


## _Section 9_ - OpenCV 3   

That's most everything you'll need to get started with Machine Learning. Since I often work with sequences of images though, I also want to install OpenCV in my virtual environment. This is an optional section. It only applies to those planning to mix CV (computer vision) with their ML (machine learning).  

Install these dependencies first  
`$ sudo apt-get install build-essential cmake pkg-config`  
`$ sudo apt-get install libtiff-dev libjasper-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libjpeg-dev libpng-dev`  
`$ sudo apt-get install libatlas-base-dev gfortran`  
`$ sudo apt-get install liblapacke-dev checkinstall`  
`$ sudo apt-get install python3.6-dev`  

Get OpenCV from the Git Repository  
`$ cd ~`  
`$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip`  
`$ unzip opencv.zip`  
`$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.2.0.zip`  
`$ unzip opencv_contrib.zip`  

`$ cd opencv-3.2.0`  
`$ mkdir build`  
`$ cd build`  
`$ act_ml36`  
Since the following cmake command spans so many lines, copy and paste it into a text editor. Replace 'username' with your username. Then copy and paste that at the BASH prompt.  
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/home/username/.ml36/lib/python3.6/site-packages/opencv \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.2.0/modules \
      -D PYTHON_EXECUTABLE=/opt/python3.6/bin/python3.6 \
      -D PYTHON_INCLUDE=/opt/python3.6/include/python3.6m \
      -D PYTHON_LIBRARY=/opt/python3.6/lib/libpython3.6.so \
      -D PYTHON_PACKAGES_PATH=/home/username/.ml36/lib/python3.6/site-packages \
      -D PYTHON_NUMPY_INCLUDE_DIR=/home/username/.ml36/lib/python3.6/site-packages/numpy/core/include/numpy \
      -D BUILD_EXAMPLES=ON ..
```  
         
`$ make -j4`  
If any errors encountered, try make with only one core:  "make clean", then "make"  

`$ make install`    
`$ sudo ldconfig`    

Make symlink to OpenCV in new Python site-packages directory  
`$ cd /home/username/.ml36/lib/python3.6/site-packages`  
`$ ln -s /home/username/.ml36/lib/python3.6/site-packages/opencv/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so`  

`$ deact`  
  
 
## _Section 10_ - MongoDB and PyMongo  

This should be regarded as another purely optional section. MongoDB is a NoSQL database which has become quite popular. The PyMongo distribution contains tools for interacting with MongoDB database from Python.  

First, install MongoDB by following the few steps listed here:  
https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/  


Now install PyMongo  
`$ act_ml36`  
`$ py36 -m pip install pymongo`  
`$ deact`  


## _Section 11_ - dlib 

Developed by Davis King, the dlib C++ library is a cross-platform package for threading, networking, numerical operations, machine learning, computer vision, and compression, placing a strong emphasis on extremely high-quality and portable code.   

From a computer vision perspective, dlib has a number of state-of-the-art implementations, including:  
* Facial landmark detection
* Correlation tracking
* Deep metric learning

First, make sure the system dependencies are installed.  
`$ sudo apt-get install build-essential cmake`  
`$ sudo apt-get install libgtk-3-dev`  
`$ sudo apt-get install libboost-all-dev`  

Activate the virtual environment.  
`$ act_ml36`  

Install the following modules if not already installed.  
`$ pip install numpy`  
`$ pip install scipy`  
`$ pip install scikit-image`  

Finally, install dlib.  
`pip36 install dlib`  


## _CONCLUSION_  

We can now use three of our BASH aliases to use our new Python 3.6.1.

To run the Python interpreter in the terminal window, just type:  
`$ py36`

To run a Jupyter notebook powered by our new Python 3.6 installation, just type:  
`$ jupyter-notebook_36`  

To code in Atom with the ability to execute the code (with Ctrl+Shift+B) with our new version of Python instead of the default version on our system, just type:  
`$ atom36`  

  
If you find any error or have any questions, please open an issue and I will respond.
