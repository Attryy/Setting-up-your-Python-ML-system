# **Setting up your Python Machine Learning system**
*Set up a Python 3 virtual environment on an Ubuntu system for doing Machine Learning / Deep Learning*

Most of my daily work involves the processing of video sequences from different sensors (visible, infrared, polarization-filtered, etc). My work environment of preference, in fact the sole environment I work in since several years ago, is **Linux / Python / C / C++**. I began my engineering career using proprietary operating systems and programming environments since this was expected and I really didn't have a choice, but my preference for Linux to other operating systems developed early in my career when I had to administer a small network. At some point a few years ago it became clear to me that Python is the lingua-franca of the machine learning community and that, thanks to powerful packages such as SciPy, Scikit-Image, Sckit-Learn, TensorFLow, and Keras (to name but a few) all my engineering needs would be met with Python. What's more, the ability to code in C and C++ can be of great use when programming to interface with hardware using camera APIs or when creating massively parallel applications for your GPU with CUDA.  
Note: this document was originally created to aid those beginning in machine learning. However its focus has shifted a bit so that it would perhaps be better named **_Setting up your Open-Source-Powered system for machine learning, video processing, scientific programming, signal processing, etc_**  

**Linux**  
These notes assume that the reader has installed or will install Ubuntu 18.04. My favorite variant, for both personal and professional use, is Ubuntu-MATE which can be downloaded, either directly or via torrent, here:  
http://ubuntu-mate.org/download/  

**CUDA**  
For readers with a CUDA-capable GPU, there is a section on downloading and installing CUDA. Check the following link to determine if your GPU is CUDA-capable:  
https://developer.nvidia.com/cuda-gpus  


**C and C++**  
Though Ubuntu 18.04 already has these compilers installed by default, versions of GCC and G++ appropriate for CUDA 9 will be installed.    

**Python 3**  
Python 3 provides a native way to set up a virtual environment called 'venv'. Below we will build Python 3.6 on our system in the /opt directory. Then we will install a virtual environment in our home directory which links to the /opt version we installed. Then we will use pip to install most any package that we need into that virtual environment.

If you have experience using Python for ML, science, or engineering, then you might be asking "why doesn't he just use Anaconda for everything?". Answer: "I don't wanna." I want to install the packages that I need and no more. With venv and pip most of your ML needs are covered. Where any software needs to be built on your system, I show you how.  

If you are unsatisfied with this virtual environment for some reason, you can simply delete the directory and no trace will remain. You could even delete the version of Python in the /opt directory.  


## _Section 1_ - Most of what you will need

#### A. Installing CUDA & cuDNN for your GPU  
If you do NOT have a GPU in your system, then no problem, you can still do deep learning. It's true that the extremely long times required to train on some larger datasets without a GPU will limit what projects you are willing to undertake, but you will be able to learn to do deep learning nonetheless with smaller datasets.  

If you DO have an NVIDIA graphics card that can support CUDA, then you're all set perform the setup described in this section.  

The instructions are as follows:  

Add the graphics drivers repository:    
`~$ sudo add-apt-repository ppa:graphics-drivers/ppa`  
`~$ sudo apt update`    

Then install the most recent NVIDIA driver using apt:

`~$ sudo apt install nvidia-387 nvidia-387-dev`  

The driver download takes several minutes. After the driver is installed, restart the system. Then verify the installation by running:

`~$ nvidia-smi`  

You should see an output which lists the NVIDIA 387 driver and your NVIDIA GPU.

Install a number of necessary build/dev packages:  
`~$ sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev`


TensorFlow requires CUDA Toolkit v9.0 and cuDNN v7
(https://www.tensorflow.org/install/install_linux)  

Now, following (https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal):

Download  
`~$ wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb`  
This could take a while to download...  

Install  
```
~$ sudo dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb
~$ sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
~$ sudo apt-get update
~$ sudo apt-get install cuda
```

Download and install the two patches  
```
~$ wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update_1.0-1_amd64-deb`  
~$ sudo dpkg -i cuda-repo-ubuntu1704-9-0-local-cublas-performance-update_1.0-1_amd64-deb
```


```
~$ wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/2/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
~$ sudo dpkg -i cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
```

`~$ gcc -v`  
Notice that the default version of gcc on 18.04 is version 7.3.0. CUDA 9.0 requires gcc 6, so we install that.
```
~$ sudo apt install gcc-6
~$ sudo apt install g++-6
```

Having installed CUDA 9.0 we create the following links:
```
~$ sudo ln -s /usr/bin/gcc-6 /usr/local/cuda/bin/gcc
~$ sudo ln -s /usr/bin/g++-6 /usr/local/cuda/bin/g++
```

Reboot the system...,yes, again. Then run
`~$ nvidia-smi`
to verify that all is well.

To really verify that all is well with our installation, we should download the CUDA samples and run one. To download the samples, use the convenience script in /usr/local/cuda/bin like so:

`~$ bash /usr/local/cuda/bin/cuda-install-samples-9.0.sh ~/`

Test one of the samples:
```
~$ cd ~/NVIDIA_CUDA-9.0_Samples/5_Simulations/smokeParticles/
~$ make
~$ ./smokeParticles
```

You should see a pretty smoke thingy.

Download cuDNN 7 from
https://developer.nvidia.com/rdp/cudnn-download
(requires login)  
Uncompress and copy to the appropriate dirs (include & lib64) in /usr/local/cuda  

Add the following to your .bashrc  

```
# The PATH variable needs to include /usr/local/cuda-9.0/bin
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
# In addition, the LD_LIBRARY_PATH variable needs to be set
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
Since we now have two versions of GCC and of G++ on our system, I will place the following lines in my .bashrc file to ensure that the default versions are the newer ones (version 7):  
```  
# make this the default gcc
alias gcc='/usr/bin/gcc-7'
alias g++='/usr/bin/g++-7'
```  

#### B. Download and Build Python 3.6.5 on Your System  

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

Download Python from:
https://www.python.org/downloads/  
Uncompress and cd into the directory  
`$ tar xvf Python-$RELEASE.tar.xz`  
`$ cd Python-$RELEASE`  

Configure the install with our settings  
(NOTE: we must run configure with the --enable-share flag in order to support theano)  
`$ ./configure --prefix=/opt/python3.6 --enable-shared LDFLAGS="-Wl,-rpath /opt/python3.6/lib"`  
`$ make`  
Optional. Running "make test" takes a while to complete, but you might want to try it once.  
`$ make test`  
Careful on this next step, be sure to use "altinstall". Do NOT "make install"; it will overwrite your existing Python used by your operating system  
`$ sudo make altinstall`      

Finished with this directory, cd back out and delete it  
`$ cd ~`  
`$ sudo rm -rf Python3.6.3`  


#### C. Adding some aliases to .bashrc  

Add the block below to your .bashrc file. Replace '/home/username' in the block below with the path to your home directory. Note that I call my Python "py36". I also have Python 3.4 installed, which I invoke with "py34", to be clear which version I am using. If you will only use the one version, then you could simply call it "py3" for example.  
```
# PYTHON 3.6.5  
alias py36='/home/username/.ml36/bin/python3.6' # to use this Python
alias pip36='/home/username/.ml36/bin/pip3.6' # to install packages within our venv
alias jupyter-notebook_36='/home/username/.ml36/bin/jupyter-notebook' # Jupyter Notebook
# activate the py36 virtual environment
act_ml36 () {
  . /home/username/.ml36/bin/./activate
}
# deactivate the virtual environment
alias deact='deactivate'
# the Atom editor
alias atom36='act_ml36; atom; deactivate'
```
reload the .bashrc file to activate our changes  
`$ source ~/.bashrc`  


#### D. Set up the Python Virtual Environment  

Next we will run our new Python from the command line with the -m flag to create the new virtual environment. I decided to call my virtual environment **.ml36**, for Machine Learning (ML) + Python 3.6, but call yours whatever you like. I preceded the name with a . to make it a hidden directory.  

`$ /opt/python3.6/bin/python3.6 -m venv .ml36`    

Check for yourself that the new directory is there.

`$ ls -la`

Was the new .ml36 directory listed?

Now let's use the first alias we created _py36_. Running this command starts our new Python 3.6.3 interpreter. This Python is used by our virtual environment. If you check in your .ml36 directory you will find a symbolic link to /opt/python3.6/bin/python3.6  

Run the py36 command and note the date and version number.  

`$ py36`  
Python 3.6.3 (default, Nov 22 2017, 23:23:32)
[GCC 7.2.0] on linux
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


#### E. Installing modules we need for machine learning  

Install some necessary / useful packages using pip   
`$ pip36 install numpy scipy`  
`$ pip36 install scikit-image`  
`$ pip36 install scikit-learn`    
`$ pip36 install h5py`        
`$ pip36 install jupyter`  
`$ pip36 install pygame`    

Install TensorFlow  
If you do NOT have a GPU:    
`$ pip36 install tensorflow`  
If you DO have a GPU and have followed the steps in section 1. A.,:      
`$ pip36 install tensorflow-gpu`  

Install Keras  
`pip36 install keras`   

Test the install  
`$ py36`  

At the interpreter prompt, type  
\>>> import keras  

Is the tensorflow backend used?  

Ctrl+D to exit.   

We can now use three of our BASH aliases to use our new Python 3.6.5. To run the Python interpreter in the terminal window, just type:  
`~$ py36`

To run a Jupyter notebook powered by our new Python 3.6 installation, just type:  
`~$ jupyter-notebook_36`  

If have installed the Atom editor (https://atom.io/), you can execute the code (with Ctrl+Shift+B) with our new version of Python instead of the default version on our system. Do this by opening the editor with the alias:    
`~$ atom36`  

## _Optional_ - PyQtGraph  

Until recently, I have been satisfied with displaying video using custom Matplotlib functions. Lately though, it has become clear that I need to be able to display large images at faster framerates. For example, a customer requires live image display from their dual-camera sensor. Initial investigations point to PyQtGraph as the best and most Pythonic solution. I hope to have some updates on this in the near future. Here is how to install the dependencies and the package itself:  

Install Qt5:
```
~$ sudo apt-get install python3-pyqt5
~$ sudo apt-get install pyqt5-dev-tools
~$ sudo apt-get install qttools5-dev-tools
```

Aside:  
  Qt5 can be run from the terminal using qtchooser:
  `~$ qtchooser -run-tool=designer -qt=5`  
  or, you can set Qt5 as the default Qt by writing the following in /usr/lib/x86_64-linux-gnu/qt-default/qtchooser/default.conf:  
  /usr/lib/x86_64-linux-gnu/qt5/bin  
  /usr/lib/x86_64-linux-gnu  

Install pyqt5:  
`~$ pip36 install pyqt5`

Install PyQtGraph:  
`~$ pip36 install pyqtgraph`  

Some functionality in PyQt5 requires OpenGL + pyopengl:  
install OpenGL libraries:  
`~$ sudo apt-get install mesa-utils`  

install freeGlut:  
`~$ sudo apt-get install freeglut3-dev`  

install pyopengl:  
`~$ pip36 install pyopengl`  

Test it:
```
~$ py36
>>> import pyqtgraph.examples
>>> pyqtgraph.examples.run()
```

## _Section 2_ - Computer Vision (CV)

#### A. Install OpenCV
Many of my computer vision needs are met by NumPy & SciKit-Image, but I sometimes want to use certain libraries from OpenCv. Let's build it on our system.     

First, update packages:
```
~$ sudo apt-get update
~$ sudo apt-get upgrade
```

Install these dependencies (see what is required and what is optional here:  
https://docs.opencv.org/3.4.1/d7/d9f/tutorial_linux_install.html)  

```
~$ sudo apt-get install build-essential
~$ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
~$ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
~$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
~$ sudo apt-get install libxvidcore-dev libx264-dev
~$ sudo apt-get install libgtk-3-dev
~$ sudo apt-get install libatlas-base-dev gfortran pylint
~$ sudo apt-get install ffmpeg
~$ sudo apt-get install python3.6-dev
```
Download OpenCV 3.4.0 and OpenCV Contrib
```
~$ wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
~$ wget https://github.com/opencv/opencv_contrib/archive/3.4.0.zip -O opencv_contrib-3.4.0.zip
```

Extract
```
~$ unzip opencv-3.4.0.zip
~$ unzip opencv_contrib-3.4.0.zip
```

Create build dir  
```
~$ cd  opencv-3.4.0
~$ mkdir build
~$ cd build
```

Create the opencv dir in your python site-packages dir  
`~$ mkdir /home/telemaque/.ml36/lib/python3.6/site-packages/opencv`

~$ act_ml36

Run this in terminal, replacing 'username' with the name of your home directory.  
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/home/telemaque/.ml36/lib/python3.6/site-packages/opencv \
-D OPENCV_EXTRA_MODULES_PATH=/home/telemaque/opencv_contrib-3.4.0/modules \
-D PYTHON_EXECUTABLE=/opt/python3.6/bin/python3.6 \
-D PYTHON_INCLUDE=/opt/python3.6/include/python3.6m \
-D PYTHON_LIBRARY=/opt/python3.6/lib/libpython3.6.so \
-D PYTHON_PACKAGES_PATH=/home/telemaque/.ml36/lib/python3.6/site-packages \
-D PYTHON_NUMPY_INCLUDE_DIR=/home/telemaque/.ml36/lib/python3.6/site-packages/numpy/core/include/numpy \
-D WITH_CUDA=OFF \
-D BUILD_EXAMPLES=ON ..
```

Find out number of CPU cores in your machine  
`~$ nproc`  
You can run make with multiple cores with the -j flag. I'll use all but one of mine:  
`~$ make -j7`  
This will take a while... study Japanese... maybe clean the attic...  

```
$ make install
$ sudo ldconfig
```

Make symlink to OpenCV in new Python site-packages directory
```
$ cd /home/username/.ml36/lib/python3.6/site-packages
$ ln -s opencv/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
```

Test now whether the shared library can be imported in Python:      
`~$ py36`  
\>>> import cv2  
\>>> cv2.\__version__  

If the result is '3.4.0', then all is well. If the library cannot be imported, then you must repeat the "make" and "make install" steps.

`~$ deact`  



#### B. Invoking this OpenCV from C++ code  

Many examples using CUDA to parallelize image-processing code is written in C++ and uses OpenCV functions. These examples assume that OpenCV is installed to the default directory under /usr/local, however we have installed ours under our virtual Python directory .ml36.   
At this point you might be asking yourself, "why didn't he just install OpenCV to the default directory and then put a symlink in the appropriate virtual Python lib directory?". This might be the best way to do it...the truth is that in the past I have never used OpenCV except with Python and saw no reason for a system-wide install. Now that I am using OpenCV with C++ and NVCC, I will consider changing to a system-wide install of OpenCV. For now though, let's carry on.   

Below are three things to do to ensure that our C++ code will work with OpenCV.  

At compile time, the linker **ld** needs to know where to find the shared library paths. We can set these manually in /etc/ld.so.conf.d  
`~$ cd /etc/ld.so.conf.d`  
`~$ sudo touch opencv.conf`  
Add the following line to the file:  
/home/username/.ml36/lib/python3.6/site-packages/opencv/lib  
Save the file, then apply the change with  
`~$ sudo ldconfig`  

We can also add the OpenCV library path to our .bashrc file. Add this line:  
`export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/username/.ml36/lib/python3.6/site-packages/opencv/lib`  

The next place we might need to specify paths to opencv is in our makefiles. Here is a Makefile from the first homework of Udacity's Intro To Parallel Programming, with explicit paths to our libraries and our header files:

```  
NVCC=nvcc

OPENCV_LIBPATH=/home/telemaque/.ml36/lib/python3.6/site-packages/opencv/lib
OPENCV_INCLUDEPATH=/home/telemaque/.ml36/lib/python3.6/site-packages/opencv/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda-9.0/include

NVCC_OPTS=-O3 -arch=sm_61 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

student: main.o student_func.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW1 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

student_func.o: student_func.cu utils.h
	nvcc -c student_func.cu $(NVCC_OPTS)

compare.o: compare.cpp compare.h
	g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o *.png
```  

(note: My NVIDIA card is of the Pascal architecture type, so I have set arch=sm_61 under the NVCC options)  



## _CONCLUSION_  

If you find any error or have any questions, please open an issue and I will respond.
