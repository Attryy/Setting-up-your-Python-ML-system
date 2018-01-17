# **Setting up your Python Machine Learning system**
*Set up a Python 3 virtual environment on an Ubuntu system for doing Machine Learning / Deep Learning*

Most of my daily work involves the processing of video sequences from different sensors (visible, infrared, polarization-filtered, etc). My preferred work environment, in fact the sole environment I work in since several years ago, is **Linux / Python / C / C++**. I began my engineering career using proprietary operating systems and programming environments since this was expected and I really didn't have a choice, but my preference for Linux to other operating systems developed early in my career when I had to administer a small network. At some point a few years ago it became clear to me that Python is the lingua-franca of the machine learning community and that, thanks to powerful packages such as SciPy, Scikit-Image, Sckit-Learn, TensorFLow, and Keras (to name but a few) all my engineering needs would be met with Python. What's more, the ability to code in C and C++ can be of great use when programming to interface with hardware using camera APIs or when creating massively parallel applications for your GPU with CUDA.  
Note: this document was originally created to aid those beginning in machine learning. However its focus has shifted a bit so that it would perhaps be better named **_Setting up your Open-Source-Powered system for ML, DL, video processing, scientific programing, signal processing, etc_**  

**Linux**  
These notes assume that the reader has or will install Ubuntu 17.10. My favorite variant, for both home and professional use, is Ubuntu-MATE which can be downloaded, either directly or via torrent, here:  
http://ubuntu-mate.org/download/  

**CUDA**  
For readers with a CUDA-capable GPU, there is a section on downloading and installing CUDA. Check the following link to determine if your GPU is CUDA-capable:  
https://developer.nvidia.com/cuda-gpus  


**C and C++**  
GCC and G++ will be installed prior to installing CUDA.    

**Python 3**  
Python 3 provides a native way to set up a virtual environment called 'venv'. Below we will build Python 3.6.3 on our system in the /opt directory. Then we will install a virtual environment in our home directory which links to the /opt version we installed. Then we will use pip to install most any package that we need into that virtual environment.

If you have experience using Python for ML, science, or engineering, then you might be asking "why doesn't he just use Anaconda for everything?". Answer: "I don't wanna." I want to install the packages that I need and no more. With venv and pip most of your ML needs are covered. Where any software needs to be built on your system, I show you how.  

If you are unsatisfied with this virtual environment for some reason, you can simply delete the directory and no trace will remain. You could even delete the version of Python in the /opt directory.  


## _Section 1_ - Most of what you will need

#### A. Installing CUDA & cuDNN for your GPU  
If you do NOT have a GPU in your system, then no problem, you can still do deep learning. It's true that the extremely long times required to train on some larger datasets without a GPU will limit what projects you are willing to undertake, but you will be able to learn to do deep learning nonetheless with smaller datasets.  

If you DO have an NVIDIA graphics card that can support CUDA 8, then you're all set perform the setup described in this section.

I installed Ubuntu-MATE 17.10 and followed the instructions for installing CUDA 9 (the latest) as described in this excellent post:  
https://askubuntu.com/questions/967332/how-can-i-install-cuda-9-on-ubuntu-17-10  

The install was successful and I was able to run the NVIDIA example code. It wasn't until I had set up my Python environment and installed TensorFlow that I encountered problems.  

**Problem 1.** Tensorflow 1.4 (current version) requires CUDA 8 and cuDNN 6. A quote from a TensorFlow post: "We anticipate releasing TensorFlow 1.5 with CUDA 9 and cuDNN 7".

**Problem 2.** CUDA 8 requires GCC 5 instead of GCC 6 (CUDA 9 requires GCC 6; the Ubuntu 17.10 default is GCC 7).

**Problem 3.** The pip binary for the latest version of TensorFlow (1.4) was built with Python 3.5 and gives import warnings.

My solution to the above problems is this:
On a freshly-installed Ubuntu 17.10 system, install **GCC/G++ 5**. Next install **CUDA 8** and **cudDNN 6**. Next install **the latest Python (which is version 3.6.3)**. Finally, install **TensorFlow 1.3**. When TensorFlow 1.5 is released in a few months, I will upgrade to GCC/G++ 6 and CUDA 9.  

The instructions are as follows:  

Add the graphics drivers repository:    
`~$ sudo add-apt-repository ppa:graphics-drivers/ppa`  
`~$ sudo apt update`    

Then install the most recent NVIDIA driver using apt:

`~$ sudo apt install nvidia-384 nvidia-384-dev`  

The driver download takes several minutes. After the driver is installed, restart the system. Then verify the installation by running:

`~$ nvidia-smi`  

You should see an output which lists the NVIDIA 384 driver and your NVIDIA GPU.

Install a number of necessary build/dev packages:  
`~$ sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev`  

The default gcc/g++ version for Ubuntu 17.10 is 7.2.0 (Ubuntu 7.2.0-8ubuntu3). This can be checked by:  
`~$ gcc -v`  

CUDA 8 requires gcc 5. ("gcc versions later than 5 are not supported!")  
Install it (and set the corresponding sym-links after the cuda installation):  
`~$ sudo apt install gcc-5`  
`~$ sudo apt install g++-5`  

download and install CUDA Toolkit 8 and cudnn 6 for use with TensorFlow 1.3  

download the latest version 8 CUDA Toolkit (Debian package)  
`~$ wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb`  

Install the .deb package.  
`~$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb`  

`$ sudo apt-get update`  
`$ sudo apt-get install cuda`  

download the latest update:  
`~$ wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb`  

Install the .deb package   
`~$ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb`  

Make those sym-links mentioned earlier:  
`~$ sudo ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc`  
`~$ sudo ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++`  

Install the samples using the convenience installation script:  
`~$ cd /usr/local/cuda/bin`  
`~$ bash cuda-install-samples-8.0.sh ~/`  

Test one of the samples:  
`~$ cd ~/NVIDIA_CUDA-8.0_Samples/5_Simulations/smokeParticles`  
`~$ make`  
`~$ ./smokeParticles`  

You'll see a pretty smoke thingy.  

Download cuDNN 6 from  
https://developer.nvidia.com/rdp/cudnn-download  
(requires login)  

Uncompress and copy to the appropriate dirs (include & lib64) in /usr/local/cuda  


Add the following to your .bashrc  

```
# The PATH variable needs to include /usr/local/cuda-8.0/bin
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
# In addition, the LD_LIBRARY_PATH variable needs to be set
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```  

#### B. Download and Build Python 3.6.3 on Your System  

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
# PYTHON 3.6.3  
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
```
reload the .bashrc file to activate our changes  
`$ source ~/.bashrc`  


#### D. Set up the Python Virtual Environment  

Next we will run our new Python from the command line with the -m flag to create the new virtual environment. I decided to call my virtual environment **.ml36**, because I do Machine Learning (ML), but call yours whatever you like. I preceded the name with a . to make it a hidden directory.  

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

Use one of the aliases you created in your .bashrc file to activate the virtual environment.  
`$ act_ml36`  

Do you see that now each command line begins with (.ml36)? Now the virtual environment is activated. We need to do this before we install packages so that they will be installed in the right place, and therefore be usable by our virtual Python.  

Install some necessary / useful packages  
`$ pip36 install numpy scipy`  
`$ pip36 install scikit-image`
`$ pip36 install scikit-learn`    
`$ pip36 install h5py`        
`$ pip36 install jupyter`  
`$ pip36 install jupyter-themer`    

Install TensorFlow  
If you do NOT have a GPU:    
`$ pip36 install tensorflow`  
If you DO have a GPU and have followed the steps in section 1. A.,:      
`$ pip36 install tensorflow-gpu==1.3.0`  

Install Keras  
`pip36 install keras`  

Now that all is installed, deactivate the virtual environment.  
`$ deact`  

Test the install  
`$ py36`  

\>>> import keras  

Is the tensorflow backend used?  

Ctrl+D to exit.   


#### F. Python + CUDA  

Installing Numba for Python + CUDA programming (IF you have a GPU and installed CUDA)  

Numba requires llvm  
`~$ sudo apt-get install llvm-5.0 llvm-5.0-dev`  

Install Numba  
`~$ pip36 install numba`  

Set path variables for Numba by adding these two lines to your .bashrc:  
```  
export NUMBAPRO_NVVM=/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda-8.0/nvvm/libdevice/
```  

The Python 3.6 header is needed to build using the makefile below  
`~$ sudo apt-get install python3.6-dev`  

##### Compile a Python extension with nvcc  
To test Python, C, and CUDA, try the Mandelbrot set visualization below:  
`~$ git clone https://github.com/tterava/Mandelbrot`  
`~$ cd Mandelbrot`  
Change the PYTHONPATH value in Makefile to that of version 3.6.  
`~$ make`  
The module requires pygame  
`~$ pip36 install pygame`  
`~$ py36 mandelbrot.py`  

You should now see a pretty fractal thingy.

##### Numba  
To test a simple Numba /@jit CUDA kernel, try this example from  
https://nyu-cds.github.io/python-numba/05-cuda/  

```  
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code   
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)
```  

You should see the following output:  
```  
[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.  2.
  2.  2.  2.  2.]

```  


## _Section 2_ - Computer Vision (CV)

#### A. Install OpenCV
Many of my Computer Vision needs are met by SciKit-Image, but I often use OpenCV 3

( this really seems to work...still verifying: ~$ pip install opencv-contrib-python )  

Install these dependencies first. See what is required and what is optional here:  
https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html  

`$ sudo apt-get install build-essential`  
`$ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev`
`sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev`   
`$ sudo apt-get install python3.6-dev`  

Get OpenCV from the Git Repository  
`$ cd ~`  
`$ wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip`  
`$ unzip opencv.zip`  
`$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip`  
`$ unzip opencv_contrib.zip`  

`$ cd opencv-3.3.0`  
`$ mkdir build`  
`$ cd build`  
`$ act_ml36`  

We are going to use cmake, but need to use GCC 5 that we installed above, instead of the default GCC 7. Thanks to this post for the fix and the reason it's prefered:  
https://stackoverflow.com/questions/17275348/how-to-specify-new-gcc-path-for-cmake  

Enter this at the command line:  
`~$ export CC=/usr/bin/gcc-5`  
`~$ export CXX=/usr/bin/g++-5`  

Using this version of g++ (5.4.1 20171010 (Ubuntu 5.5.0-1ubuntu1)) will throw an error if we do not make one small change to the config file. Thanks to this post for the fix:  
https://github.com/opencv/opencv/issues/10032  
`~$ cd /usr/include/x86_64-linux-gnu/c++/5/bits`  
Make a backup copy of the config file  
`~$ sudo cp c++config.h c++config.h.bak`  
Edit the file:  
`~$ sudo gedit c++config.h &`  
In the section beginning on line 1344    
```
/* Define if C99 functions or macros in <math.h> should be imported in <cmath>
   in namespace std. */
/* #undef _GLIBCXX_USE_C99_MATH */
```  
add the following line:  
`#define _GLIBCXX_USE_C99_MATH 1`  
Save this edit.  

Return to the OpenCV build directory  
`~$ cd ~/cd ~/opencv-3.3.0/build/`  

Since the following cmake command spans so many lines, copy and paste it into a text editor. Replace 'username' with your username. Then copy and paste that at the BASH prompt.  
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/home/username/.ml36/lib/python3.6/site-packages/opencv \
-D OPENCV_EXTRA_MODULES_PATH=/home/username/opencv_contrib-3.3.0/modules \
-D PYTHON_EXECUTABLE=/opt/python3.6/bin/python3.6 \
-D PYTHON_INCLUDE=/opt/python3.6/include/python3.6m \
-D PYTHON_LIBRARY=/opt/python3.6/lib/libpython3.6.so \
-D PYTHON_PACKAGES_PATH=/home/username/.ml36/lib/python3.6/site-packages \
-D PYTHON_NUMPY_INCLUDE_DIR=/home/username/.ml36/lib/python3.6/site-packages/numpy/core/include/numpy \
-D WITH_CUDA=OFF \
-D BUILD_EXAMPLES=ON ..
```  

Find out number of CPU cores in your machine  
`~$ nproc`  
You can run make with multiple cores with the -j flag. I'll use all but one of mine:   
`~$ make -j7`  
This will take a while...  study Japanese... maybe clean the attic...     

`$ make install`   
`$ sudo ldconfig`    

Make symlink to OpenCV in new Python site-packages directory  
`$ cd /home/username/.ml36/lib/python3.6/site-packages`  
`$ ln -s /home/username/.ml36/lib/python3.6/site-packages/opencv/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so`  

`$ deact`  


#### B. Invoking this OpenCV from C++ code  

Many examples using CUDA to parallelize image-processing code is written in C++ and uses OpenCV functions. These examples assume that OpenCV is installed to the default directory under /usr/local, however we have installed ours under our virtual Python directory .ml36.   
At this point you might be asking yourself, "why didn't he just install OpenCV to the default directory and then put a symlink in the appropriate virtual Python lib directory?". This might be the best way to do it...the truth is that in the past I have never used OpenCV except with Python and saw no reason for a system-wide install. Now that I am using OpenCV with C++ and NVCC, I will consider changing to a system-wide install of OpenCV. For now though, let's carry on.   

Below are two things to do to ensure that our C++ code will work with OpenCV.  

At compile time, the linker **ld** needs to know where to find the shared library paths. We can set these manually in /etc/ld.so.conf.d  
`~$ cd /etc/ld.so.conf.d`  
`~$ sudo touch opencv.conf`  
Add the following line to the file:  
/home/username/.ml36/lib/python3.6/site-packages/opencv/lib  
Save the file, then apply the change with  
`~$ sudo ldconfig`  

The next place we might need to specify paths to opencv is in our makefiles. Here is a Makefile from the first homework of Udacity's Intro To Parallel Programming, with explicit paths to our libraries and our header files:

```  
NVCC=nvcc

OPENCV_LIBPATH=/home/telemaque/.ml36/lib/python3.6/site-packages/opencv/lib
OPENCV_INCLUDEPATH=/home/telemaque/.ml36/lib/python3.6/site-packages/opencv/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include

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


#### C. dlib  

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


## _Section 3_ - Other  

#### A. Atom
In case you don't already know, Atom is a hackable text editor. When I say hackable, I mean that you can configure just about anything you want configured. It can be used to code almost any language you can think of. I use it to program in Python, C and C++. I use it for makefiles. I'm using it now to create this markdown document.

Download the .deb from https://atom.io/  
You can simply right-click to install it.  
Did you notice that earlier we put an alias in our .bashrc for Atom? We did this so that Atom would be aware our new version of Python instead of the one installed by default on our system.  
` $ atom36`  

Note: I'm not saying that Atom is better than your editor-of-choice, merely that I prefer it to others I have used. Learning Atom's keyboard shortcuts for the most common editing tasks will make you noticeably more productive.  

#### B. Jupyter Themer
We installed Jupyter above. We then installed Jupyter-themer. I prefer dark themes for when I code so I use the color 'midnignt'. Go to the Jupyter-themer page and see what you might like best.  

Change the Jupyter notebook colors if you wish (OPTIONAL). I like 'midnight'  
`$ jupyter-themer_36 -c 'midnight'`  

Then start Jupyter to check that the change took effect.  
`$ jupyter-notebook_36`  



## _CONCLUSION_  

We can now use three of our BASH aliases to use our new Python 3.6.3.

To run the Python interpreter in the terminal window, just type:  
`$ py36`

To run a Jupyter notebook powered by our new Python 3.6 installation, just type:  
`$ jupyter-notebook_36`  

To code in Atom with the ability to execute the code (with Ctrl+Shift+B) with our new version of Python instead of the default version on our system, just type:  
`$ atom36`  


If you find any error or have any questions, please open an issue and I will respond.
