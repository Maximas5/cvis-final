# CS 5680 Computer Vision Final Project

Repo found [here](https://github.com/Maximas5/cvis-final)

Paper found [here](https://ieeexplore.ieee.org/document/7014233)

### 11/25 Phase I Update:

- Laid out outline

- Created classes for Experts and Final Classifier

- Completed pre-classification algorithm to separate objects for individual classification

- Having some issues getting the dataset from https://mivia.unisa.it/datasets-request/

### 12/2 Phase II Update:

- Further research done on Movement Expert

- Further investigation of data (No meaningful progress, but promising directions)

- I Better Understand the missing areas of the paper I need to extrapolate/research, specifically the bag-of-words and corner location algorithms (SIFT)

### 12/11 Phase III (Final Due Date) Update:

#### Steps to run

Clone my git repository found here: https://github.com/Maximas5/cvis-final 

git clone https://github.com/Maximas5/cvis-final.git

Install the following packages:
	
    pip install opencv-python
	
    pip install numpy
	
    pip install pandas
	
    pip install python-abc

In main.py, adjust the constant variable FRAMES to adjust how many frames you wish to ingest. Note: the more frames included, the longer it will take to process.

Execute main.py

	python3 main.py

After the program concludes, the results will be saved in data/fire_class.csv

#### Execution time

Execution per frame takes from 5 to 15 seconds. On average, the program will take 166 minutes per 1000 frames

#### Code Summary

##### main.py

My implementation of the algorithm proposed in the paper using the components specified below.

##### Fire_Id.py

Class responsible for facilitating the entire process (pre-processing, data handling, etc)

##### Expert.py

Base class for experts

##### Color_Expert.py

The class responsible for classifying a blob using color

##### Shape_Expert.py

The class responsible for classifying a blob using the differences in shape between the current and previous frame

##### Movement_Expert.py

The class responsible for classifying a blob using the motion of the object from the previous frame

##### MES.py

The class responsible for synthesizing the classifications of the other experts into a final classification

##### test.ipynb

Just some experiments used to inform my technical decisions in the rest of the program

#### Packages

The following packages can be installed as stated above.

	opencv-python: For common computer vision tasks
	
    numpy: For linear algebra operations
	
    pandas: For data storage
	
    python-abc: Abstract class implementation for python

#### Improvement

There were no improvements made to the proposed algorithm 😓

#### Changes

Due to time restrictions (and my temporal inadequacy) I had to simplify MES.py and remove Movement_Expert.com. I was also unable to implement the confusion matrix weight training for all experts.

#### Experimental results (paper)

The researchers who wrote the paper my project is based on were able to get their model to classify instances of fire with a 93.55% accuracy rating with a 0% false negative rate.

#### Experimental results (personal)

I was unable to fully evaluate my model as I was unable to access labeled data and did not have the requisite time to label 3 minutes of the 30fps video I was able to obtain, however, simply looking over the results did not seem promising. There are likely many issues with my implementation that cause a very high false negative rate, which is very unlike the actual results of the paper.
