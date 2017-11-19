<h2>Miles Porter
Painted Harmony Group, Inc.</h2>
<h1>TensorFlow Speech Recognition Challenge</h1>
<h3>
mporter@paintedharmony.com<br>
http://datascience.netlify.com<br>
November 19, 2017
</h3>

# Introduction

This repository contains items related to Miles Porter's entry into the (Kaggle TensorFlow Speech Recognition Challenge)[https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion?sortBy=hot&group=all&page=1&pageSize=20&category=all&kind=all].

The overall goal of this project is to use the provided audio files to build a model that will correctly predict words based on audio recordings.  The words for this challenge
are:

- yes
- no
- up
- down
- left
- right 
- on
- off
- stop
- go
- silence  (This is actual silence, not the word "silence")
- unknown  (This is any word that is not in the above set.)

# Approach

The approach for this project will consist of data preprocessing, model training, model evaluation, and Raspberry Pi implementation.  

## Preprocessing

Preprocessing will consist of taking the individual training files and performing a Mel Frequency Capstral Coefficient analysis of the individual files.  This work will closely
follow the work that was done by Miles Porter [in this blog post](http://datascience.netlify.com/general/2017/09/14/data_science_20.html).  The files will essentially be broken down
into a audio spectral map, and that map will be used in training.

##  Training

The training set will be partitioned into an 80/20 split for training and model evaluation.  Once the audio files have been converted into spectral maps, a convolutional neural network will be trained to identify the category of the data.  The approach for doing the training
will involve using the Keras python framework running on top of Google TensorFlow.

##  Evaluation

One the set has been trained on the 80% training set, it will be further evaluated on the remaining 20% of the training set with the known values.  Lastly, the model will be used to predict the categorization of the contest testing set.  This set contains over 150,000 data files.  The results of each of these files will be placed in a .csv file and submitted.  Not all of the testing set is used as part of the project evaluation.

##  Raspberry Pi

Lastly, the model will be converted to run on a Raspberry Pi platform, and a small demo app will be created to process audio files in real time based on the trained model...  If all goes well.  :)

#  Conclusion

All of the code and tests created by Miles Porter and included in this project should be considered opensource.

Information for data associated with this project can be found through the following citation:
 
"Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"., ex
