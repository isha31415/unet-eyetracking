Isha Puri
2017
Work Done at Cox Lab, Harvard University
Accurate Tracking of Eye Movements Using Deep Learning with UNETS

________________ ________________ ________________ ________________ ________________
UnetTrainTest.py is the file that contains the training and testing code for both pupil and corneal models. 

Terminal Commands for execution: 
- Testing pupil model on a single file:
    - python UnetTrainTest.py pupil -f file_with_path_name
- Training pupil model
    - python UnetTrainTest.py train pupil
- Testing corneal model on a single file:
    - python UnetTrainTest.py corneal -f file_name
- Training corneal model
    - python UnetTrainTest.py train corneal
________________ ________________ ________________ ________________ ________________

The datasets folder includes all of the training/testing data (for both corneal and pupil models). Model was trained on rodents but works on most human eyes as well. ________________ ________________ ________________ ________________ ________________

original_pupil.h5 contains the weights for the pupil model. 
original_corneal.h5 contains the weights for the corneal model. 
________________ ________________ ________________ ________________ ________________