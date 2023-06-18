## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
project developed to 
	* analyse driver awareness via two webcames
 	* read data in real time from accelerometer - uses threshold points to predict an accident.
  	* feed from cameras will be recorded and will present the last x seconds / minutes of driving before collision / crash
   Would love any suggestions on other features I could add to this project.
 

## Technologies
Project is created with:
* Python 3.7
	* mediapipe
	* opencv
	* numpy
	
## Setup
To run this project, install it locally :
connect two cameras to your pc - your built in camera (if you have one) will be your first camera
```
gh repo clone calsickofthis/driver_awareness
```

```
$ cd driver_awareness
$ pip install -r requirements.txt
$ python main.py
```


## References
* to read : https://www.youtube.com/watch?v=We1uB79Ci-w
