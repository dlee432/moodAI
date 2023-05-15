# Project 90 - MoodAI Continued: Biophysically-based Mood State Analysis Using Explainable AI
By Marcus Li, Do Hyun Lee</br>
Part IV Research Project 2022</br>
University of Auckland

## Introduction
This repository is regarding the implementation of predicting and explaining depression classes of an individual through a compositional approach using LSTM models.

## Setting Up Virtual Environment and Dependencies

Set up a virtual environment before proceeding with executing the project code. From the root directory, execute:

### MacOS
```zsh
pip3 install virtualenv
virtualenv -p python3 venv
source ./venv/bin/activate
pip install -r requirements.txt
```
### Windows 
```zsh
pip3 install virtualenv
virtualenv -p python3 venv
source ./venv/Scripts/activate
pip install -r requirements.txt
```

Before running the project, activate the virtual environment. To activate it run:

### MacOS
````zsh
source ./venv/bin/activate
````

### Windows
````zsh
source ./venv/Scripts/activate
````

## Development
Before executing the project code, you can change the cluster used in `cpnn_lstm.py` (line 11). You can refer to the cluster enums from `lstmClusters.py`.

To execute the main program code:
````zsh
python3 cpnn_lstm.py
````

After executing the program, you would need to select the target participant used for the program.

## Feature Importance
To find the most influential feature conditions, create a `test.txt` file and populate the `test.txt` file with all the feature conditions of a depression class. </br>
Then execute the `feature_importance.py` file:
````zsh
python3 feature_importance.py
````

## Reference
````zsh
R. V. Shah, G. Grennan, M. Zafar-Khan, F. Alim, S. Dey, D. Ramanathan, and J. Mishra,
“Personalized machine learning of depressed mood using wearables,” Nature News, 09-Jun-2021.
[Online]. Available: https://www.nature.com/articles/s41398-021-01445-0.
````

## Acknowledgements
````zsh
Dr Partha S. Roop
Dr Frederick Sundram
Aron Jeremiah
Henry Liu
````
