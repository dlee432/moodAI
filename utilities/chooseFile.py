from termcolor import colored
import yaml
import pandas as pd
import os

## This class chooses the .csv file of a user to run the neural network on 
class InputData():
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_dfs = []
        self.labels = []
        self.chosen_file_index = ''
        self.index_list = []
        self.directory_content = []
        self.directory_content_dict = dict()

    def preprocessing(self, directory_content):
        # Read data from CSV into dataframes
        for i, filename in enumerate(directory_content):
            raw_df = pd.read_csv(self.file_path + filename, header=0)
            self.raw_dfs.append(raw_df)         
        # print(f'Number of Participants: {len(raw_dfs)}')
        # print(f'Shape of dataframe: {raw_dfs[0].shape}')

        # Extract labels
        for raw_df in self.raw_dfs:
            label = raw_df["depressed"]
            self.labels.append(label)
        # print(f'Number of Labels: {len(labels)}')
        # print(f'First label file: {labels[0]}')
        # print(f'Shape of label dataframe: {labels[0].shape}')

    def chooseFile(self, directory):
        # get all contents from directory
        self.directory_content = sorted(os.listdir(directory), key=str.lower)
        self.index_list = [str(i+1) for i, _ in enumerate(self.directory_content)]
        self.directory_content_dict = dict(zip(self.index_list, self.directory_content))

        # ask user to choose what file or if all files should be used for the neural network
        print("")
        self.chosen_file_index = input(colored("Choose the following file to run the neural network, enter a number from 1 to " +
                        str(len(self.index_list)) + ", type " + str(len(self.index_list) + 1) + " to run all files: \n\n" + 
                        yaml.dump(self.directory_content_dict, sort_keys=False, default_flow_style=False) + "\nEnter input number here: ", "green"))
        print("")