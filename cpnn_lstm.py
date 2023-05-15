from termcolor import cprint
from utilities.chooseFile import InputData
from lstmModels import LSTMModels
from lstmClusters import ClustersFeatures

## This class is the main class that runs the LSTM model on the provided cluster and returns the most accurate model with its most important features. 

if __name__ == "__main__":
    file_path = "jyotis/"
    # Choose a cluster from the lstmClusters class to run the LSTM model 
    cluster_feature_names = ClustersFeatures.ucsd1_synthesis_feature_names.value

    input = InputData(file_path)
    input.chooseFile(file_path)

    # if input from user is numeric
    if input.chosen_file_index.isnumeric():
        # run neural network on specific file in the directory
        if 1 <= int(input.chosen_file_index) <= len(input.index_list):

            filename = input.directory_content_dict[input.chosen_file_index]
            nn = LSTMModels([filename])
            input.preprocessing([filename])
            
            # Run the best model's neural network to find the most important features
            cluster_best_weights = nn.tunedLSTM(cluster_feature_names, input.raw_dfs, input.labels)
            cluster_name = ClustersFeatures(cluster_feature_names).name
            if "ucsd" in cluster_name:
                cluster_name = cluster_name.split("_",1)[1]
            cluster = cluster_best_weights[0][cluster_name]
            window_size=cluster["best_w"]
            best_model=cluster["best_model"]

            nn.basicLSTM(cluster_feature_names, input.raw_dfs, input.labels, window_size=5, best_model="modelCheckPoints/ucsd1/synthesis_feature_names/run_194_bestmodel.hdf5")
        else:
            cprint("ERROR: Please enter a number between 1 to " + str(len(input.index_list)), "red")
            print("")
    else:
        cprint("ERROR: Please enter a number between 1 to " + str(len(input.index_list)), "red")
        print("")
