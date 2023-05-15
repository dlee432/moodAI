import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from lime import lime_tabular
import math
import os

import numpy as np
import pandas as pd
import random
import itertools

from lstmClusters import ClustersFeatures

## This class contains all methods and functions regarding the LSTM models including the preprocessing and the basic and tuned LSTM models
## to determine the most important features.

class LSTMModels():
    # Initialisation
    def __init__(self, files):
        # Training Variables
        self.train = 0.75
        self.test = 0.25
        self.seed = 10

        self.neurons = 20
        self.epochs = 150
        self.window_size = 15
        self.batch_size = 32
        self.learning_rate = 0.01
        self.features=5

        self.files = files
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    # This function initialises the characteristics of the LSTM model
    def LSTMModel(self):
        model = Sequential()

        # add in the training variables in the LSTM
        model.add(LSTM(self.neurons, input_shape=(self.window_size, self.features)))

        # add a Dense layer with 7 output neurons 
        model.add(Dense(7, activation = 'softmax'))

        # use cross entropy for loss function and Adam optimizer from keras
        model.compile(loss="sparse_categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])

        return model

    # This function is the basic LSTM model that is used when running given parameters
    def basicLSTM(self, cluster_feature_names, raw_dfs, labels, neurons=None, epochs=None, window_size=None, batch_size=None, best_model=None):
        
        if neurons != None: self.neurons = neurons
        if epochs != None: self.epochs = epochs
        if window_size != None: self.window_size = window_size
        if batch_size != None: self.batch_size = batch_size

        # retrieve cluster features from dataset and preprocess
        normalised_cluster_dfs = self.preprocessings(raw_dfs, cluster_feature_names)

        # retrieve cluster name to print out during output
        self.features = len(cluster_feature_names)
        cluster_name = ClustersFeatures(cluster_feature_names).name
        if "ucsd" in cluster_name:
            cluster_name = cluster_name.split("_",1)[1]
        print (cluster_name)

        for index, f in enumerate(self.files):
            # create sliding window with a given window size, where x is the dataset for each feature and y is the target (depressed EMA)
            x, y = self.generate_3d_sliding_arrays(normalised_cluster_dfs[index], labels[index], self.window_size)
            
            # split the dataset into training and testing subset
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= self.test, random_state=self.seed)

            name = f.replace(".csv", "")

            # if there is no best model found yet
            if best_model == None:
                # print parameters for the current model and put them in appropriate folder
                print(f"neurons: {self.neurons}, epochs: {self.epochs}, window_size: {self.window_size}, batch_size: {self.batch_size}\n")
                filepath = f'modelCheckPoints/{name}/{cluster_name}/basic_model_bestmodel.hdf5'
                model = self.LSTMModel()
                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, monitor = 'val_accuracy', verbose = 0, save_best_only = True)

                # train the model using given parameters 
                model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = self.epochs, batch_size = self.batch_size, callbacks = [checkpoint], verbose = 0)

                # load all layer weights
                model.load_weights(filepath)
                loss, acc = model.evaluate(x_test, y_test)
                print("model, accuracy: {:5.2f}%".format(100 * acc))

            # if there is a best model identified
            else:
                model = tf.keras.models.load_model(best_model)

                mape_list = []
                mae_list = []

                # 5 fold cross validation
                kfold= KFold(n_splits=5, shuffle=True, random_state=self.seed)

                average_acc = 0
                average_mape = 0
                average_mae = 0

                # find total accuracy, MAPE and MAE
                for train_index, test_index in kfold.split(x, y):
                    temp_model = tf.keras.models.load_model(best_model)
                    temp_model.fit(x[train_index], y[train_index], validation_data = (x[test_index], y[test_index]), epochs = self.epochs, batch_size = self.batch_size, verbose = 0)

                    temp_loss, temp_acc = temp_model.evaluate(x[test_index], y[test_index])

                    average_acc = average_acc + temp_acc
                    
                    temp_y_pred_prob = temp_model.predict(x[test_index])
                    temp_one_hot_encoding = np.eye(7, dtype=int)[np.argmax(temp_y_pred_prob, axis=1)]
                    temp_y_pred = np.where(temp_one_hot_encoding == 1)[1]
                    print("k-fold Actual:     " + str(y[test_index]+1))
                    print("k-fold Preditions: " + str(temp_y_pred+1))

                    mape = np.mean(np.abs((temp_y_pred - y[test_index]) / (y[test_index] + 1))) * 100
                    mae = np.mean(np.abs(temp_y_pred - y[test_index]))
                    average_mape += mape
                    average_mae += mae
                    mape_list.append(mape)
                    mae_list.append(mae)
                
                # find average accuracy, MAPE and MAE of best model
                average_acc = average_acc/5
                average_mape = average_mape/5
                average_mae = average_mae/5

                std_mape = 0
                temp_mape = 0
                # find MAPE std
                for m in mape_list:
                    temp_mape += (abs(m - average_mape))**2
                
                std_mape = math.sqrt(temp_mape / 5)

                std_mae = 0
                temp_mae = 0
                # find MAE std
                for m in mae_list:
                    temp_mae += (abs(m - average_mae))**2
                
                std_mae = math.sqrt(temp_mae / 5)

                # the average accuracy of the model is
                print("5-fold CV average accuracy: " + str(average_acc))

                # the average difference between the predicted value and the actual value is
                print("5-fold CV average mape: " + str(average_mape))

                print("5-fold CV std mape: " + str(std_mape))

                print("5-fold CV mae: " + str(average_mae))

                print("5-fold CV std mae: " + str(std_mae))

                loss, acc = model.evaluate(x_test, y_test)
                print("Restored model with best configurations, accuracy: {:5.2f}%".format(100 * acc))

                # compare the prediction and the actual values
                y_pred_prob = model.predict(x_test)
                one_hot_encoding = np.eye(7, dtype=int)[np.argmax(y_pred_prob, axis=1)]
                y_pred = np.where(one_hot_encoding == 1)[1]
                print("Actual:     " + str(y_test+1))
                print("Predictions: " + str(y_pred+1))

            # get most important features 
            important_features_dict = dict.fromkeys(cluster_feature_names, 0)
            full_important_features_dict = dict()

            # make directory to store LIME HTMLs
            dir = f'lime/{name}/{cluster_name}'
            if not os.path.exists(dir):
                os.makedirs(dir)
            explainer = lime_tabular.RecurrentTabularExplainer(x_train, mode='classification', training_labels=y_train, feature_names=cluster_feature_names,
                                                    discretize_continuous=True,
                                                    class_names=['1', '2', '3', '4', '5', '6', '7'],
                                                    discretizer='decile',
                                                    random_state=self.seed)
                                                    
            # make directory to store explanations for each class and user
            exp_dir = f'explanation/{name}/{cluster_name}'
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            # generate explanation for predictions
            for i in range(0, len(x_test)):
                exp = explainer.explain_instance(x_test[i], model.predict, num_features=len(cluster_feature_names)*self.window_size, top_labels=7)
                exp.save_to_file(f'{dir}/run_{i}.html')
                for j in range(0, 7):
                    exp_list = exp.as_list(label=j)
                    print(exp_list)
                    print(y_test[i])
                    print(y_pred[i])
                    # exp.save_to_file(f'lime/synthesis_run_{i}.html')

                    # add the influence score from same features that predicts towards the correct value
                    for item in exp_list:
                        head, sep, tail = item[0].split('_t-', 1)[0].partition('< ')
                        if item[0][0].isdigit() or item[0][0] == '-':
                            full_feature_name = item[0].split(' ')[2]
                        else:
                            full_feature_name = item[0].split(' ', 1)[0]
                        weight = item[1]
                        if weight > 0.0:
                            if tail == '':
                                up_dict = {head:important_features_dict[head]+weight}
                                important_features_dict.update(up_dict)
                            else:
                                up_dict = {tail:important_features_dict[tail]+weight}
                                important_features_dict.update(up_dict)
                            value = full_important_features_dict.get(f'{full_feature_name}', 0)
                            full_important_features_dict[f'{full_feature_name}'] = value+weight
            print(full_important_features_dict)
            print(important_features_dict)

            most_important_features = []
            
            # take the features with influence of over 50% of the value of the total influence score
            fifty_percent_values = sum(important_features_dict.values()) / 2
            while fifty_percent_values > 0:
                max_value_feature = max(important_features_dict, key=important_features_dict.get)
                most_important_features.append(max_value_feature)
                fifty_percent_values -= important_features_dict[max_value_feature]
                del important_features_dict[max_value_feature]
            
            print(most_important_features)
            print("\n")

    # This function is for running different parameters for the basic LSTM model to find the best model
    def tunedLSTM(self, cluster_feature_names, raw_dfs, labels):

        learning_rates = [1e-2, 1e-3]
        neurons = [10,20,40,60,80]
        epochs = [50, 100, 150]
        window_sizes = [5,10,15]
        batch_sizes = [4,8,16,32]

        # retrieve cluster features from dataset and preprocess
        normalised_cluster_dfs = self.preprocessings(raw_dfs, cluster_feature_names)

        # retrieve cluster name to print out during output
        features = len(cluster_feature_names)
        cluster_name = ClustersFeatures(cluster_feature_names).name
        if "ucsd" in cluster_name:
            cluster_name = cluster_name.split("_",1)[1]
        print(cluster_name)

        for index, f in enumerate(self.files):
            name = f.replace(".csv", "")
            print(f'Tuning for participant: {name}')
        
            run = 0 
            best_accuracy = 0
            best_model = None

            cluster_best_weights = []
            
            
            for n, e, w, b, l in itertools.product(neurons, epochs, window_sizes, batch_sizes, learning_rates):
                print(f'Run:{run} Neurons:{n}, Epochs:{e}, Window Size:{w}, Batch Size:{b}, Learning Rate:{l}')

                # create sliding window with a given window size, where x is the dataset for each feature and y is the target (depressed EMA)
                x, y = self.generate_3d_sliding_arrays(normalised_cluster_dfs[index], labels[index], w)

                self.neurons=n
                self.window_size=w
                self.learning_rate=l
                self.features=features

                # split the dataset into training and testing subset
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = self.test, random_state=self.seed)
                filepath = f'modelCheckPoints/{name}/{cluster_name}/run_{run}_bestmodel.hdf5'
                model = self.LSTMModel()
                checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, monitor = 'val_accuracy', verbose = 0, save_best_only = True)

                # train the model using given parameters
                model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = e, batch_size = b, callbacks = [checkpoint], verbose = 0)
                # load all layer weights
                model.load_weights(filepath)
                score = model.evaluate(x_test, y_test)
                
                # find model with the best accuracy
                if score[1] > best_accuracy:
                    best_accuracy = score[1]
                    best_model = filepath
                    best_n = n
                    best_e = e
                    best_w = w
                    best_b = b
                    best_l = l
                run = run + 1
            run = 0

            print(f'Best accuracy for {name}: {best_accuracy} using weights from: {best_model}')
            cluster_best_weights.append({cluster_name: {"best_model":best_model, "best_n":best_n, "best_e":best_e, "best_w":best_w,"best_b":best_b, "best_l":best_l}})
            print(cluster_best_weights)
        
        return cluster_best_weights

    # Generate a sliding window array using a 2D np array and a given sliding window size
    def generate_3d_sliding_arrays(self, np_array, labels, window_size):
        a = []

        for i in range(window_size - 1, np_array.shape[0]):
            slide = np_array[i-(window_size - 1):i+1,:]
            # print(str(i-(window_size - 1)) + " " + str(i+1))
            a.append(slide)
        return np.vstack(np.array(a)[np.newaxis, :]), np.array(labels)[window_size - 1:] - 1

    # Filter features
    def filter_features(self, input_dfs, feature_list):
        result_dfs = []
        for input_df in input_dfs:
            result_df = pd.DataFrame()
            for feature in feature_list:
                result_df[feature] = input_df[feature]
            result_dfs.append(result_df)
        return result_dfs

    # Normalise list of pandas dataframes
    def normalise_dataframes(self, dfs):
        normalised_dfs = []
        for df in dfs:
            scaler = StandardScaler()
            normalised_df = scaler.fit_transform(df)
            normalised_dfs.append(normalised_df)

        return normalised_dfs
    
    # Run the preprocessing process which includes filtering of features and normalisation of data 
    def preprocessings(self, raw_dfs, cluster_feature_names):
        cluster_dfs = self.filter_features(raw_dfs, cluster_feature_names)
        # Normalise intake dataset
        normalised_cluster_dfs = self.normalise_dataframes(cluster_dfs)

        return normalised_cluster_dfs 
