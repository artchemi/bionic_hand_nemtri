"""
    Description: Utilities for extracting and preprocessing sEMG signals data.
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import scipy.io
import random
import os
import config
import seaborn as sns

from matplotlib.patches import ConnectionPatch

def folder_extract(root_dir, exercises=["E2"], myo_pref="elbow"):
    """
    Purpose:
        Extract sEMG signals data from files beneath folder 'root_dir'(from args)

    Args:
        1. root_dir (str):
            Root directory of the Ninapro DB5. (With folders and files storing sEMG data underneath)
        
        2. exercises (1D list, optional):
            Exercises with dedicated gestures stored. Defaults to "E2".
            - Note:
                "E3" may match to file: Ninapro_DB5\s2\S2_E3_A1.mat
                "E2" may match to file: Ninapro_DB5\s2\S2_E2_A1.mat
                "E1" may match to file: Ninapro_DB5\s2\S2_E1_A1.mat
            - Example:
                ["E3", "E2"] as args collects sample from both exercise 3 and 2.
        
        3. myo_pref (str, optional):
            Ninapro DB5 data was collected via 2 Myo armband
            - "elbow" collects sEMG from 1:8 channels, samples closest to elbow (From Myo Armband 1)
            - "wrist" collects sEMG from 9:16 channels, samples closest to wrist (From Myo Armband 2)

            Defaults to "elbow".

    Returns:
        1. (numpy.ndarray):
            - Samples collected from "emg" column within each .mat files wihtin the folder 'root_dir'(from args)
            - Shape: [num samples, 8(1 sEMG sample from each 8 Myo sensors/channels)]
            
        2. (numpy.ndarray): _description_
            - Targets/labels collected from "stimulus" column within each .mat files wihtin the folder 'root_dir'(from args)
            - Shape: [num samples]
    """
    
    emg = []
    emg_label = []
    
    # Parse through sub folders underneath 'root_dir'(from args)
    for folder in os.listdir(root_dir):
        subfolder_dir = root_dir + "/" + folder
        # Parse through .mat files underneath sub folders
        for file in os.listdir(subfolder_dir):
            # Get sEMG signals of dedicated Myo armband and Exercise
            if file.split("_")[1] in exercises:
                file_path = subfolder_dir + "/" + file
                # Read .mat file
                mat = scipy.io.loadmat(file_path)
                
                # Get first 8 Myo sensors/channels closest to elbow
                if myo_pref == "elbow":
                    emg += [sensors[:8] for sensors in mat["emg"]]
                # Get last 8 Myo sensors/channels closest to wrist
                elif myo_pref == "wrist":
                    emg += [sensors[8:] for sensors in mat["emg"]]
                # Get all 16 Myo sensors/channels
                else:
                    emg += mat["emg"]
                
                current_exercise = file.split("_")[1]
                
                if current_exercise == "E2":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 12)
                        else:
                            new_labels.append(0)
                    
                    emg_label.extend(new_labels)
                
                elif current_exercise == "E3":
                    labels = mat["stimulus"].reshape(-1)
                    new_labels = []
                    
                    for label in labels:
                        if label != 0:
                            new_labels.append(label + 29)
                        else:
                            new_labels.append(0)
                    
                    emg_label.extend(new_labels)
                
                else:
                    # Collect corresponding labels
                    emg_label.extend(mat["stimulus"].reshape(-1))
    
    return np.array(emg), np.array(emg_label)


def standarization(emg, save_path=None):
    """
    Purpose:
        Apply Standarization (type feature scaling) to sEMG samples 'emg'(from args)

    Args:
        1. emg (numpy.ndarray):
            The sEMG samples to apply Standarization (First output of function "folder_extract")
            
        2. save_path (str, optional):
            Path of json storing MEAN and Standard Deviation for each sensor Channel. Defaults to None.

    Returns:
        (numpy.ndarray):
            sEMG signals scaled with Standarization.
    """

    # Dictionary storing MEAN and Standard Deviation for each sensor Channel
    params = {i:[None, None] for i in range(8)}
    
    # Transform shape of 'emg'(from args)
    # [num samples, 8(sensors/channels)] -> [8(sensors/channels), num samples]
    new_emg = []
    for channel_idx in range(8):
        # Collect all samples of each sensor/channel
        new_emg.append([emg_arr[channel_idx] for _, emg_arr in enumerate(emg)])
    new_emg = np.array(new_emg)
    
    # Apply Standarization
    for channel_idx in range(8):
        # Calculate Mean from samples of each local sensor/channel
        params[channel_idx][0] = float(np.mean(new_emg[channel_idx]))
        # Calculate Standard Deviation from samples of each local sensor/channel
        params[channel_idx][1] = float(np.std(new_emg[channel_idx]))
        # Apply Standarization to samples of each local sensor/channel
        new_emg[channel_idx] = (new_emg[channel_idx] - params[channel_idx][0])/params[channel_idx][1]
    
    # Transform shape of new_emg
    # [8(sensors/channels), num samples] -> [num samples, 8(sensors/channels)]
    final_emg = []
    for idx in range(new_emg.shape[1]):
        # Convert back to sEMG arrays with 1 sample from each sensor/channel
        final_emg.append([sensor_samples[idx] for _, sensor_samples in enumerate(new_emg)])
    final_emg = np.array(final_emg)
    
    # Save MEANs and Standard Deviations if 'save_path'(from args) was provided
    if save_path != None:
        with open(save_path, 'w') as f:
            json.dump(params, f)
    
    return np.array(final_emg)


def gestures(emg, label, targets=[0, 1, 3, 6],
             relax_shrink=80000, rand_seed=config.SEED):
    """
    Purpose:
        Organize sEMG samples to dictionary with:
            - key: gesture/label
            - values: array of sEMG sigals corresponding to the specific gesture/label

    Args:
        1. emg (numpy.ndarray):
            The array of sEMG samples (First output of function "folder_extract" or "standarization")
        
        2. label (numpy.ndarray):
            Array of labels for the sEMG samples (Second output of function "folder_extract")
        
        3. targets (list, optional):
            Array of specified wanted gesture/label. Defaults to [0, 1, 3, 6].
        
        4. relax_shrink (int, optional): Shrink size for relaxation gesture. Defaults to 80000.
        
        5. rand_seed (int, optional): Random seed for shuffling before shrinking relaxation gesture samples. Defaults to 2022.

    Returns:
        gestures (dict):
            - Dictionary with:
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
                
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                    num gestures (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                }
    """
    
    if relax_shrink != None:
        assert 0 in targets
        assert rand_seed != None
    
    gestures = {label:[] for label in targets}
    # Sort each sEMG array to the corresponding gesture/label
    for idx, emg_array in enumerate(emg):
        if label[idx] in gestures:
            gestures[label[idx]].append(emg_array)
    
    # Too much relaxation gesture, just randomly shrink some
    if relax_shrink != None:
        random.seed(rand_seed)
        gestures[0] = random.sample(gestures[0], relax_shrink)
    
    return gestures


def plot_distribution(gestures):
    """
    Purpose:
        Plot distribution of number of gesture samples in pie chart.

    Args:
        1. gestures (dict):
            (Output of function "gestures")
            
            - Dictionary with: 
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
            
            
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                }
    """
    labels = []
    for _, (label, signals) in enumerate(gestures.items()):
        signals = np.array(signals)
        labels += [label for _ in range(len(signals))]
        
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(20, 6))
    plt.pie(counts, labels = unique, autopct='%1.0f%%')
    plt.show()
    
    
def train_test_split(gestures, split_size=0.75, rand_seed=config.SEED):
    """
    Purpose:
        Perform train test split

    Args:
        1. gestures (dict):
            (Output of function "gestures")
            
            - Dictionary with:
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                }
                
        2. split_size (float, optional):
            Split size, 0.25 refers to 25% test samples, 75% train samples. Defaults to 0.25.
            
        3. rand_seed (int, optional):
            Random seed for random shuffling reproducibility. Defaults to 2022.

    Returns:
        1. train_gestures (dict):
            - Dictionary with:
                - key: gesture/label for TRAINING set
                - values: array of sEMG sigals corresponding to the gesture/label
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                }
                
        2. test_gestures (dict): 
            - Dictionary with:
                - key: gesture/label for TESTING set
                - values: array of sEMG sigals corresponding to the gesture/label
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                }
    """
    train_gestures = {key:None for key in gestures}
    valid_gestures = {key:None for key in gestures}
    test_gestures = {key:None for key in gestures}
    
    # Shuffle sEMG data and split to training and testing set
    # |---------- 75% ----------|----- 12.5% -----|----- 12.5% -----|   - signals
    #            train          ^     valid       ^      test     
    #                    threshold_train   threshold_valid
    for _, (label, signals) in enumerate(gestures.items()):
        random.Random(rand_seed).shuffle(signals)
        
        threshold_train = int(len(signals) * split_size)
        threshold_valid = threshold_train + int(len(signals) * 0.125)

        train_gestures[label] = signals[0:threshold_train]
        valid_gestures[label] = signals[threshold_train:threshold_valid]
        test_gestures[label] = signals[threshold_valid:]

        # threshold_valid = int(len(test_gestures[label]) * 0.5)    # Размер валидационной выборка
        # valid_gestures[label] = signals[threshold_valid:threshold]
    
    return train_gestures, valid_gestures, test_gestures
    
    
def apply_window(gestures, window=32, step=16):
    """
    Purpose:
        Convert sEMG signal samples to sEMG image format.

    Args:
        1. gestures (dict):
            (Any output from function "gestures" or "train_test_split")
        
            - Dictionary with:
                - key: gesture/label
                - values: array of sEMG sigals corresponding to the gesture/label
            - Structure:
                {
                    0 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    1 (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                    ...
                    num gestures (gesture/label) : [...] (sEMG samples of dedicated gesture/label)
                }
                
        2. window (int, optional):
            How many samples each sEMG image channel contains. Defaults to 52.

    Returns:
        1. signals (numpy.ndarray):
            Processed sEMG signals in sEMG image format.
            - Example shape: [num samples, 1, 8(sensors/channels), 52(window)]
            
        2. outputs (numpy.ndarray):
            Labels for the sEMG signals
    """
    inputs = []
    outputs = []

    # Segment samples to list of windows
    for idx, (label, signals) in enumerate(gestures.items()):
        # signals.shape: [num samples, 8(sensors/channels)]
        signals = np.array(signals)
            
        windowed_signals = [signals[i:i+window] for i in range(0, len(signals)-window, step)]
        
        inputs.extend(windowed_signals)
        outputs.extend(
            [idx for _ in range(len(windowed_signals))]    
        )

    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    signals = []

    # Transform dimensions:
    #   [num samples, window, sensors/channels] -> [num samples, sensors/channels, window]
    for samples in inputs:
        # sample.shape: [window, sensors/channels]
        
        temp_window = []
        for channel_idx in range(len(samples[0])):
            # Collect channel/sensor sample from each emg_array
            temp_window.append([emg_array[channel_idx] for _, emg_array in enumerate(samples)])
            
        signals.append(temp_window)

    signals = np.array(signals)
    
    return signals, outputs


def realtime_preprocessing(emg, params_path=None, num_classes=4, window=32, step=16):
    """
    Purpose:
        Preprocess data samples obtained from realtime.py

    Args:
        1. emg (list):
            The sEMG samples obtained from realtime.py
        
        2. params_path (list, optional):
            - Path of json storing MEAN and Standard Deviation for each sensor Channel. Defaults to None.
        
        3. num_classes (int, optional):
            - Number of gestures/classes the new finetune model would like to classify. Defaults to 4.

    Returns:
        1. inputs (numpy.ndarray):
            Processed sEMG signals in sEMG image format.
            - Example shape: [num samples, 1, 8(sensors/channels), 52(window)]
        2. outputs (numpy.ndarray):
            Labels for the sEMG signals
    """
    emg = np.array(emg)
    
    # Apply Standarization feature scaling to samples if 'params_path'(from args) was provided
    if params_path != None:
        scaled_signals = []
        
        with open(params_path, 'r') as f:
            params = json.load(f)
            
        for channel_idx in range(8):
            mean = params[str(channel_idx)][0]
            std = params[str(channel_idx)][1]
            
            current_sample = emg[channel_idx]
            
            scaled_signals.append(
                (current_sample - mean) / std
            )
        scaled_signals = np.array(scaled_signals)
    else:
        scaled_signals = np.array(emg)
    
    # Convert sEMG sampels to sEMG windows appropriate for training
    sEMG = []
    for i in range(len(scaled_signals[0])):
        sEMG.append([scaled_signals[channel_idx][i] for channel_idx in range(8)])
    
    gesture = {i:[] for i in range(num_classes)}
    curr_gest = 0
    gest_size = int(len(sEMG)/num_classes)
    
    for i in range(0, len(sEMG), gest_size):
        gesture[curr_gest] = sEMG[i:i+gest_size]
        curr_gest += 1
    
    inputs, outputs = apply_window(gesture, window, step)
    
    return inputs, outputs



def main():
    emg, labels = folder_extract(config.folder_path, exercises=config.exercises, myo_pref=config.myo_pref)
    gest = gestures(emg, labels, targets=config.targets)

    train_gestures, valid_gestures, test_gestures = train_test_split(gest, rand_seed=config.SEED)
    print(train_gestures.keys())
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    X_valid, y_valid = apply_window(valid_gestures, window=config.window, step=config.step)
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)

    print(f"Train distr: {np.unique(y_train, return_counts=True)}")
    print(f"Valid distr: {np.unique(y_valid, return_counts=True)}")
    print(f"Test distr: {np.unique(y_test, return_counts=True)}")
    # print(gest.shape)
    # print(np.unique(gest, return_counts=True))

    # data from https://allisonhorst.github.io/palmerpenguins/
    
    sns.set_theme(style='whitegrid')    # Настройка визуала
    sns.set_palette("pastel")

    species = [str(name) for name in np.unique(y_train, return_counts=True)[0]]
    
    penguin_means = {
        'Train': np.unique(y_train, return_counts=True)[1],
        'Valid': np.unique(y_valid, return_counts=True)[1],
        'Test': np.unique(y_test, return_counts=True)[1],
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(5, 10))

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Sets by gesture types')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper right')
    # ax.set_ylim(0, 250)
    ax.grid(visible=False)

    plt.show()

if __name__ == '__main__':
    main()
