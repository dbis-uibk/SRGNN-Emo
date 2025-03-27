import os
import numpy as np
import pickle


def load_preprocessed_data(data_dir, feature_file="features-jukebox.npy", label_file="label_dict.pkl"):
    """
    Loads data from data directory.
    """
    features = np.load(os.path.join(data_dir, feature_file))
    label_dict = pickle.load(open(os.path.join(data_dir, label_file), 'rb'))
    return features, label_dict


def split_label_validation(label_dict, ratio=0.2):
    labelled_item_ids = np.array(list(label_dict.keys()))
    n_samples = len(labelled_item_ids)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - ratio)))
    #train_set_items = [labelled_item_ids[s] for s in sidx[:n_train]]
    valid_set_items = [labelled_item_ids[s] for s in sidx[n_train:]]

    valid_label_dict = {}
    for k in valid_set_items:
        valid_label_dict[k] = label_dict.pop(k, None)

    return label_dict, valid_label_dict
