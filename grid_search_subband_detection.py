import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from tqdm import tqdm
from os import listdir
from os.path import join
import os

from scipy.stats import skew, kurtosis


def get_moments(cover_path, stego_path, N_IMG=2_400):

    los = sorted(listdir(stego_path))[:N_IMG]
    loc = sorted(listdir(cover_path))[:N_IMG]

    samples = np.zeros((len(loc), 2, 2, 4))
    for i in tqdm(range(len(loc))):
        tmp = np.load(join(cover_path, loc[i]))[:,:]
        samples[i,0,:,0] = tmp.mean(axis=0)
        samples[i,0,:,1] = tmp.std(axis=0)
        samples[i,0,:,2] = skew(tmp, axis=0)
        samples[i,0,:,3] = kurtosis(tmp, axis=0)

    for i in tqdm(range(len(los))):
        tmp = np.load(join(stego_path, los[i]))[:,:]
        samples[i,1,:,0] = tmp.mean(axis=0)
        samples[i,1,:,1] = tmp.std(axis=0)
        samples[i,1,:,2] = skew(tmp, axis=0)
        samples[i,1,:,3] = kurtosis(tmp, axis=0)

    samples_training = np.concatenate([samples[:,0], samples[:,1]])
    samples_training = samples_training.reshape(-1,samples_training.shape[1]*samples_training.shape[2])
    labels = np.concatenate([np.zeros((len(loc))), np.ones((len(los)))])

    return samples_training, labels


if __name__=="__main__":

    results_mean = np.zeros((12,12))
    results_std = np.zeros((12,12))
    for i in range(12):
        for j in range(i):
            if i != j:
                cover_path = f"dataset/cover/ica_all_pairs/{i}_{j}"
                stego_path = f"dataset/hinet/ica_all_pairs/{i}_{j}"
                samples_training, labels = get_moments(cover_path, stego_path)
                svm = SVC(kernel="rbf")

                cv_results = cross_validate(svm, samples_training,labels, cv=5)
                print(i,j, cv_results["test_score"], cv_results["test_score"].mean(), cv_results["test_score"].std())
                results_mean[i,j] = cv_results["test_score"].mean()
                results_std[i,j]  = cv_results["test_score"].std()
        print()

    np.save("all_subbands_mean.npy", results_mean)
    np.save("all_subbands_std.npy" , results_std)