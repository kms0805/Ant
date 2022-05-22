import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def getCentroid(body, _eps = 1 , _min_samples =4):
    centroid = np.empty((0,2))
    nonzero_index = np.transpose(np.nonzero(body))
    features = nonzero_index
    clustering = DBSCAN(eps = _eps, min_samples= _min_samples).fit(features)
    core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
    core_samples_mask[clustering.core_sample_indices_] = True
    labels = clustering.labels_
    unique_labels = set(labels)
    
    for i,k in enumerate(unique_labels):
        if k == -1: #노이즈
            break
        class_member_mask = (labels == k)
        xy = nonzero_index[class_member_mask & core_samples_mask]
        x = xy[:,1].mean()
        y = xy[:,0].mean()
        centroid = np.append(centroid,np.array([[x,y]]),axis=0)
        
    return centroid


if __name__ == '__main__':
    masks = np.load('pixel_ant.npy')
    centroids = []
    i = 0
    for mask in masks:
        i = i + 1
        if i%100==0:
            print(i)
        centroid = getCentroid(mask)
        centroids.append(centroid)
        np.save('centroids_ant.npy', centroids)