import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

## frame들의 평균을 통해서 배경 구하기
def meanOfFrames(video, start_frame, end_frame, xmin, xmax, ymin, ymax):
    print("getting backgorund")
    frame1 = utils.load_video_frame(video, start_frame)
    frame1 = frame1[ymin:ymax,xmin:xmax,:]
    sum = np.zeros(frame1.shape)
    n = end_frame - start_frame
    for i in range(n):
        if i%25 == 0:
            print(i,">>>>")
        frame = utils.load_video_frame(video, start_frame + i)
        frame = frame[ymin:ymax,xmin:xmax,:]
        sum = sum + frame
        mean = (sum/n).astype('uint8')
    print('finish')
    return mean
def medianOfFrames(video, start_frame, end_frame, step, xmin, xmax, ymin, ymax):
    print("getting backgorund")
    frames = []
    n = end_frame - start_frame
    i = 0
    while(i < n):
        frame = utils.load_video_frame(video, start_frame + i)
        frames.append(frame)
        i += step
    median = np.median(frames, axis=0).astype(dtype=np.uint8)
    print('finish')
    return median
    
#background subtraction
def getMask(img,background,threshold, is_black = None):
    diff = cv2.absdiff(img,background)
    diff = diff.mean(axis=2)
    mask = diff > threshold
    mask = mask[:,:,np.newaxis]
    if(is_black is not None):
        mask_b =   is_black & (diff > threshold - 20)
        mask_b = mask_b[:,:,np.newaxis] | mask
        mask = mask_b
    return mask

#색깔로 갈매기의 몸체 추출하기(흰색 부분만 추출)
def getBodyMaskByColor(img,threshold):
    if img.ndim == 3 :
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if img.ndim == 1:
        gray = img
    mask = gray > threshold
    mask = mask[:,:,np.newaxis]
    return mask

#mask에 해당하는 부분만 빨간색으로 칠해서 plot
def visualize_detection(img,mask,alpha = 0.5):
    if mask.ndim == 2:
        mask = mask[:,:,np.newaxis]
    if mask.dtype != 'bool':
        mask = mask.astype('bool')
    mask_inv = np.invert(mask)
    mask = mask.astype(np.uint8)
    mask_inv = mask_inv.astype(np.uint8)
    red = np.zeros(img.shape, dtype = 'uint8')
    red[:] = 255,0,0
    fg =  red * mask
    bg = img * mask_inv
    output = fg + bg
    plt.imshow(img)
    plt.imshow(output,alpha = alpha)



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
    video = 'AntClusterVideo/0417_short.mkv'
    start_frame = 0
    end_frame = 3000


    testimg = utils.load_video_frame(video,1)
    xmin, xmax, ymin ,ymax = 0, testimg.shape[0], 0, testimg.shape[1]
    median = medianOfFrames(video, start_frame, end_frame, 100, xmin, xmax, ymin, ymax)
    is_black = median.mean(axis=2) < 100
    
    whole_frames = []
    cap = cv2.VideoCapture(video)
    if cap.isOpened() is False:
        print('Failed to open video')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            whole_frames.append(frame)
        else:
            break
    cap.release()

    masks = []
    for frame in whole_frames:
        mask = getMask(frame, median, 50, is_black)
        masks.append(mask)
    np.save('pixel_ant.npy', masks)
    print("save")