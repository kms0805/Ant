"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment as linear_assignment
import time
import argparse
from filterpy.kalman import KalmanFilter
import cv2
import utils
#import seagull_detection as sdt



def convert_x_to_index(x):
  return np.array([x[0],x[2]]).reshape((1,2))

class KalmanFilterTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as index.
  """
  count = 0
  def __init__(self,index,time,id):
   
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=4, dim_z=2)
    self.kf.F = np.array( [[1 ,1, 0, 0], [0, 1, 0, 0], 
										[0, 0, 1, 1],  [0, 0, 0, 1]] )
    self.kf.H = np.array( [[1,0,0,0], [0,0,1,0]] ) 

    # self.kf.R[2:,2:] *= 10.
    self.kf.P[1,1] *= 1000.
    self.kf.P[3,3] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    # self.kf.Q[-1,-1] *= 0.01
    # self.kf.Q[4:,4:] *= 0.01

    self.kf.x[0] = index[0]
    self.kf.x[2] = index[1]
    self.time_since_update = 0
    self.id = id
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.time = time - 1

    self.trajactory_history = []
    self.trajactory_history.append(np.concatenate(([self.id],index,[self.time,1])).reshape(1,-1))

  def update(self,index):
    """
    Updates the state vector with observed index.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(index)
    self.trajactory_history[-1] = np.concatenate(([self.id],convert_x_to_index(self.kf.x)[0],[self.time,1])).reshape(1,-1)


  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.time += 1
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_index(self.kf.x))
    self.trajactory_history.append(np.concatenate(([self.id],self.history[-1][0],[self.time,0])).reshape(1,-1))

    return self.history[-1]

  def get_state(self):
    """
    Returns the current index estimate.
    """
    return convert_x_to_index(self.kf.x)
  
  def get_trajactory(self):
    return np.concatenate(self.trajactory_history)

def associate_detections_to_trackers(detections,trackers,threshold):
  """
  Assigns detections to tracked object 
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0) or (len(detections)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  cost_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
  cost_matrix_copy = cost_matrix.copy()
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      cost_matrix[d,t] = np.linalg.norm(det-trk)


  matched_indices = linear_assignment(cost_matrix)
  matched_indices = np.hstack((matched_indices[0].reshape(-1,1),matched_indices[1].reshape(-1,1)))
  # matched_indices = np.empty((0,2),dtype=int)
  # for c in range(cost_matrix.shape[0]):
  #     minidx = np.array((c,np.argmin(cost_matrix_copy[c], axis=0)))
  #     matched_indices = np.vstack((matched_indices,minidx))
  #     cost_matrix_copy[minidx[0],:] = 100000
  #     cost_matrix_copy[:,minidx[1]] = 100000
  #print(cost_matrix)
  matches = []

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  
  for m in matched_indices:
    if(cost_matrix[m[0],m[1]]>threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)




class Sort(object):
  def __init__(self,max_age=60,min_hits=5):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.deleted_trackers = []
    self.num_trk = 0

  def update(self,dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    #get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),2))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1]]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,50)

    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanFilterTracker(dets[i,:], self.frame_count, self.num_trk)
        self.num_trk += 1
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        #if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
        ret.append(np.concatenate((d,[trk.id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
          
          self.deleted_trackers.append(self.trackers.pop(i))
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,3))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args




if __name__ == '__main__':

  ## np.load error(?)
  np_load_old = np.load
  np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

  xmin1, xmax1, ymin1 ,ymax1 = 1200, 2400, 1000, 1600
  xmin2, xmax2, ymin2, ymax2 = 1600, 2800, 900, 1500

  images = []
  data = np.load('centroidInVideo2.npy')
  tracker = Sort()
  colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]
  tracklist = []
  for i in range(1000):
    tracklist.append([])
  for i in range(data.shape[0]):
      
    # full = utils.load_video_frame('Trial6/Cam1_C0010.mp4',i)
    full = utils.load_video_frame('Trial6/Cam3_C0007.mp4',i+856)
    # frame = full[ymin1:ymax1,xmin1:xmax1,:]
    frame = full[ymin2:ymax2,xmin2:xmax2,:]
    centers = np.array(data[i])
  
    trackers = tracker.update(centers)
    for d in trackers: 
      d = d.astype(np.int32)
      color = colours[d[2]%11]
      x = d[0]
      y = d[1]
      
      for t in tracklist[d[2]]:
        xl = t[0]
        yl = t[1]
        cv2.circle(frame,(xl,yl),1,color,1)
      tl = (x-10,y-10)
      br = (x+10,y+10)
      # cv2.rectangle(frame,tl,br,color,1)
      cv2.circle(frame,(x,y),6,color,1)
      cv2.putText(frame,str(d[2]), (x-10,y-20),0, 0.5, color,2)
      tracklist[d[2]].append(d)
      # cv2.imshow('image',frame)
      # time.sleep(0.1)
      # if cv2.waitKey(1) & 0xFF == ord('q'):
      #   cv2.destroyAllWindows()
      #   break
    images.append(frame)

  utils.making_video('tracking2.mp4',images,5)