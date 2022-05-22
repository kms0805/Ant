import cv2
import numpy as np

def load_video_frame(path, frame_num):
    """
    Read and return n'th frame from video.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened() is False:
        print('Failed to open video')
    # print(path + "  has {} frames".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_num > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        print('Exceeded total number of frames')
        raise ValueError
    else:
        cap.set(1, frame_num)

    ret, frame = cap.read()
    if ret:
        return frame
    else:
        raise RuntimeError('fail to read frame{}'.format(frame_num))


def load_whole_video(path, crop=True, x0=1500, y0=1080, w=2500, h=540, sample_rate=30):
    # Read and return whole video sequence as a numpy array
    cap = cv2.VideoCapture(path)
    if not crop:
        w = int(cap.get(3))
        h = int(cap.get(4))
        x0 = 0
        y0 = 0
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))
    video = np.zeros((int(framenum/sample_rate), h, w, 3))
    cnt = 1
    while(cap.isOpened()):
        cap.set(1, cnt)
        _, frame = cap.read()
        video[cnt] = frame[y0:y0+h, x0:x0+w, :]
        cnt += sample_rate
    
    return video



def play_video(path, downsample=True, res=(960, 540)):
    """
    Play video with opencv.
    Quit video with pressing 'q' button.
    Args:
        path: video path
        res: resolution of video, tuple of (W, H)
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened() is False:
        print('Failed to open video')

    while cap.isOpened():
        ret, frame = cap.read()
        if downsample:
            frame = cv2.resize(frame, dsize=res)
        if ret:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def making_video(path_out,frames,fps):
    """
    frames -> list
    """
    temp = frames[0] 
    size = (temp.shape[1],temp.shape[0])
    out = cv2.VideoWriter(path_out,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    print("finish making video")
