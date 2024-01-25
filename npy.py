import cv2
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
import pickle
def process_video(video_file_path,optical_frames,rgb_frames):
    cap = cv2.VideoCapture(video_file_path)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # Calculate the dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Visualize the dense optical flow
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        x=np.array(bgr)
        y=np.array(frame2)
        optical_frames.append(x)
        rgb_frames.append(y)


    cap.release()

if __name__ == '__main__':
    folder_path = "F:\\rahul was here\\non_violence_2"
    files = os.listdir(folder_path)
    mp4_files = [f for f in files if f.endswith(".mp4")]
    mp4_file_paths = [os.path.join(folder_path, f) for f in mp4_files]
    for i in tqdm(range(0,len(mp4_file_paths))):
        optical_frames=[];
        rgb_frames=[];
        process_video(mp4_file_paths[i],optical_frames,rgb_frames)
        optical_frames_np=np.array(optical_frames)
        rgb_frames_np=np.array(rgb_frames)
        with open(mp4_files[i][:-4]+'_optical.pkl', 'wb') as f:
            pickle.dump(optical_frames_np, f)
        with open(mp4_files[i][:-4]+'_rgb.pkl', 'wb') as f:
            pickle.dump(rgb_frames_np, f)
            
    print('fin')
