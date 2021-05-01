import numpy as np
import cv2
import csv
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=200,
                      qualityLevel=0.3,
                      minDistance=100,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def speed_of_track(track, l=10):
    if len(track) < 2:
        return 10
    track = track[-l:]
    dists = [np.linalg.norm(np.asarray(d0) - np.asarray(d1)) for d1, d0 in zip(track, track[1:])]
    avg_speed = sum(dists) / len(dists)
    return avg_speed

class App:
    
    def __init__(self, video_src='33_Trim+Trim.mp4'):
        self.track_len = 50
        self.detect_interval = 5
        self.tracks = []
        self.savedTracks = []
        self.prev_gray = None
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.writer = None

    def close(self): 
        self.saveTracks()
        self.cam.release()
        if self.writer is not None:
            self.writer.release()
        
            
    def saveTracks(self):
        saved = self.savedTracks
        with open("out.csv","w",newline="") as f:
            cw=csv.writer(f,delimiter = ';',escapechar=' ',quoting=csv.QUOTE_NONE)
            #cw.writerow(data)    # wrong
            for x in saved:
                
                    cw.writerow(x)
              
        #np.savetxt("Tracks.csv", saved, delimiter=",",fmt='%s')
        
    def avgSpeed(self):
        #This function will give the average speed of all the vehicles for a particular moment (based on the track of each vehicules)
        speeds = []
        for tr in self.tracks :
            #The time reference is the number of frames so the speed is = (distance/nbr of frame during distance)
            
            speeds.append(speed_of_track(tr))
        
        avgSpeed =  sum(speeds) / len(speeds)
        return avgSpeed
    
    
    def linearReg(self,vis):
        #predicting the following trajectory using linear regression
                predictedTracks = []
                for tr in self.tracks :
                    if len(tr) > 5 :
                        X = np.array(tr)[:,0]. reshape(-1,1)
                        Y = np.array(tr)[:,1].reshape(-1,1)
                        #absiss of value we want to predict
                        dist = X[-2] - X[-1]
                        
                        X_pred = [X[-1] - i*dist for i in range(int(np.ceil(len(X)/3)))]
                    
                        
                        to_predict_x= np.array(X_pred).reshape(-1,1)
                        
                        #now we predict 
                        regsr=linear_model.Ridge()
                        regsr.fit(X,Y)
                        predicted_y= regsr.predict(to_predict_x)
                        predictedTracks.append([(x, y) for x,y in zip(X_pred,predicted_y)])
                        
                    
                    
                
                
                cv2.polylines(vis, [np.int32(tr) for tr in predictedTracks], False, (0, 0, 255))  
                
    def run(self):
        while True:
            _ret, frame = self.cam.read()
            if not _ret:
                self.close()
                break
            if self.frame_idx == 0:
                img_shape = frame.shape
                self.writer = cv2.VideoWriter("output.avi",
                                              cv2.VideoWriter_fourcc(*'XVID'), 30, (img_shape[1], img_shape[0]))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                # fast = d > 1e-5
                # good = good & fast
                new_tracks = []
                new_tracks2 = []
                globalSpeed = self.avgSpeed()
                for tr, tr2, (x, y), good_flag in zip(self.tracks,self.savedTracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    if speed_of_track(tr) < 0.05:
                        continue
                    tr2.append((x , y , globalSpeed))
                    tr.append((x, y))
                
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    new_tracks2.append(tr2)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                
                self.tracks = new_tracks
                self.savedTracks = new_tracks2
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
            self.linearReg(vis)
      
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        self.savedTracks.append([(x,y,10)])
                        
            

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            self.writer.write(vis)

            ch = cv2.waitKey(1)
            
            if ch == 27:
                
                self.close()
                
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = './33_Trim_Trim.mp4'

    App(video_src).run()
    print('Done')



if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
