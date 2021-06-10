import numpy as np
import cv2
import json
import sys
import lstm_model
import matplotlib.pyplot as plt

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

def showAccident(images):
    
    for count,img in enumerate(images):
        name = 'accident' + str(count) + '.png'
        print(name)
        cv2.imwrite(name,img)


class App:

    def __init__(self, video_src='91_Trim2.mp4'):
        self.track_len = 250
        self.detect_interval = 5
        self.tracks = []
        self.savedTracks = []
        self.savedPredictions = [] #there we save the last sets of predicted points
        self.prev_gray = None
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.writer = None
        self.track_count = 0
        self.model = lstm_model.LstmModel(20, 10)
        self.save_tracks = False
        self.accidentPos = []

    def close(self,vis=0):
        if self.save_tracks:
            self.saveTracks()
        self.cam.release()
        if self.writer is not None:
            self.writer.release()
        if self.accidentPos :
            print("Saving the images")
            showAccident(self.accidentPos)
            
            

    def saveTracks(self):
        saved = self.savedTracks
        with open("out.json", 'w') as f:
            json.dump(saved, f, default=lambda x: str(x))

    def plot_predictions(self, vis):
        predicted_tracks = []
        for tr in self.tracks:
            if len(tr) < 20:
                continue
            x = tr[-20:]
            pred = self.model.predict(x)
            predicted_tracks.append(pred)
        cv2.polylines(vis, [np.int32(tr) for tr in predicted_tracks], False, (0, 0, 255))
        
    def plot_accidentPos(self,pos,vis):
        #This gives a screen shot of all the positions where an accident occured
        color = (255, 0, 0)
        
        h, w, c = vis.shape
        
        start_point = (int(round(pos[0]-h*0.1)) , int(round(pos[1]-w*0.05)))
        end_point =   (int(round(pos[0] + h*0.1)), int(round(pos[1] + w*0.05)))
        thickness = 2
        
        print(vis)
        img = cv2.rectangle(vis , start_point , end_point , color,thickness)
        self.accidentPos.append(img)
        
        
        
    def CheckAccident(self,frame_gray,vis):
        #Here we are gonna check the distance between the prediction made for the last 5 points and the actual trajectory. 
        # we are going to measure the distance between the two vectors. Using the euclidian norm for the moment
        
        threshold_speed = 16
        threshold_trajectory = 20
        
        
        for idx, tr in enumerate(self.tracks):
            if(len(self.savedPredictions[idx]) == 10 ) : #checking if there is enough information to compare
                predictedTraj = np.array(self.savedPredictions[idx][0])
                
                Len = len(predictedTraj)
                realTraj = np.array(tr[-Len:]) 
                
                dist = 0
                
                traj = 0
                speed = 0
                
                #The metric used will be of our own creation
                for index in range(1,len(predictedTraj)):
                       a = np.linalg.norm(realTraj[-index]-realTraj[-index-1])
                       b = np.linalg.norm(predictedTraj[-index]- realTraj[-index-1])
        
                       c = np.linalg.norm(predictedTraj[-index]- predictedTraj[-index-1])
                       
                       dot_product = np.dot(realTraj[-index]-realTraj[-index-1], predictedTraj[-index]- realTraj[-index-1])
                       angle = np.arccos(dot_product/(a*b))
                       
                       traj += abs(np.sin(angle))
                       
                       
                       dist += max(0,threshold_trajectory-abs(np.sin(angle))) * max(0,np.abs(a - b)-threshold_speed)/max(1,a)
                        
                #☻if traj > threshold_trajectory :
                #  print('trajectory : ',abs(np.sin(angle)))
                
                speed = np.linalg.norm(predictedTraj[-1]- predictedTraj[0]) - np.linalg.norm(realTraj[-1]-realTraj[0]) 
                    
    
                if speed > threshold_speed :
                    self.plot_accidentPos(tr[-1],vis)
                       
                      
                #if dist == 0 :
                    #print("accident")
                    #self.plot_accidentPos(tr[-1],vis)
                    
                    
                    
    def run(self):
        while True:
            _ret, frame = self.cam.read()
            if not _ret:
                self.close()
                break
            if self.frame_idx == 0:
                img_shape = frame.shape
                print(img_shape)
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
                new_tracks = []
                new_predictions = []
                for idx, (tr, pr, (x, y), good_flag) in enumerate(zip(self.tracks, self.savedPredictions, p1.reshape(-1, 2), good)):
                    if not good_flag or speed_of_track(tr) < 0.05:
                        if self.save_tracks and len(tr) > 30:
                            self.savedTracks.append([((int(x), int(y)), None) for (x, y) in tr])
                        continue
                    
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    
                    #Predicting
                    if len(tr) > 20 :
                        pred = self.model.predict(tr[-20:])
                        pr.append(pred)
                            
                    if(len(pr)>10): # We need to change that to adapt it to the real size of the predictions !
                        del pr[0]
                    new_predictions.append(pr)

                self.tracks = new_tracks
                self.savedPredictions = new_predictions #now the prediction are in the same order as are the tracks, therefore we can find the corresponding ones.
                cv2.polylines(vis, [np.int32(tr[-10:]) for tr in self.tracks], False, (0, 255, 0))
                self.CheckAccident(frame_gray,vis)
            self.plot_predictions(vis)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
                        self.savedPredictions.append([]) #we add an empty list waiting to have enough points to make a prediction

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            self.writer.write(vis)

            ch = cv2.waitKey(1)

            if ch == 27:
                self.close(vis)
                break


def main():
    try:
        video_src = sys.argv[1]
    except:
        print("No args, Choosing default video")
        video_src = '83_Trim.mp4'

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
