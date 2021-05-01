import numpy as np
import cv2
import csv
import json


class App:

    def __init__(self, tracks='predicts.json'):
        self.frame_idx = 0
        self.savedTracks = []
        with open(tracks, 'r') as f:
            # expecting predict format as list of tracks where a track is (coordinates as string or ints):
            # a list of points such as [ [row, col], None ] for no prediction or
            # a list of points such as [ [row, col], [[px0, py0],  [px1, py1], ...]] ] for  prediction
            self.savedTracks = json.load(f)
        self.prev_gray = None
        self.writer = None
        self.max_tracks = 5

    def run(self):
        self.writer = cv2.VideoWriter("output.avi",
                                      cv2.VideoWriter_fourcc(*'XVID'), 30, (410, 800))
        for track in self.savedTracks:
            frame = np.zeros((410, 800, 3), dtype=np.uint8)
            if self.frame_idx == 0:
                img_shape = frame.shape
                print(img_shape)
                self.writer = cv2.VideoWriter("output.avi",
                                              cv2.VideoWriter_fourcc(*'XVID'), 30, (img_shape[1], img_shape[0]))
            for point in track:
                pt = point[0]
                pd = point[1]
                row, col = int(pt[0]), int(pt[1])
                cv2.circle(frame, (row, col), 2, (0, 255, 0))
                pred_frame = np.zeros((410, 800, 3), dtype=np.uint8)
                if pd is not None:
                    for p in pd:
                        row, col = int(p[0]), int(p[1])
                        cv2.circle(pred_frame, (row, col), 2, (0, 0, 255))

                self.frame_idx += 1
                cv2.imshow('lk_track', frame+pred_frame)
                self.writer.write(frame+pred_frame)
                ch = cv2.waitKey(1)
                if ch == 27:
                    self.writer.release()
                    return
        self.writer.release()
        return


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        # video_src = './33_Trim_Trim.mp4'
        video_src = 'predicts.json'

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
