import numpy as np
import cv2
import csv
import json


class App:

    def __init__(self, tracks='out.json'):
        self.frame_idx = 0
        self.savedTracks = []
        with open(tracks, 'r') as f:
            self.savedTracks = json.load(f)
        self.prev_gray = None
        self.writer = None
        self.max_tracks = 5

    def close(self):
        if self.writer is not None:
            self.writer.release()

    def run(self):
        for track in self.savedTracks:
            frame = np.zeros((410, 800, 3))
            for point in track:
                row, col = int(float(point[1])), int(float(point[0]))
                frame[row, col, :] = 255
                if self.frame_idx == 0:
                    img_shape = frame.shape
                    # self.writer = cv2.VideoWriter("output.avi",
                    #                               cv2.VideoWriter_fourcc(*'XVID'), 30, (img_shape[1], img_shape[0]))
                self.frame_idx += 1
                cv2.imshow('lk_track', frame)
                # self.writer.write(frame)
                ch = cv2.waitKey(1)
                if ch == 27:
                    self.close()
                    break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        # video_src = './33_Trim_Trim.mp4'
        video_src = 'out.json'

    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
