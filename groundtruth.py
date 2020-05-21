import numpy as np 

class Groundtruth:

    def __init__(self,filename):
        self.filename= filename
        self.scale = 1
        with open(self.filename) as f:
            self.data = f.readlines()

    def getPoseAndAbsoluteScale(self, frame_id):
        ss = self.getDataLine(frame_id-1)
        x_prev = self.scale*float(ss[0])
        y_prev = self.scale*float(ss[1])
        z_prev = self.scale*float(ss[2])     
        ss = self.getDataLine(frame_id) 
        x = self.scale*float(ss[0])
        y = self.scale*float(ss[1])
        z = self.scale*float(ss[2])
        abs_scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        return x,y,z,abs_scale 

    def getDataLine(self, frame_id):
        return self.data[frame_id].strip().split()
