
import time 
import numpy as np

import matplotlib.pyplot as plt
import multiprocessing as mp 
from multiprocessing import Process, Queue, Lock, RLock, Value



plt.ion()
    

# use mplotlib figure to draw in 3D trajectories 
class Mplot3d:
    def __init__(self, title=''):
        self.title = title 

        self.data = None  
        self.got_data = False 

        self.axis_computed = False 
        self.xlim = [float("inf"),float("-inf")]
        self.ylim = [float("inf"),float("-inf")]
        self.zlim = [float("inf"),float("-inf")]        

        self.handle_map = {}     
        
        self.key = Value('i',0)
        self.is_running = Value('i',1)         

        self.queue = Queue()
        self.vp = Process(target=self.drawer_thread, args=(self.queue,RLock(), self.key, self.is_running,))
        self.vp.daemon = True
        self.vp.start()

    def quit(self):
        self.is_running.value = 0
        self.vp.join(timeout=5)     
        
    def drawer_thread(self, queue, lock, key, is_running):  
        self.init(lock) 
        while is_running.value == 1:
            self.drawer_refresh(queue, lock)                 
            time.sleep(0.04)    
        print(mp.current_process().name,'closing fig ', self.fig)     
        plt.close(self.fig)                                 

    def drawer_refresh(self, queue, lock):            
        while not queue.empty():    
            self.got_data = True  
            self.data = queue.get()  
            traj, name, color, marker = self.data         
            np_traj = np.asarray(traj)        
            if name in self.handle_map:
                handle = self.handle_map[name]
                self.ax.collections.remove(handle)
            self.updateMinMax(np_traj)
            handle = self.ax.scatter3D(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2], c=color, marker=marker)
            handle.set_label(name)
            self.handle_map[name] = handle     
        if self.got_data is True:               
            self.plot_refresh(lock)          

    def on_key_press(self, event):
        #print(mp.current_process().name,"key event pressed...", self._key)     
        self.key.value = ord(event.key) # conver to int 
        
    def on_key_release(self, event):
        #print(mp.current_process().name,"key event released...", self._key)             
        self.key.value = 0  # reset to no key symbol
        
    def get_key(self):
        return chr(self.key.value) 
    
    def init(self, lock):
        lock.acquire()      
        self.fig = plt.figure()
        self.fig.canvas.draw_idle()         
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)       
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)             
        self.ax = self.fig.gca(projection='3d')
        if self.title is not '':
            self.ax.set_title(self.title)     
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')		   		

        self.setAxis()
        lock.release() 

    def setAxis(self):		
        #self.ax.axis('equal')   # this does not work with the new matplotlib 3    
        if self.axis_computed:	
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)  
            self.ax.set_zlim(self.zlim)                             
        self.ax.legend()
        self.fig.canvas.flush_events()            

    def drawTraj(self, traj, name, color='r', marker='.'):
        if self.queue is None:
            return
        self.queue.put((traj, name, color, marker))

    def updateMinMax(self, np_traj):
        xmax,ymax,zmax = np.amax(np_traj,axis=0)
        xmin,ymin,zmin = np.amin(np_traj,axis=0)        
        cx = 0.5*(xmax+xmin)
        cy = 0.5*(ymax+ymin)
        cz = 0.5*(zmax+zmin) 
        if False: 
            # update maxs       
            if xmax > self.xlim[1]:
                self.xlim[1] = xmax 
            if ymax > self.ylim[1]:
                self.ylim[1] = ymax 
            if zmax > self.zlim[1]:
                self.zlim[1] = zmax                         
            # update mins
            if xmin < self.xlim[0]:
                self.xlim[0] = xmin   
            if ymin < self.ylim[0]:
                self.ylim[0] = ymin        
            if zmin < self.zlim[0]:
                self.zlim[0] = zmin     
        # make axis actually squared
        if True:
            #smin = min(self.xlim[0],self.ylim[0],self.zlim[0])                                            
            #smax = max(self.xlim[1],self.ylim[1],self.zlim[1])
            smin = min(xmin,ymin,zmin)                                            
            smax = max(xmax,ymax,zmax)            
            delta = 0.5*(smax - smin)
            self.xlim = [cx-delta,cx+delta]
            self.ylim = [cy-delta,cy+delta]
            self.zlim = [cz-delta,cz+delta]      
        self.axis_computed = True   

    def plot_refresh(self, lock):        
        lock.acquire()          
        self.setAxis()     
        lock.release()

