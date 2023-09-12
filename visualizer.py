import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    
    def draw(self):
        for i in range(len(self.data)):
            y = np.array(self.data[i][0:len(self.data[i]):self.interval//100])
            x = np.arange(0, len(y) * self.interval, self.interval)
            self.axs[i].plot(x, y)

        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
  
        # giving a title to my graph
        plt.title(self.title)
  
        # function to show the plot
        plt.draw()
        plt.pause(0.01)

    def plot(self):
        self.x = np.arange(0, len(self.data) * 100, 100)
        self.y = np.array(self.data)
        plt.plot(self.x, self.y)

        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
  
        # giving a title to my graph
        plt.title(self.title)
  
        # function to show the plot
        plt.show()
    
    def __init__(self, title, interval = 500, maxlen = 10000, subplots=1):
        plt.style.use('dark_background')
        self.data = []
        for x in range(subplots):
           self.data.append([])
        self.interval = interval
        self.maxlen = maxlen
        self.last = 0
        self.title = title
        self.fig, self.axs = plt.subplots(subplots, sharex=True, sharey=True)

    def getInterval(self):
        return self.interval

    def updateData(self, *values):
        for idx, value in enumerate(values):
            self.data[idx].append(value)



