import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import code

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
    args = vars(ap.parse_args())


    x = np.linspace(0, 10, 100)
    axes = AxesSequence()
    for i, ax in zip(range(10), axes):
        print '{0:04d}'.format(i)
        print'crop/crop{0:04d}.tif'.format(i)
        img=cv2.imread('crop/crop{0:04d}.tif'.format(i))

        code.interact(local=locals())
        ax.imshow(img)


        ax.set_title(i)
        # ax.plot(x, np.sin(i * x))
        # ax.set_title('Line {}'.format(i))
    # for i, ax in zip(range(5), axes):
    #     ax.imshow(np.random.random((10,10)))
    #     ax.set_title('Image {}'.format(i))
    axes.show()

class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('scroll_event', self.on_mousescroll)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8],
                               visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_mousescroll(self, event):
        if event.button == 'down':
            self.next_plot()
        elif event.button == 'up':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes):
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()

if __name__ == '__main__':
    main()
