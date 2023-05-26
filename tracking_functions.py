import cv2
import math
import numpy as np
#from pyqtgraph.Qt import QtGui, QtWidgets
#import pyqtgraph as pg
import collections
import sys


def tail_fitting(
        frame,
        head,
        tail_length,
        baseline,
        num_points=9,
        resolution=20,
        filter=(7,7)
        ):

    tail_points = [head]
    tail_points_cv = [(head[0], head[1])]
    width = tail_length

    x = head[0]
    y = head[1]

    img_filt = cv2.boxFilter(frame, -1, filter)
    lin = np.linspace(0, np.pi, resolution)

    for _ in range(num_points):
        try:

            # Find the x and y values of the arc
            xs = x+width/num_points*np.sin(lin)
            ys = y+width/num_points*np.cos(lin)
            # Convert them to integer, because of definite pixels
            xs, ys = xs.astype(int), ys.astype(int)
            # ident = np.where(img_filt[ys,xs]==min(img_filt[ys,xs]))[0][0]
            ident = np.where(img_filt[ys,xs]==max(img_filt[ys,xs]))[0][0]
            # ident = np.argmin(img_filt[ys,xs])
            # print ident
            x = xs[ident]
            y = ys[ident]
            lin = np.linspace(lin[ident]-np.pi/2,lin[ident]+np.pi/2,20)
            # Add point to list
            tail_points.append([x,y])
            tail_points_cv.append((x,y))

        except IndexError:

            tail_points.append([np.nan,np.nan])
            tail_points_cv.append((np.nan,np.nan))

    tailangle = float(
        math.atan2(
            np.nanmean(np.asarray(tail_points)[-3:-1, 1]) - np.asarray(tail_points)[0, 1],
            np.nanmean(np.asarray(tail_points)[-3:-1, 0]) - np.asarray(tail_points)[0, 0]) * 180.0/3.1415
    ) - baseline
    return np.asarray(tail_points), tailangle*-1, tail_points_cv


def eyetracker_function_opencv(img, eyec, gv):

    ret, thresh = cv2.threshold(img, gv, 255,0)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours[1:]:
        if cv2.pointPolygonTest(cnt,(eyec[0, 0], eyec[0, 1]),False)>0:
            eyex1 = cnt
    for cnt in contours[1:]:
        if cv2.pointPolygonTest(cnt,(eyec[1, 0], eyec[1, 1]),False)>0:
            eyex2 = cnt
    (x1,y1),(MA1,ma1),angle1 = cv2.fitEllipse(eyex1)
    (x2,y2),(MA2,ma2),angle2 = cv2.fitEllipse(eyex2)
    return ((x1,y1),(MA1,ma1),angle1), ((x2,y2),(MA2,ma2),angle2)


class DynamicPlotter:

    def __init__(self, frame_rate=200, framecount=15000, size=(600,350), timewindow=False):

        self.app = QtWidgets.QApplication(sys.argv)
        self.size = size
        self.framecount = framecount
        self.framerate = frame_rate
        self.sampleinterval = 1.0/self.framerate
        if not timewindow:
            self.timewindow = int(framecount/self.framerate)
        else:
            self.timewindow = timewindow

        # Data stuff
        print(timewindow)
        self._interval = int(self.sampleinterval *1000)
        self._bufsize = int(self.timewindow/self.sampleinterval)
        self.databuffer = collections.deque([0.0]*self._bufsize, self._bufsize)
        self.x = np.linspace(-self.timewindow, 0.0, self._bufsize)
        self.y = np.zeros(self._bufsize, dtype=np.float)
        # PyQtGraph stuff
        self.plt = pg.plot(title='Dynamic Plotting with PyQtGraph')
        self.plt.resize(*self.size)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'Tailangle', 'deg')
        self.plt.setLabel('bottom', 'time', 's')
        self.plt.setRange(yRange=[-90,90])
        self.curve = self.plt.plot(self.x, self.y, pen=(255,0,0))
        print('wow')
        self.app.exec_()
        print('done')

    def updateplot(self, newy):

        for tp in newy:
            self.databuffer.append(tp)
        self.y[:] = self.databuffer
        self.curve.setData(self.x, self.y)
        self.app.processEvents()

    def run(self):

        self.app.exec_()

    def close(self):
        self.app.closeAllWindows()


def run_plot(

        fps_track=200,
        repeat_time=600,
):

    plotter = DynamicPlotter(
        frame_rate=fps_track,
        framecount=repeat_time * fps_track,
        size=(1000, 350),
        timewindow=3
    )

    plotter.run()
    plot_flag = True
    while plot_flag:

        tailangles, dots_cv, frame, nframes, now = plot_q.get()
        plotter.updateplot(tailangles)
        frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
        for circle in dots_cv:
            cv2.circle(frame, circle, 2, (0, 0, 255), thickness=1, lineType=8, shift=0)

        cv2.putText(frame, "%d fps" % int(all_fps[-1]), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(frame, "%d frame" % nframes, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(frame, "%d time" % int(now), (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))

        cv2.imshow("im", frame)
        key = cv2.waitKey(1)

    plotter.close()
    cv2.destroyWindow("im")

if __name__ == '__main__':

    pass
