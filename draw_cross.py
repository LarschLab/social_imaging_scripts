from __future__ import division, print_function
from psychopy import visual, core, monitors
import numpy as np
from itertools import chain

###stimulus functions###
def draw_cross(t):
    
    line1 = visual.Line(win=mywin, start=(0,10), end=(0,-10), lineWidth=10)
    line2 = visual.Line(win=mywin, start=(10,0), end=(-10,0), lineWidth=10)
    
    line1 = visual.Line(win=mywin, start=(-10,10), end=(10,-10), lineWidth=10)
    line2 = visual.Line(win=mywin, start=(-10,-10), end=(10,10), lineWidth=10)
    for s in range(t):
        
        line1.draw()
        line2.draw()
        mywin.flip()
        core.wait(1)
        
    return


def circular_step(
                    speed=10., 
                    ccw=False, 
                    rad=10., 
                    size=.5,
                    frate=60.,
                    boutrate = 2.,
                    delay=20.
                ):
    
    ccw = ccw #direction
    rad = float(rad)#cm
    speed = float(speed) #cm/s
    boutrate = float(boutrate) # 1/s
    
    circ = float(2.*np.pi*rad) #circumference
    period_rot = circ/float(speed) # s (sec per rotation)
    
    ang_velocity = float(2.*np.pi)/float(period_rot) # radians/s
    ang_velocity_f = ang_velocity * (1./float(frate)) #radians/frame
    angle = 270*(np.pi*2/360) #start point (top), previously  1.5*np.pi
    
    x = rad * np.cos(angle) #Starting x coordinate
    y = rad * np.sin(angle) #Starting y coordinate

    xys = []
    bout = []
    
    nframes = 2*np.pi/ang_velocity_f #2* pi for whole rotation
    interval = round(frate/boutrate)

    new_angle = angle
    for frame in range(int(round(nframes) + 1)):

        if ccw:
            new_angle = new_angle - ang_velocity_f
        else:
            new_angle = new_angle + ang_velocity_f

        if frame % interval == 0:
            
            bout.append((x, y, new_angle))
            xys.append(bout)
            bout = []
            
            x = rad * np.cos(new_angle)
            y = rad * np.sin(new_angle)
            
        else:  
            bout.append((x, y, new_angle))
        
    params = {
        'radius': rad,
        'speed': speed,
        'ccw': ccw,
        'frate': frate,
        'boutrate': boutrate,
        'delay': delay,
        'size': size
    }
    return list(chain(*xys)), params


def display_dot(
                    xys,
                    size = 0.5,
                    delay = 1.

                    ):

    
    x = xys[0][0]
    y = xys[0][1]
    tpoints = []
    
    dot = visual.Circle(win=mywin, 
                        fillColor='black', 
                        fillColorSpace=u'rgb', 
                        lineColor='black', 
                        lineColorSpace= u'rgb', 
                        units='cm', 
                        size=size, 
                        pos=[x, y]
                        )
    
    dot.setAutoDraw(False)

    angle_5V = 5*((xys[0][2] / (np.pi *2/360)) + 270) / (450 + 270)
    #d.writeRegister(DAC0_REGISTER, angle_5V)
    
    dot.draw()
    mywin.flip()
    
    t0 = clock.getTime()
    for a in range(int(delay)):
        core.wait(1)
 
    for xy in xys:

        dot.pos = xy[:2]

        angle_5V = 5*((xy[2]/(np.pi*2/360)) + 270)/(450 + 270)
        #d.writeRegister(DAC0_REGISTER, angle_5V)
        
        dot.draw()
        mywin.flip()

        t1 = clock.getTime()
        t_now = int(np.round(t1-t0, 3)*1000.)
        tpoints.append([t_now, xy])

    x = xys[0][0]
    y = xys[0][1]
    dot.pos= (x, y)
    
    angle_5V = 5*((xys[0][2] / (np.pi * 2 / 360)) + 270) / (450 + 270)
    #d.writeRegister(DAC0_REGISTER, angle_5V)
    
    dot.draw()
    mywin.flip()
    return t0, tpoints




###setting up monitor###
my_monitor = monitors.Monitor(u'jj_display')
w = my_monitor.getWidth()
dist = my_monitor.getDistance()
width = my_monitor.getSizePix()[0]
height = my_monitor.getSizePix()[1]
screen_size = my_monitor.getSizePix()


###create a window###
mywin = visual.Window(size=screen_size, fullscr=True, screen=-1, allowGUI=True, allowStencil=False,
    monitor=my_monitor, color=[0,0,0], colorSpace=u'rgb',
    blendMode=u'avg', useFBO=True, units='cm', rgb=list(np.asarray([0,0,0])))


##show stimuli###
draw_cross(1000000)



mywin.close()
core.quit()