import time
import os
from ximea import xiapi
import subprocess
from matplotlib import pyplot as plt
from rotifera_tracker_helpers import Canvas
from tracking_functions import *
from psychopy import visual, core, monitors

import pandas as pd
import pickle
from multiprocessing import Process, Queue, Value
import copy
import random
import u3
from stimulus_functions import *
import json
import datetime

def init_monitor(

        name=u'katja_monitor',
        fullscr=True,
        warp=False,
        units='cm'

):

    my_monitor = monitors.Monitor(name)
    #
    # my_monitor = monitors.Monitor('default', width=6.93, distance=2)
    # my_monitor.setSizePix((1066, 800))
    # width = my_monitor.getWidth()
    px, py = my_monitor.getSizePix()
    screen_size = my_monitor.getSizePix()

    mywin = visual.Window(
        size=screen_size,
        fullscr=fullscr,
        screen=-1,
        allowGUI=True,
        allowStencil=False,
        monitor=my_monitor,
        color=[0, 0, 0],
        colorSpace=u'rgb',
        blendMode=u'avg',
        useFBO=True,
        units=units,
        rgb=list(np.asarray([0, 0, 0])))
    if warp:
        warper = windowwarp.Warper(
           mywin,
           warp='spherical',
           warpfile="",
           warpGridsize=512,
           eyepoint=[0.5, 0.5],
           flipHorizontal=False,
           flipVertical=False)
    return mywin


def present_stimuli(
                    q,
                    monitor,
                    timestamp,
                    stimuli,
                    repeat_time,
                    stimfile,
                    register=5000,
                    trigger='in',
                    randomize=False,
                    pseudo_random_order=[],
                    draw_start=True,
                    draw_end=True,
                    rno=0,
                    warp=False
):
    print('Initiating monitor')
    mywin = init_monitor(name=monitor, warp=warp)
    print('Starting stimulus presentation...')
    protocol = list()
    u = u3.U3()

    time.sleep(.1)
    trigger_microscope(u, register, trigger=trigger)
    q.put('Stimulus')
    clock = core.Clock()

    while clock.getTime() < (repeat_time):

        stim_instance = copy.deepcopy(stimuli)
        if randomize:
            random.shuffle(stim_instance)

        if len(pseudo_random_order) > 0:
            stim_instance = [stim_instance[i] for i in pseudo_random_order[rno]]
        # elif 'type' in stimuli[0][1]:
        #     stim_instance = stim_instance[rno * 4:(rno+1) * 4] #TODO remove hardcoded 4 natural stim per repeat.

        if clock.getTime() > repeat_time:
            break

        for xys, params in stim_instance:

            if clock.getTime() > repeat_time:
                break

            trackstart = timestamp.value
            print('Stim time: ', clock.getTime(), 'Track start: ', trackstart)
            print(params)
            #print('Size :', params['size'] / 4, 'Bout frequency :', params['boutrate'])

            if 'image' in params.keys():

                tstart, tstop = display_pattern(mywin, clock, params['imagepath'], ccw=params['ccw'],
                                                  phase=params['phase'], size=[25,25], boutfreq=params['boutrate'],
                                                  static_s=params['delay'], moving_s=params['stimtime'])
                params['tstart'] = tstart
                params['tstop'] = tstop

            elif 'movie' in params.keys():

              ft = display_frames(
                mywin,
                clock,
                params['movie'],
                frate=params['frate'],
                size=params['size'],
                transpose=params['transpose'],
                nframes=params['nframes']

              )
              params['frametimes'] = ft
            elif 'sf' in params.keys():

                params = display_grating(
                    mywin,
                    clock,
                    size=[2000, 2000],
                    phase=params['phase'],
                    ori=params['ori'],
                    sf=params['sf'],
                    tex=params['tex'],
                    static_s=params['static_s'],
                    moving_s=params['moving_s']
                )

            elif 'max_size' in params.keys():

                params = display_loom(
                    mywin,
                    clock,
                    max_size=params['max_size'],
                    speed_loom=params['speed_loom'],
                    delay=params['delay']
                )

            elif 'inverse' in params.keys():

                params = dimming(

                    mywin,
                    clock,
                    frate=60.,
                    delay=params['delay'],
                    duration=params['duration'],
                    inverse=params['inverse']
                )
            elif 't_on' in params.keys():

                print('FLICKER')
                params = flicker_dot_single(mywin, clock, **params)

            elif not 't_on' in params.keys():

                print('Circular stim')
                tstart, tpoints = display_dot(
                    xys,
                    mywin,
                    clock,
                    size=float(params['size']),
                    delay=params['delay'],
                    draw_start=draw_start,
                    draw_end=draw_end

                )
                print(trackstart, tstart)

                params['tstart'] = tstart
                params['tpoints'] = tpoints
                params['trackstart'] = trackstart

            protocol.append(params)

            trackstart = timestamp.value
            print('Stim time: ', clock.getTime(), 'Track start: ', trackstart)

    core.wait(repeat_time - clock.getTime())
    df_stim = pd.DataFrame(protocol)
    pickle.dump(df_stim, open(stimfile, 'wb'))
    print('Pickle dumped: ', stimfile)

    u.writeRegister(register, 0)
    u.close()
    core.quit()
    mywin.close()


def startangle_tail(tail):

    dx, dy = tail[1, 0] - tail[0, 0], tail[1, 1] - tail[0, 1]
    rads = math.atan(dy / dx)
    degs = math.degrees(rads)
    return degs * (-1)


def trigger_microscope(
        u,
        register,
        trigger='in'
):

    if trigger == 'out':

        u.writeRegister(register, 0)
        time.sleep(.1)
        u.writeRegister(register, 5)
        trigval = u.readRegister(register)
        time.sleep(.1)
        u.writeRegister(register, 0)
        return

    elif trigger == 'in':

        print('Waiting for microscope trigger...')

        trig = 0

        u.writeRegister(6106, 0)  #

        while int(trig) != 1:

            trig = u.readRegister(register)

        return


class ImagingSession:

    def __init__(self, **kwargs):

        self.monitor = kwargs.get('monitor', 'katja_monitor')
        self.register = kwargs.get('register', None)
        self.trigger = kwargs.get('trigger', 'out')
        if self.register is None:

            if self.trigger == 'in':
                self.register = 6006  # FI06

            elif self.trigger == 'out':
                self.register = 5000  # DAC0

            else:
                pass

        self.track_tail = kwargs.get('track_tail', True)
        self.track_eyes = kwargs.get('track_eyes', False)
        self.show_plot = kwargs.get('show_plot', False)
        self.randomize = kwargs.get('randomize', False)
        self.warp_screen = kwargs.get('warp_screen', False)

        self.pseudo_random_order = kwargs.get('pseudo_random_order', [])
        self.draw_start = kwargs.get('draw_start', True)
        self.draw_end = kwargs.get('draw_end', True)

        self.repeats_total = kwargs.get('repeats_total', 8)
        self.repeat_time = kwargs.get('repeat_time', 600)
        self.rec_sleep = kwargs.get('rec_sleep', 60)

        self.camxy = kwargs.get('camxy', (400, 400))
        self.fps_plot = kwargs.get('fps_plot', 20)
        self.fps_track = kwargs.get('fps_track', 200)
        self.fps_screen = kwargs.get('fps_screen', 60)

        self.filedir = kwargs.get('filedir', None)
        self.exp_id = kwargs.get('exp_id', None)
        self.ffmpeg_exe = kwargs.get('ffmpeg_exe',
                                     r'C:\Program Files\ffmpeg\ffmpeg-20200115-0dc0837-win64-static\bin\ffmpeg.exe')
        self.stim_json = kwargs.get('stim_json', None)
        self.stim_params = kwargs.get('stim_params', None)
        self.n_repeats = 0
        self.all_fps = list()

        self.fproc = None
        self.vid_dir = None
        self.cam = None
        self.img = None
        self.tstamp = None

        self.baseline = np.nan
        self.head = np.nan
        self.tail = np.nan
        self.tail_length = np.nan
        self.mywin = None
        self.stimuli = list()

    def start_cam(self):

        self.cam = xiapi.Camera()
        self.img = xiapi.Image()
        print('Opening first camera... ')
        self.cam.open_device()

        self.cam.set_exposure(int(1E+6/self.fps_track))
        self.cam.set_width(self.camxy[0])
        self.cam.set_height(self.camxy[1])
        self.cam.start_acquisition()

        self.tstamp = time.strftime("%H%M")
        vid_file = str(self.exp_id) + '_' + str(self.tstamp) + '_vid_' + str(self.n_repeats) + '.avi'
        self.vid_dir = os.path.join(self.filedir, vid_file)
        print('Camera setup done!')

    def get_frame(self):

        self.cam.get_image(self.img)
        return self.img.get_image_data_numpy()

    def start_ffmpeg(self):

        dimension = '{0}x{1}'.format(np.long(self.camxy[0]), np.long(self.camxy[1]))
        command = [self.ffmpeg_exe,
                   '-y',
                   '-f', 'rawvideo',
                   '-c:v', 'rawvideo',
                   '-s', dimension,
                   '-pix_fmt', 'gray',
                   '-i', 'pipe:0',
                   '-an',
                   '-r', str(self.fps_track),
                   '-c:v', 'mpeg4',
                   '-b:v', '5000k',
                   '-q:v', '1',
                   '-vsync', '0',
                   self.vid_dir]

        self.fproc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=None, stderr=None)

    def draw_tail(self):

        self.cam.get_image(self.img)
        frame = self.img.get_image_data_numpy()

        fig = plt.figure(7, (8, 8))
        ax = fig.add_subplot(111)
        plt.imshow(frame, origin='upper', cmap='Greys')
        cnv = Canvas(ax)
        plt.connect('button_press_event', cnv.update_path)
        plt.connect('motion_notify_event', cnv.set_location)
        plt.title('Define head-tail axis')
        plt.xlim(-5, np.shape(frame)[1]+5)
        plt.ylim(np.shape(frame)[0]+5, -5)
        plt.show()

        self.tail = np.array([[int(i[0]), int(i[1])] for i in cnv.vert])
        self.head = self.tail[0]
        self.tail_length = self.tail[1,0] - self.tail[0,0]+10

        self.baseline = startangle_tail(self.tail)
        print(self.baseline, self.tail)

    def record(self, stim_q, timestamp):

        tracklist = list()
        message = ''
        while message != 'Stimulus':

            message = stim_q.get()

        start = time.time()
        now = time.time() - start
        nframes = 0

        print('Start recording...', start)
        while now <= self.repeat_time:

            frame = self.get_frame()
            self.fproc.stdin.write(frame.tostring())
            now = time.time() - start
            timestamp.value = now

            dots, tailangle, dots_cv = tail_fitting(frame, self.head, self.tail_length, self.baseline)

            self.all_fps.append(nframes / (now + .0000001))
            tracklist.append({

                'Tail position': dots,
                'Tail angle': tailangle,
                'Time point': now,
                'Repeat#': self.n_repeats,

            })

            nframes += 1
            #tailangles = [i['Tail angle'] for i in tracklist[:-self.fps_plot]]
            if nframes % self.fps_plot == 0:

                frame = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2RGB)
                for circle in dots_cv:
                    try:
                        cv2.circle(frame, circle, 2, (0, 0, 255), thickness=1, lineType=8, shift=0)
                    except TypeError:
                        pass
                cv2.putText(frame, "%d fps" % int(self.all_fps[-1]), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255))
                cv2.putText(frame, "%d frame" % nframes, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255))
                cv2.putText(frame, "%d time" % int(now), (150, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255))

                cv2.imshow("im", frame)
                key = cv2.waitKey(1)
                #self.plot_q.put([tailangles, frame, dots_cv, nframes, now])

        cv2.destroyWindow("im")

        df_tailfit = pd.DataFrame(tracklist)
        trackfile = str(self.exp_id) + '_' + str(self.tstamp) + '_tailfit_' + str(self.n_repeats) + '.p'
        trackfile = os.path.join(self.filedir, trackfile)
        pickle.dump(df_tailfit, open(trackfile, 'wb'))

        # stop data acquisition
        print('Stopping acquisition...')

        self.fproc.stdin.close()
        self.fproc.wait()

        self.n_repeats += 1
        for i in range(self.rec_sleep):
            time.sleep(1)

    def start_session(self):

        print('Starting session')
        self.generate_stimuli()

        if self.track_tail:

            self.start_cam()
            self.draw_tail()
            self.cam.stop_acquisition()
            self.cam.close_device()

        for rno, repeat in enumerate(range(self.repeats_total)):

            if self.track_tail:

                self.start_cam()
                self.start_ffmpeg()

            stimfile = str(self.exp_id) + '_' + str(self.tstamp) + '_stimuli_' + str(self.n_repeats) + '.p'
            stimfile = os.path.join(self.filedir, stimfile)

            print('Repeat ' + str(rno))
            timestamp = Value('d', 0.0)

            stim_q = Queue(maxsize=1)
            if self.show_plot:

                plot_q = Queue()

            p_stim = Process(
                target=present_stimuli,
                args=(
                    stim_q,
                    self.monitor,
                    timestamp,
                    self.stimuli,
                    self.repeat_time,
                    stimfile
                ),
                kwargs={

                    'register': self.register,
                    'trigger': self.trigger,
                    'randomize': self.randomize,
                    'pseudo_random_order': self.pseudo_random_order,
                    'rno': rno,
                    'draw_start': self.draw_start,
                    'draw_end': self.draw_end,
                    'warp': self.warp_screen
                }
            )
            p_stim.start()
            if self.track_tail:
                self.record(stim_q, timestamp)

                self.cam.stop_acquisition()
                self.cam.close_device()

    def generate_stimuli(self):

        if self.stim_params is None:
            stim_params = json.load(open(self.stim_json, 'r'))
        else:
            stim_params = self.stim_params
        print(stim_params, '!!!')
        loaded_nat_path = False
        for p_idx, pr in enumerate(stim_params):
            if 'pos_path' in pr.keys():
                if loaded_nat_path is False:
                    all_xys = np.loadtxt(stim_params[p_idx]['pos_path'])
                    loaded_nat_path = True
                xys, params = natural_path(
                    frate = self.fps_screen,
                    all_xys = all_xys,
                    size = pr['size'],
                    type = pr['type'],
                    boutrate = pr['boutrate'],
                    delay = pr['delay'],
                    idx = p_idx,
                    num_stim = pr['num_stim']
                )
                self.stimuli.append((xys, params))

            elif 'boutrate' in pr.keys():

                if 't_on' not in pr.keys():

                    xys, params = circular_step(
                        frate=self.fps_screen,
                        speed=pr['speed'],
                        rad=pr['rad'],
                        delay=pr['delay'],
                        start_angle=pr['start_angle'],
                        ccw=pr['ccw'],
                        boutrate=pr['boutrate'],
                        size=pr['size'],
                        offset=pr['offset'],
                        fraction=pr['fraction']
                    )
                    self.stimuli.append((xys, params))

                else:

                    self.stimuli.append((None, pr))

            else:

                self.stimuli.append((None, pr))


if __name__ == '__main__':

    mywin = init_monitor(name=u'jj_display', fullscr=True)
    dot_params = create_flicker_params(

        onoffs=((333., 333.), (300., 500.)),
        stimtimes=(5., 5.),
        angles=(90., 270.),
        rad=1.8 * 4,
        start_angle=225.,
        sizes=[0.4 * 4],
        delay=1.

    )
    for params in dot_params:

        flicker_dot_single(mywin, core.Clock(), **params)
    pass
