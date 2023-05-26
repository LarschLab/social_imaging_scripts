from psychopy import visual, core, monitors
import numpy as np
from itertools import chain
import sys
sys.path.insert(1, 'C:/Users/jkappel/PycharmProjects/track_stim_sync')
import sparse_noise
import cv2
from psychopy.visual import windowwarp
from psychopy import core
"""
Functions for displaying stimuli
"""


def draw_cross(t, mywin):

    line1 = visual.Line(win=mywin, start=(-10, -10), end=(10, 10), lineWidth=10)
    line2 = visual.Line(win=mywin, start=(-10, 10), end=(10, -10), lineWidth=10)

    for s in range(t):
        line1.draw()
        line2.draw()
        mywin.flip()
        core.wait(1)

    return


def natural_path(

        frate=60.,
        all_xys=[[]],
        size=.5,
        type='real',
        boutrate=1.5,
        delay = 0,
        idx = 0,
        num_stim = 1
):

    stim_len = int(all_xys.shape[0] / num_stim)
    xys = all_xys[idx * stim_len:(idx+1)*stim_len, :]
    xys = 0.028 * (np.array(xys) - 250) #Normalize to screen

    params = {
        'size': size,
        'type': type,
        'frate': frate,
        'boutrate': boutrate,
        'delay': delay
    }

    return xys, params


def circular_step(

        delay,
        speed=10.,
        ccw=False,
        rad=10.,
        size=.5,
        frate=60.,
        boutrate=2.,
        start_angle=None,
        offset=0,
        fraction=1.,
):
    """

    :param delay: int, delay in sec before stimulus start
    :param speed: float, stimulus speed in cm/sec
    :param ccw: bool, stimulus direction True=counter-clockwise
    :param rad: float, radius of circle
    :param size: float, diameter of stimulus
    :param frate: float, frame rate in fps
    :param boutrate: float, # of jumps per second
    :param start_angle: int, dot starting point in degrees, interpreted as in frontal view of the animal
    :param offset: int, offset from frontal view in degrees
    :param fraction: float, factor to reduce the trajectory of the dot stimulus to a specific fraction
    :return: xy locations of dot stimulus per frame, stimulus params
    """

    rad = float(rad)  # cm
    speed = float(speed)  # cm/s
    boutrate = float(boutrate)  # 1/s

    circ = float(2. * np.pi * rad) * fraction  # circumference
    period_rot = circ / float(speed)  # s (sec per rotation)

    ang_velocity = float(2. * np.pi) * fraction / float(period_rot)  # radians/s
    ang_velocity_f = ang_velocity * (1. / float(frate))  # radians/frame

    if start_angle is None:

        start_angle = np.deg2rad(0) + np.deg2rad(offset)

    else:

        start_angle = np.deg2rad(start_angle) + np.deg2rad(offset)

    x = rad * np.cos(start_angle)  # Starting x coordinate
    y = rad * np.sin(start_angle)  # Starting y coordinate

    xys = []
    bout = []

    nframes = 2 * np.pi * fraction / ang_velocity_f  # 2 * pi for whole rotation
    interval = round(frate / boutrate)

    new_angle = start_angle
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
        'size': size,
        'start angle': start_angle,
        'offset': offset,
        'fraction': fraction
    }

    return list(chain(*xys)), params

def display_pattern(
        mywin,
        clock,
        image,
        ccw=True,
        phase=.1,
        size=[200, 200],
        boutfreq=60.,
        ori=0,
        static_s=30,
        moving_s=23

):
    DOstim = visual.GratingStim(
        win=mywin,
        tex=image,
        size=size,
        ori=ori,
        pos=[0, 0],
        color=[1, 1, 1],
        colorSpace=u'rgb',
        units='cm',
        sf=(1/size[0], 1/size[1])
    )
    if ccw:
        dir = '+'
    else:
        dir = '-'

    print('image info: ', phase, size, boutfreq)
    now_static = clock.getTime()
    times = clock.getTime()

    while times < now_static + static_s/2:
        times = clock.getTime()
        mywin.flip()

    while times < now_static + static_s:
        times = clock.getTime()
        DOstim.draw()
        mywin.flip()

    now_moving = clock.getTime()
    while times < now_moving + moving_s:
        times = clock.getTime()
        DOstim.setOri(phase, dir)
        DOstim.draw()
        mywin.flip()
        core.wait(1/boutfreq)
    image_stop = clock.getTime()

    return now_static, image_stop


def display_grating(

        mywin,
        clock,
        dir='+',
        temporal_frequency=5,  # Hz
        phase=.1,
        size=[200, 200],
        sf=0.12,
        ori=315,
        static_s=20,
        moving_s=20,
        tex='sqr',
        units='cm'

):

    core.wait(2)
    DOstim = visual.GratingStim(
        win=mywin,
        tex=tex,
        size=size,
        sf=sf,
        ori=ori,
        pos=[0, 0],
        color=[1, 1, 1],
        colorSpace=u'rgb',
        units=units,
        phase=phase
    )
    print('grating info: %.2f Hz, SF: %.2f' % (float(phase*60), sf))
    now_static = clock.getTime()
    times = clock.getTime()

    while times < now_static + static_s:
        times = clock.getTime()
        DOstim.draw()
        mywin.flip()

    now_moving = clock.getTime()
    while times < now_moving + moving_s:
        times = clock.getTime()
        DOstim.setPhase(phase, dir)
        DOstim.draw()
        mywin.flip()
    grating_stop = clock.getTime()

    return {

        'grating': True,
        'radius': None,
        'speed': temporal_frequency,
        'phase': phase,
        'ccw': None,
        'frate': None,
        'boutrate': None,
        'delay': static_s,
        'sf': sf,
        'start_static': now_static,
        'start_moving': now_moving,
        'stop': grating_stop,
        'tex': tex,
        'dir': dir,
        'ori': ori

    }


def display_radial(

        mywin,
        clock,
        dir='+',
        phase=.1,
        size=[200, 200],
        ori=135,
        static_s=20,
        moving_s=20,
        radialCycles=3,
        angularCycles=4,
        radialPhase=0

):
    DOstim = visual.RadialStim(
        win=mywin,
        size=size,
        ori=ori,
        pos=[0, 0],
        color=[1, 1, 1],
        colorSpace=u'rgb',
        units='cm',
        angularPhase=phase,
        radialCycles=radialCycles,
        angularCycles=angularCycles,
        radialPhase=radialPhase
    )

    now_static = clock.getTime()
    times = clock.getTime()

    while times < now_static + static_s:
        times = clock.getTime()
        DOstim.draw()
        mywin.flip()

    now_moving = clock.getTime()
    while times < now_moving + moving_s:
        times = clock.getTime()
        DOstim.setAngularPhase(phase, dir)
        DOstim.draw()
        mywin.flip()
    grating_stop = clock.getTime()
    return


def display_dot(

        xys,
        mywin,
        clock,
        size=0.5,
        delay=15.,
        draw_start=True,
        draw_end=True

):
    x = xys[0][0]
    y = xys[0][1]
    tpoints = []

    dot = visual.Circle(win=mywin,
                        fillColor='black',
                        fillColorSpace=u'rgb',
                        lineColor='black',
                        lineColorSpace=u'rgb',
                        units='cm',
                        size=size,
                        pos=[x, y]
                        )

    dot.setAutoDraw(False)

    #angle_5V = 5 * ((xys[0][2] / (np.pi * 2 / 360)) + 270) / (450 + 270)
    # d.writeRegister(DAC0_REGISTER, angle_5V)

    t0 = clock.getTime()
    mywin.flip()
    for a in range(int(delay/2)):
        core.wait(1)

    if draw_start:
        dot.draw()
        mywin.flip()

    for a in range(int(delay/2)):
        core.wait(1)

    for xy in xys:
        dot.pos = xy[:2]

        #angle_5V = 5 * ((xy[2] / (np.pi * 2 / 360)) + 270) / (450 + 270)
        # d.writeRegister(DAC0_REGISTER, angle_5V)

        dot.draw()
        mywin.flip()

        t1 = clock.getTime()
        t_now = int(np.round(t1 - t0, 3) * 1000.)
        tpoints.append([t_now, xy])
    if draw_end:

        x = xys[0][0]
        y = xys[0][1]
        dot.pos = (x, y)
        dot.draw()
        mywin.flip()

    return t0, tpoints


def circular_step_multi(

        ndots=0,
        speed=10.,
        startangle=225,
        ccw=False,
        rad=10.,
        size=.5,
        frate=60.,
        boutrate=2.,
        delay=20.
):
    xys_super = []
    ccw = ccw  # direction
    rad = float(rad)  # cm
    speed = float(speed)  # cm/s
    boutrate = float(boutrate)  # 1/s

    circ = float(2. * np.pi * rad)  # circumference
    period_rot = circ / float(speed)  # s (sec per rotation)

    ang_velocity = float(2. * np.pi) / float(period_rot)  # radians/s
    ang_velocity_f = ang_velocity * (1. / float(frate))  # radians/frame

    circ = 2 * np.pi * rad
    if ndots == 0:
        ndots = int((circ / size) * .5)

    step = 360. / float(ndots)
    start_angles = [startangle + (step * i) for i in range(ndots)]
    print(start_angles)

    for start_angle in start_angles:

        angle = start_angle * (np.pi * 2 / 360)  # start point (top), previously  1.5*np.pi

        x = rad * np.cos(angle)  # Starting x coordinate
        y = rad * np.sin(angle)  # Starting y coordinate

        xys = []
        bout = []

        nframes = 2 * np.pi / ang_velocity_f  # 2* pi for whole rotation
        interval = round(frate / boutrate)

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
        xys = list(chain(*xys))
        xys_super.append(xys)

    params = {
        'radius': rad,
        'speed': speed,
        'ccw': ccw,
        'frate': frate,
        'boutrate': boutrate,
        'delay': delay,
        'size': size
    }

    return xys_super, params


def get_dot_locations(
        bouts,
        startangle=225.,
        prefangles=[0, 180]
):
    best_bouts = []
    bouts = np.array(bouts)
    bouts[:, 2] *= (360. / (np.pi * 2.))
    bouts_shifted = bouts[:, 2] - startangle
    print('Bouts shifted', [round(i) for i in bouts_shifted])
    #    if sum(bouts_shifted) < 0:
    #
    #        unique_angles = [360 + i for i in bouts_shifted]
    #
    #    else:

    unique_angles = [i for i in bouts_shifted]

    for prefangle in prefangles:
        bestidx = min(range(len(unique_angles)), key=lambda i: abs(unique_angles[i] - prefangle))
        best_bouts.append(bouts[bestidx])
    print(np.array(best_bouts)[:, 2] - 225)
    return best_bouts


def display_loom(

        mywin,
        clock,
        max_size=120.,
        refresh_rate=1. / 60.,
        speed_loom=120.,
        stim_size=0.,
        delay=20,

):
    #core.wait(delay)

    # DOstim = visual.Circle(mywin,
    #                        radius=stim_size,
    #                        edges=200,
    #                        fillColor='black',
    #                        fillColorSpace=u'rgb',
    #                        lineColor='black',
    #                        lineColorSpace=u'rgb',
    #                        units='deg')
    DOstim = visual.Circle(win=mywin,
                        fillColor='black',
                        fillColorSpace=u'rgb',
                        lineColor='black',
                        lineColorSpace=u'rgb',
                        units='cm',
                        size=.2,
                        pos=[0, 0]
                        )
    # DOstim.setPos([300, -300]) #Changed from 0,0
    DOstim.setPos([0, 0]) #Changed from 0,0
    DOstim.draw()
    mywin.flip()
    core.wait(delay)
    tstart = clock.getTime()
    print('speed', speed_loom)
    cnt = 1
    while stim_size < max_size:
        stim_size = stim_size + (speed_loom * refresh_rate)
        stim_size = stim_size + (speed_loom * refresh_rate * np.square(cnt))
        print(stim_size)
        DOstim.draw()
        DOstim.setSize(stim_size)
        mywin.flip()
        cnt=cnt+1
    print('Exit', stim_size, max_size
          )
    stop = clock.getTime()
    core.wait(delay)

    params = {
        'loom': True,
        'radius': stim_size,
        'max_size': max_size,
        'speed': speed_loom,
        'ccw': None,
        'frate': None,
        'boutrate': None,
        'delay': delay,
        'size': max_size,
        'start_moving': tstart,
        'stop': stop
    }

    return params


def flicker_dots(

        mywin,
        clock,
        rad=7.2,
        start_angle=225,
        t_on=180,
        t_off=400,
        stimtime=10.,
        frate=30.,
        size=1.6,
        delay=5.

):
    xys_start = []

    rad = float(rad)  # cm
    circ = float(2. * np.pi * rad)  # circumference
    ndots = int((circ / size) * .5)
    ndots = 2
    print('# dots: ', ndots)

    step = 360. / float(ndots)
    start_angles = [start_angle + (step * i) for i in range(ndots)]

    for start_angle in start_angles:
        angle = start_angle * (np.pi * 2 / 360)  # start point (top), previously  1.5*np.pi

        x = rad * np.cos(angle)  # Starting x coordinate
        y = rad * np.sin(angle)  # Starting y coordinate
        xys_start.append([x, y])

    print(t_on * frate / 1000., 'nframes ON')
    print(t_on * frate / 1000., 'nframes OFF')

    flicker_on = np.ones(int(round(t_on * frate / 1000.)))
    flicker_off = np.zeros(int(round(t_off * frate / 1000.)))

    flicker_onoff = np.concatenate((flicker_on, flicker_off), axis=0).flatten()
    flicker_trace = np.concatenate([flicker_onoff] * 1000, axis=0)

    dots = []
    for no in range(len(xys_start)):
        x = xys_start[no][0]
        y = xys_start[no][1]
        tpoints = []
        print(x, y)
        dot = visual.Circle(win=mywin,
                            fillColor='black',
                            fillColorSpace=u'rgb',
                            lineColor='black',
                            lineColorSpace=u'rgb',
                            units='cm',
                            size=size,
                            pos=[x, y]
                            )
        dot.setAutoDraw(False)
        dots.append([dot, no])
        dot.draw()

    mywin.flip()

    t0 = clock.getTime()
    for a in range(int(delay)):
        core.wait(1)

    for frameno in range(int(frate * stimtime)):

        for [dot, n] in (dots):
            dot.opacity = flicker_trace[frameno]
            dot.draw()
        mywin.flip()
    t1 = clock.getTime()

    params = {

        't_on': t_on,
        't_off': t_off,
        'flicker_boolean': flicker_trace,
        'frate': frate,
        'boutrate': None,
        'delay': delay,
        'size': size,
        'start_flicker': t0,
        'stop_flicker': t1

    }

    return


def flicker_dot_single(

        mywin,
        clock,
        t_on=180,
        t_off=400,
        stimtime=10.,
        frate=60.,
        pos=(0, 0),
        size=1.6,
        delay=5.,
        **kwargs

):
    print(t_on * frate / 1000., 'nframes ON')
    print(t_off * frate / 1000., 'nframes OFF')
    print(pos, 'xy dot position')

    flicker_on = np.ones(int(round(t_on / 1000. * frate)))
    flicker_off = np.zeros(int(round(t_off / 1000. * frate)))

    flicker_onoff = np.concatenate((flicker_on, flicker_off), axis=0).flatten()
    flicker_trace = np.concatenate([flicker_onoff] * int(stimtime * frate), axis=0)

    dot = visual.Circle(win=mywin,
                        fillColor='black',
                        fillColorSpace=u'rgb',
                        lineColor='black',
                        lineColorSpace=u'rgb',
                        units='cm',
                        size=size,
                        pos=pos
                        )
    dot.setAutoDraw(False)

    mywin.flip()
    tpoints = []
    t0 = clock.getTime()
    for a in range(int(delay/2)):
        core.wait(1)
    dot.draw()
    mywin.flip()
    for a in range(int(delay/2)):
        core.wait(1)

    for frameno in range(int(frate * stimtime)):
        dot.opacity = flicker_trace[frameno]
        dot.draw()
        mywin.flip()
        tpoints.append(clock.getTime())
    t1 = clock.getTime()
    print(t1)
    params = {

        't_on': t_on,
        't_off': t_off,
        'stimtime': stimtime,
        'flicker_tpoints': tpoints,
        'flicker_boolean': flicker_trace,
        'frate': frate,
        'boutrate': None,
        'delay': delay,
        'size': size,
        'pos': pos,
        'start_flicker': t0,
        'stop_flicker': t1

    }

    return params

def create_natural_params(
        pos_path = '',
        order = [0, 1, 2],
        size = 0.4 * 4.,
        boutrate = 1.5,
        delay = 30.
):
    #TODO raise error if stim_len times len(order) is not len(xys)

    base_dict = {
        'pos_path': pos_path,
        'size': size,
        'type': [],
        'boutrate': boutrate,
        'num_stim': len(order),
        'delay': delay
    }

    dot_params = list()
    for idx, type in enumerate(order):
        dict_instance = base_dict.copy()
        if type == 0:
            dict_instance['type'] = 'real'
            dict_instance['boutrate'] = 'real'
        elif type == 1:
            dict_instance['type'] = 'cont'
            dict_instance['boutrate'] = 30.
        elif type == 2:
            dict_instance['type'] = 'fish'
            dict_instance['boutrate'] = 'fish'
        else:
            print('error stim type {} not known.',format(i))

        dot_params.append(dict_instance)

    return dot_params

def create_multiple_freqs(

        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=20.,
        start_angle=225,
        size=0.4 * 4.,
        freqs=(0.75, 1.5, 3.0, 6.0, 60.),
        freq_size={},
        offset=0.,
        fraction=1
):
    base_dict = {
        'speed': speed,
        'rad': rad,
        'delay': delay,
        'start_angle': start_angle,
        'ccw': True,
        'boutrate': np.nan,
        'size': np.nan,
        'offset': offset,
        'fraction': fraction
    }
    dot_params = list()

    for freq in freqs:

        dict_instance = base_dict.copy()
        dict_instance['boutrate'] = freq
        dict_instance['size'] = size
        dict_instance['ccw'] = True
        dict_instance['offset'] *= -1

        dot_params.append(dict_instance)

        dict_instance = base_dict.copy()
        dict_instance['boutrate'] = freq
        dict_instance['size'] = size
        dict_instance['ccw'] = False

        dot_params.append(dict_instance)

        if freq in freq_size.keys():

            for fsize in freq_size[freq]:

                dict_instance = base_dict.copy()
                dict_instance['boutrate'] = freq
                dict_instance['size'] = fsize
                dict_instance['ccw'] = True

                dot_params.append(dict_instance)

                dict_instance = base_dict.copy()
                dict_instance['boutrate'] = freq
                dict_instance['size'] = fsize
                dict_instance['ccw'] = False

                dot_params.append(dict_instance)

    return dot_params


def create_pizza_slices(

        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=20.,
        start_angle=225,
        size=0.4 * 4.,
        freqs=(0.75, 1.5, 3.0, 6.0, 60.),
        freq_size={},
        offset=0.,
        fraction=1,
        ccw=True
):
    base_dict = {
        'speed': speed,
        'rad': rad,
        'delay': delay,
        'start_angle': start_angle,
        'ccw': ccw,
        'boutrate': np.nan,
        'size': np.nan,
        'offset': offset,
        'fraction': fraction
    }
    dot_params = list()

    for freq in freqs:

        dict_instance = base_dict.copy()
        dict_instance['boutrate'] = freq
        dict_instance['size'] = size

        dot_params.append(dict_instance)

        if freq in freq_size.keys():

            for fsize in freq_size[freq]:

                dict_instance = base_dict.copy()
                dict_instance['boutrate'] = freq
                dict_instance['size'] = fsize

                dot_params.append(dict_instance)


    return dot_params


def create_flicker_params(

        onoffs=((),()),
        stimtimes=(),
        angles=(180.),
        rad=1.8*4,
        start_angle=225,
        sizes=(0.4*4.),
        delay=20.


):
    dot_params = []
    base_dict = {

        'boutrate': np.nan,
        'stimtime': np.nan,
        'rad': rad,
        'delay': delay,
        'start_angle': start_angle,
        'ccw': None,
        't_on': np.nan,
        't_off': np.nan,
        'size': np.nan,
        'pos': (np.nan, np.nan),
        'speed': 1.8 * 0.4,
        'offset': 45,
        'fraction': 1
    }

    for angle in angles:

        x = np.cos(np.deg2rad(angle+start_angle)) * rad
        y = np.sin(np.deg2rad(angle+start_angle)) * rad

        for t_on, t_off in onoffs:

            for stimtime in stimtimes:

                for size in sizes:

                    dict_instance = base_dict.copy()
                    dict_instance['boutrate'] = 1./(t_on+t_off)
                    dict_instance['start_angle'] = start_angle
                    dict_instance['angle'] = angle
                    dict_instance['stimtime'] = stimtime
                    dict_instance['t_on'] = t_on
                    dict_instance['t_off'] = t_off
                    dict_instance['size'] = size
                    dict_instance['pos'] = (x,y)
                    dict_instance['delay'] = delay
                    dot_params.append(dict_instance)

    return dot_params


def create_rotate_pattern_params(
        imagepath,
        stimtime=23,
        boutrate=1.5,
        delay=20.
):
    base_dict = {
        'imagepath': imagepath,
        'phase': np.nan,
        'stimtime': stimtime,
        'delay': delay,
        'image': True,
        'size': np.nan,
        'boutrate': np.nan
    }

    pattern_params = list()

    for freq in boutrate:
        phase = 360 / (freq * stimtime)

        dict_instance = base_dict.copy()
        dict_instance['phase'] = phase
        dict_instance['ccw'] = True
        dict_instance['boutrate'] = freq

        pattern_params.append(dict_instance)

        dict_instance = base_dict.copy()
        dict_instance['phase'] = phase
        dict_instance['ccw'] = False
        dict_instance['boutrate'] = freq

        pattern_params.append(dict_instance)

    return pattern_params


def dimming(

        mywin,
        clock,
        frate=60.,
        delay=20.,
        duration=5.,
        inverse=True
):

    if inverse:
        l = (-1, -1, -1)
    else:
        l = (1, 1, 1)
    rect = visual.Rect(
        win=mywin, pos=(0, 0),
        lineWidth=0,
        fillColor=l,
        fillColorSpace='rgb',
        size=(mywin.size[0], mywin.size[1])
    )
    rect.draw()
    mywin.flip()
    steps_size = 2/(duration * frate)
    lum_steps = np.arange(-1, 1 + steps_size, steps_size)

    if not inverse:
        lum_steps = lum_steps[::-1]

    t0 = clock.getTime()
    print('Dimming start', t0)

    core.wait(delay)
    step = 0
    while clock.getTime() - t0 - delay < duration:

        rect.fillColor = [lum_steps[step] for i in range(3)]
        rect.draw()
        mywin.flip()
        step += 1

    t1 = clock.getTime()

    params = {

        'dim': True,
        'radius': np.nan,
        'max_size': np.nan,
        'speed': np.nan,
        'ccw': np.nan,
        'frate': np.nan,
        'boutrate': np.nan,
        'delay': delay,
        'size': np.nan,
        'start_static': t0,
        'start_moving': t0 + delay,
        'stop': t1,
        'inverse': inverse,
        'duration': duration

    }

    return params


def display_frames(
        mywin,
        clock,
        ts,
        frate=1/2,
        size=(1280, 740),
        nframes=None,
        transpose=True
):
    core.wait(1)
    stim = visual.ImageStim(
        mywin,
        units="pix",
        colorSpace='rgb',
        size=size
        )
    if isinstance(ts, str):
        ts = np.load(ts)
    ft = np.zeros(shape=(ts.shape[0])) * np.nan
    for i in range(ts[:nframes].shape[0]):

        ch = ts[i]
        if transpose:
            ch = ch.T
        ch = np.flip(cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB), 0)
        ch = ch.astype(float) / 255
        stim.setImage(ch)
        stim.draw()
        mywin.flip()
        ti = clock.getTime()
        ft[i] = ti
        core.wait(1/frate)
    core.wait(30)
    return ft

if __name__ == "__main__":

    #ims = sparse_noise.create_spots(nits=1000000, spotsize=4, ndots=24)
    #ims = np.load('sparse_ts.npy').astype(np.uint8)
    #print(ims.shape)
    my_monitor = monitors.Monitor('default', width=6.93, distance=2)
    my_monitor.setSizePix((1280, 740))
    width = my_monitor.getWidth()
    px, py = my_monitor.getSizePix()
    clock = core.Clock(

    )
    mywin = visual.Window(
        size=(1280, 740),
        fullscr=True,
        screen=-1,
        allowGUI=True,
        allowStencil=False,
        monitor=my_monitor,
        color=[0, 0, 0],
        colorSpace=u'rgb',
        blendMode=u'avg',
        useFBO=True,
        units='pix')


    ch = np.zeros(shape=(1280, 740), dtype=np.uint8).T
    step = int(ch.shape[1]/24)
    for k in [0, step]:
        for i in np.arange(0, ch.shape[0], step*2):
            for j in np.arange(0, ch.shape[1], step*2):

                ch[int(i)+k:int(i)+step+k, int(j)+k:int(j)+step+k] = 255

    #ch = warp.rev_warp(ch, plot=False).T
    import tifffile as tiff
    #ts = tiff.imread('C:/Users/jkappel/Desktop/natscene.tif')
    #ch = ts[0]
    warper = windowwarp.Warper(mywin,
                    warp='spherical',
                    warpfile="",
                    warpGridsize=512,
                    eyepoint=[0.5, 0.5],
                    flipHorizontal=False,
                    flipVertical=False)

    ch = np.flip(cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB), 0)
    ch = ch.astype(float) / 255

    stim = visual.ImageStim(
        mywin,
        image=ch,
        units="pix",
        colorSpace='rgb',
        size=(
            1280,
            740,
        ))
    clock = core.Clock()
    stim.draw()
    mywin.flip()
    core.wait(2000)
    ft = display_frames(mywin, clock, np.array(ims[:]), frate=2)
    print(ft, np.diff(ft.reshape(1, -1)))
    grating_params = [{

        'phase': 1/60,
        'static_s': .5,
        'moving_s': 1,
        'sf': sf,
        'tex': tex,
        'ori': ori

    } for tex in ['sqr'] for sf in [0.04*120/width] for ori in np.arange(0, 360, 360/12)]

    for params in grating_params:
        display_grating(
            mywin,
            clock,
            size=[width, width*(py/px)],
            phase=params['phase'],
            ori=params['ori'],
            sf=params['sf'],
            tex=params['tex'],
            static_s=params['static_s'],
            moving_s=params['moving_s'],
            units='cm'
        )
    import glob
    for vpath in glob.glob(r'J:\Johannes Kappel\Stimuli data\NatScenes\2 - ImageJ versions\*.npy'):
        natscene = np.load(vpath)
        ft = display_frames(mywin, clock, np.array(natscene), frate=30, transpose=False, size=(1280, 740))
