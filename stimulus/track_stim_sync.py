__author__ = 'jkappel'

import datetime
from imaging_session import *
from shutil import copyfile
from stimulus_functions import *
import numpy as np

def run(
        exp_tag='H2BG6s-TEST',
        stim_params=None,
        json_file=None,
        addLoomGrating=False,
        randomize=False,
        pseudo_random_order=[]

):

    today = datetime.datetime.today().strftime('%Y%m%d')
    datedir = "F:/JJ/{}/".format(today)
    if not os.path.exists(datedir):
        os.mkdir(datedir)

    nfish = len([i for i in os.listdir(datedir) if i.startswith('fish')])
    filedir = os.path.join(datedir, 'fish{}'.format(nfish+1))
    os.mkdir(filedir)
    exp_id = today + '_{}_'.format(exp_tag) + 'fish' + str(nfish+1)
    stim_json = r'{0}/{1}.json'.format(filedir, exp_id)

    # if stim_params is None:
    #
    #     _ = copyfile(json_file, stim_json)
    #
    # else:
    #
    with open(stim_json, 'w') as outfile:
        json.dump(stim_params, outfile)

    imsess = ImagingSession(

        exp_id=exp_id,
        fps_plot=20,
        fps_track=200,
        repeat_time=600,
        repeats_total=2,
        camxy=(400, 400),
        filedir=filedir,
        show_plot=False,
        trigger='out',
        monitor='jj_display',
        rec_sleep=60,
        stim_json=stim_json,
        track_tail=True,
        addLoomGrating=addLoomGrating, # True if circular dot trajectory
        randomize=randomize, # True if circular dot trajectory
        pseudo_random_order=pseudo_random_order
        )
    imsess.start_session()


if __name__ == '__main__':

    stim_params = []
    #Use this one for original protocol.
    dot_params = create_multiple_freqs(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=20.,
        start_angle=225,
        size=0.4 * 4.,
        freqs=(0.75, 1.5, 3.0, 6.0, 60.),
        freq_size={1.5: [0.2 * 4., 0.8 * 4.], 60.: [0.2 * 4., 0.8 * 4.]}
    )

    #Use this one for tectum test, with num volumes 600 instead of 3000 (scanimage), repeat_time 120 instead of 600 and repeats_total=1 instead of 6
    test_dot_params = create_multiple_freqs(

       speed=.5 * 4.,
       rad=1.8 * 4.,
       delay=30.,
       start_angle=225,
       size=0.4 * 4.,
       freqs=(1.5, 60.),
       freq_size={1.5: [0.2 * 4., 0.8 * 4.], 60.: [0.2 * 4., 0.8 * 4.]}
    )


    # order should be an array containing 0 (real), 1 (continuous) and 2 (bout) in the desired order. This is just
    # used to give the correct name in the stimulus file.
    # order = np.loadtxt('F:/Katja/nat_stim_order_1min.txt')

    # The file at pos_path should contain all xy coordinates of the entire experiment, no delays or breaks should be indicated here.
    # Delays are added between each stimulus based on the number of stimuli indicated in the order file.
    # nat_params = create_natural_params(
    #     pos_path = 'F:/Katja/nat_stim_experiment_1min.txt',
    #     order = order,
    #     size = 0.4 * 4.,
    #     boutrate = 1.5,
    #     delay = 20.
    # )
    #stim_params.extend(nat_params)

    # pattern_params = create_rotate_pattern_params(
    #     'F:/Katja/randompattern_12.png',
    #     stimtime=22.6,
    #     boutrate=(1.5, 60.),
    #     delay=20.
    # )
    #stim_params.extend(pattern_params)

    delay = 30
    simple_dot_params = create_multiple_freqs(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=delay,
        start_angle=225,
        size=0.4 * 4.,
        freqs=[1.5, 60.]
    )
    #stim_params.extend(simple_dot_params)


    pizza_slices_r1 = create_pizza_slices(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=delay,
        start_angle=180,
        size=0.4 * 4.,
        freqs=[1.5, 60.],
        fraction=.25,
        ccw=True
    )
    pizza_slices_r2 = create_pizza_slices(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=delay,
        start_angle=90,
        size=0.4 * 4.,
        freqs=[1.5, 60.],
        fraction=.25,
        ccw=False
    )
    stim_params.extend(pizza_slices_r1)
    stim_params.extend(pizza_slices_r2)

    pizza_slices_l1 = create_pizza_slices(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=delay,
        start_angle=270,
        size=0.4 * 4.,
        freqs=[1.5, 60.],
        fraction=.25,
        ccw=False
    )
    pizza_slices_l2 = create_pizza_slices(
        speed=.5 * 4.,
        rad=1.8 * 4.,
        delay=delay,
        start_angle=360,
        size=0.4 * 4.,
        freqs=[1.5, 60.],
        fraction=.25,
        ccw=True
    )
    stim_params.extend(pizza_slices_l1)
    stim_params.extend(pizza_slices_l2)


    fl_params = create_flicker_params(

        onoffs=[(750., 750.)],
        stimtimes=[5.],
        angles=[90., 135., 270., 315.],
        rad=1.8 * 4,
        start_angle=225.,
        sizes=[0.4 * 4],
        delay=20.

    )
    stim_params.extend(fl_params)

    print(fl_params)
    # Inbal params
    # grating_params = [{
    #
    #     'phase': 0.01,
    #     'static_s': 10,
    #     'moving_s': 10,
    #     'sf': sf,
    #     'tex': tex,
    #     'ori': 135
    #
    # } for tex in ['sqr', 'sin'] for sf in [0.09, 0.12, 0.4, 0.7]]
    grating_params = [{

        'delay': delay,
        'phase': 0.01,
        'static_s': 10,
        'moving_s': 10,
        'sf': sf,
        'tex': tex,
        'ori': 135

    } for tex in ['sqr'] for sf in [0.4]]
    stim_params.extend(grating_params)
    dimming_params = [{

        'delay': 20.,
        'duration': 2.,
        'inverse': inverse} for inverse in [True, False]]
    #stim_params.extend(dimming_params)


    #stim_params.append(loom_params)

    #To add loom and grating at end of each session set AddLoomGrating to True in imaging_session.
    #stim_params.extend(fl_params)

    #FOR NATURAL TRACK UNCOMMENT THIS and set both AddLoomGrating and Randomize to False in imaging_session
    #stim_params=nat_params

    # Used for the experiment with natural traj, bouting dots over a circle, rotating pattern and flicker stimuli. Numbers assume 18 natural trajectories to be added to the stim_params first.
    stimulus_order = [[22, 19, 0, 25, 24, 27, 20, 23, 21, 1, 26, 18],
                      [25, 3, 27, 21, 23, 26, 18, 24, 19, 22, 20, 2],
                      [20, 24, 21, 19, 23, 22, 26, 5, 18, 25, 27, 4],
                      [6, 22, 23, 27, 20, 21, 7, 24, 18, 26, 19, 25],
                      [27, 25, 24, 26, 8, 9, 21, 18, 19, 23, 20, 22],
                      [25, 18, 26, 11, 19, 21, 27, 22, 10, 24, 20, 23],
                      [21, 13, 26, 12, 24, 23, 27, 25, 18, 22, 20, 19],
                      [14, 21, 25, 26, 20, 23, 22, 19, 27, 15, 24, 18],
                      [21, 19, 27, 17, 23, 25, 18, 16, 22, 20, 24, 26]]

    #run(exp_tag='I2bGAL4-UAS-GCaMP6s-8dpf', stim_params=stim_params, json_file=None, addLoomGrating=False, randomize=False, pseudo_random_order=stimulus_order)

    order = np.loadtxt('F:/Katja/nat_stim_order_fish_only.txt')
    nat_exp_params = create_natural_params(
        pos_path = 'F:/Katja/nat_stim_experiment_2min_fish_only.txt',
        order = order,
        size = 0.4 * 4.,
        boutrate = 1.5,
        delay = 30.
    )
    #stim_params.extend(nat_exp_params[:2])
    loom_params = {
        'stim_size': 0.1,
        'max_size': 120.,
        'speed_loom': 1,
        'delay': delay}
    stim_params.append(loom_params)
    stim_params.append(loom_params)

    run(
        exp_tag='elavH2BGCaMP6s-6dpf',
        stim_params=stim_params,
        json_file=None,
        randomize=False,
        pseudo_random_order=[]

    )

    #NOTE: SET REPEATS TOTAL BACK TO 6!!!!!!!!!!!
