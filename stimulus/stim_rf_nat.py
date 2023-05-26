__author__ = 'jkappel'

from imaging_session import *
import numpy as np
import glob


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
    #stim_json = r'{0}/{1}.json'.format(filedir, exp_id)

    # if stim_params is None:
    #
    #     _ = copyfile(json_file, stim_json)
    #
    # else:
    #
    #     with open(stim_json, 'w') as outfile:
    #         json.dump(stim_params, outfile)

    imsess = ImagingSession(

        exp_id=exp_id,
        fps_plot=20,
        fps_track=200,
        repeat_time=1200,
        repeats_total=1,
        camxy=(400, 400),
        filedir=filedir,
        show_plot=False,
        trigger='out',
        monitor='default',
        warp_screen=True,
        rec_sleep=60,
        stim_json='',
        stim_params=stim_params,
        track_tail=False,
        addLoomGrating=addLoomGrating, # True if circular dot trajectory
        randomize=randomize, # True if circular dot trajectory
        pseudo_random_order=pseudo_random_order
        )
    imsess.start_session()


if __name__ == '__main__':



    stim_params = []
    vh_ims = r'J:\Johannes Kappel\Stimuli data\vanhateren_iml\van_hateren_ims_1.npy'
    natpaths = glob.glob(r'J:\Johannes Kappel\Stimuli data\NatScenes\2 - ImageJ versions\*.npy')
    sparse_ims = 'sparse_ts_dims128_74_size465_ndots16.npy'

    rep = 'rfs'
    #rep = 'vh'
    #rep = 'movies'

    if rep == 'movies':

        for vpath in natpaths[:12]:
            stim_params.append({'movie': vpath,
                                'frate': 60,
                                'size': (1280, 740),
                                'transpose': False,
                                'nframes': 60*60
                                })
    elif rep == 'rfs':

        grating_params = [{

            'phase': 1/60,
            'static_s': 5,
            'moving_s': 5,
            'sf': sf,
            'tex': tex,
            'ori': ori

        } for tex in ['sqr'] for sf in [0.04*120/6.93] for ori in np.arange(0, 360, 360/12)]
        stim_params.extend(grating_params)
        stim_params.extend(grating_params)

        stim_params.append({'movie': sparse_ims,
                            'frate': 2,
                            'size': (1280, 740),
                            'transpose': True,
                            'nframes': 900})
        stim_params.append({'movie': sparse_ims,
                            'frate': 2,
                            'size': (1280, 740),
                            'transpose': True,
                            'nframes': 900})

    elif rep == 'vh':

        stim_params.append({'movie': vh_ims,
                            'frate': 2,
                            'size': (1280, 740),
                            'transpose': False,
                            'nframes': 1200 * 2
                            })
    else:

        pass

    run(
        exp_tag='elavH2BGCaMP8m-7dpf',
        stim_params=stim_params,
        json_file=None,
        randomize=False

    )

    #NOTE: SET REPEATS TOTAL BACK TO 6!!!!!!!!!!!
