import numpy as np
import sys
sys.path.insert(1, 'C:/Users/jlarsch/Documents/social_imaging_scripts/analysis')
from suite2p import classification
from suite2p.run_s2p import default_ops, run_s2p
import os
import tifffile as tiff
import skimage.io
from skimage.filters import gaussian
import shutil
import glob, re
import utils as Utils
from scipy import interpolate


def detect_bidi_offset(plane, maxoff=5, zrange=(0, -1)):
    '''
    # Function written by Joe Donovan for bidirectional scan offsets correction
    :param plane:
    :return:
    '''

    aplane = plane[zrange[0]:zrange[1]].mean(0)
    offsets = np.arange(-maxoff+1, maxoff)
    x = np.arange(aplane.shape[-1])
    minoff = []
    for row in range(2, plane.shape[-2]-1, 2):
        f = interpolate.interp1d(x, (aplane[row + 1, :] + aplane[row - 1, :]) * .5, fill_value='extrapolate')

        offsetscores = np.asarray([np.mean(np.abs(aplane[row, :] - f(x + offset))) for offset in offsets])
        minoff.append(offsetscores.argmin())
    bestoffset = offsets[np.bincount(minoff).argmax()]
    print(bestoffset)
    return bestoffset


def bidi_offset_correction_plane(plane, offset=None, maxoff=5, zrange=(0, -1)):
    if not offset:
        offset = detect_bidi_offset(plane, maxoff=maxoff, zrange=zrange)
    plane[:, 1::2, :] = np.roll(plane[:, 1::2, :], -offset, -1)  # .astype(aplane.dtype)
    return


class ROITracesExtraction:
    def __init__(self, **kwargs):

        """

        :param path: str, path to the imaging folder containing date folders.
        :param inputdict: dict, example below
                  inputdict = {
                      '20190402': {
                          'nfish': 1,
                          'nplanes': [6,],
                          'rec_crop': [(0,6)]
                      },
                      '20200701': {
                          'nfish': 3,
                          'nplanes': [6, 6, 6],
                          'rec_crop': [(0,6), (0, 6), (1, 6)]
                      }
                   }

        :param frate: float, framerate (not volumerate) of the microscope. Default is 30Hz.
        :param ds_fac: int, downsampling factor. Downsampling is done by averaging ds_fac frames. Default is 5.
        :param tau: float, Ca indicator decay constant
        :param diameter: int, expected pixel diameter of neuron ROI
        :param kwargs:
        """
        self.path = kwargs.get('path', r'J:/Johannnes Kappel/Imaging data/Theia')
        self.date = kwargs.get('date', None)
        self.dpath = os.path.join(self.path, self.date)
        self.classfile = kwargs.get('classfile', r'J:/_Projects/Enrico_Katja/EK_classifier16012020.npy')
        self.recwise = kwargs.get('recwise', False)
        self.delete = kwargs.get('delete', False)
        self.user = kwargs.get('user', 'jj')
        self.filter_sigma = kwargs.get('filter_sigma', 4)
        self.recompute = kwargs.get('recompute', True)
        self.do_preprocessing = kwargs.get('preprocess', True)

        if not os.path.exists(self.path):
            sys.exit('Imaging path {0} does not exist.'.format(self.path))

        self.tau = kwargs.get('tau', 7)
        self.diameter = kwargs.get('diameter', 4)
        self.ds_fac = kwargs.get('ds_fac', 5)
        self.nplanes = kwargs.get('nplanes', 6)
        self.frate = kwargs.get('frate', 30)
        self.rec_crop = kwargs.get('rec_crop', [0, 1])
        self.exf = kwargs.get('exf', [])

        self.fish = None
        self.dsvolrate = None
        self.fpath = None
        self.vidpath = None
        self.plane = None

        self.traces = None
        self.classify = kwargs.get('classify', True)
        self.iscell = None
        self.stat = None
        self.frames_per_file = None
        self.offsets = list()

    def init_temp_folders(self):

        if self.dpath == None:
            sys.exit('Path to datefolder is not know.')
        # Suite2p works with tiffs per folder, so make some temporary folders to put the files we will process.
        self.temp_path_1 = os.path.join(self.dpath, 'suite_temp_1')
        self.temp_path_2 = os.path.join(self.dpath, 'suite_temp_2')
        self.temp_path_3 = os.path.join(self.dpath, 'suite_temp_3')
        try:
            os.mkdir(self.temp_path_1)
        except OSError as error:
            # Directory already exists, nothing to do
            pass
        try:
            os.mkdir(self.temp_path_2)
        except OSError as error:
            # Directory already exists, nothing to do
            pass
        try:
            os.mkdir(self.temp_path_3)
        except OSError as error:
            # Directory already exists, nothing to do
            pass
        return

    def set_current_fish(self, fish, rec=None):

        self.fish = fish
        if rec is None:
            self.fpath = os.path.join(self.dpath, 'fish{0}'.format(fish))
        else:
            self.fpath = os.path.join(self.dpath, 'fish{0}'.format(fish), 'rec{0}'.format(rec))

        self.volrate = self.frate/self.nplanes
        self.dsvolrate = self.volrate / self.ds_fac
        return

    # Call suite2p to transform the tiffs to binary.
    def make_bin(self):
        ops = default_ops()

        db = {
            'look_one_level_down': False,
            'data_path': [self.temp_path_1],
            'roidetect': False,
            'do_registration': False,
        }
        run_s2p(ops=ops, db=db)
        return

    # This function puts a gaussian blur over the frames to ease motion correction.
    def filter_plane(self, sigma=4):
        merged_files = glob.glob(self.temp_path_1 + '/merge_plane{0}*raw.tif'.format(self.plane))

        for file in merged_files:
            print('loading ', file)
            ts = tiff.imread(file)
            print('opening zeros')
            tsg = np.zeros(ts.shape)
            print('filtering', file)
            # ts is gaussian-filtered in xy (sigma=4), not in z
            tsg = gaussian(ts, sigma=(0, sigma, sigma), mode='wrap', preserve_range=True)
            filename = file.split('\\')[-1]
            save_path = os.path.join(self.temp_path_2, filename.split('.')[0] + '_filt.tif')
            print('saving ', save_path)
            skimage.io.imsave(save_path, tsg.astype('int16'), plugin='tifffile')

        print('Finished filtering planes')
        return

    # This function does the motion correction, we use a gaussian filtered version for this because the ripple noise
    # otherwise gets too strong of an effect and determines the shifts.
    def do_register_on_filter(self):
        ops = default_ops()
        #TODO check if tau and diameter are actually needed here.
        db = {
            'look_one_level_down': False,
            'data_path': [self.temp_path_2],
            'tau': self.tau,
            'fs': self.volrate,
            'diameter': self.diameter,
            'reg_tif': True,
            'roidetect': False,
            'do_registration': True,
            'nonrigid': True,
            'keep_movie_raw': True,
            'do_regmetrics': True
        }

        run_s2p(ops=ops, db=db)
        return

    # This function applies the shifts found previously to the raw image.
    def apply_shift_to_raw(self):
        ops_path = os.path.join(self.temp_path_2, 'suite2p', 'plane0', 'ops.npy')

        filt_ops = np.load(ops_path, allow_pickle=True).item()

        offsets = []  # registration.utils.init_offsets(filt_ops)
        [offsets.append(filt_ops[i]) for i in ['yoff', 'xoff', 'corrXY']]

        if filt_ops['nonrigid']:

            [offsets.append(filt_ops[i]) for i in ['yoff1', 'xoff1', 'corrXY1']]

        filt_ops['reg_tif_chan2'] = True
        print(filt_ops.keys())
        binfile_raw = os.path.join(self.temp_path_1, 'suite2p', 'plane0', 'data.bin')
        binfile_reg = os.path.join(self.temp_path_1, 'suite2p', 'plane0', 'reg.bin')
        Utils.apply_shifts_to_binary(filt_ops, offsets, binfile_reg, binfile_raw)

        return

    # Downsampling the motion corrected file and saving as both tiff and mmap.
    def downsample(self):

        dpath = os.path.join(self.temp_path_2, 'suite2p', 'plane0', 'reg_tif_chan2')
        mcfiles = [os.path.join(dpath, i) for i in os.listdir(dpath) if i.startswith('file')]

        for idx, mcfile in enumerate(mcfiles):
            print('doing file: ', idx)
            with tiff.TiffFile(mcfile, is_scanimage=False) as ts_file:

                ts = np.array([i.asarray() for i in ts_file.pages])
            ts_ds = np.array([np.mean(ts[i:i + self.ds_fac], axis=0) for i in np.arange(0, ts.shape[0], self.ds_fac)])
            del ts


            if idx == 0:
                ts_mc = ts_ds
            else:
                ts_mc = np.concatenate((ts_mc, ts_ds), axis=0)
            del ts_ds


        skimage.io.imsave(
            os.path.join(self.temp_path_3, 'downsampled_{0:.2}Hz_registered_plane{1}.tif'.format(self.dsvolrate, self.plane)),
            ts_mc.astype('int16'), plugin='tifffile')

        del ts_mc

        return

    # This function calls suite2p to do ROI and trace extraction.
    def do_ROI_detect(self):

        ops = default_ops()
        data_paths = [self.temp_path_3]
        db = {
            'look_one_level_down': False,
            'data_path': data_paths,
            'tau': self.tau,
            'fs': self.dsvolrate,
            'diameter': self.diameter,
            'reg_tif': False,
            'roidetect': True,
            'do_registration': False
        }

        run_s2p(ops=ops, db=db)
        return

    #
    def read_frate_singleplane(self, planepath):

        txtfile = [i for i in os.listdir(planepath) if i.endswith('metadata.txt')][0]
        txt = open(os.path.join(planepath, txtfile), 'rb')
        a = [str(i) for i in txt.readlines() if 'D3Step' in str(i)][0]
        print(a, planepath)
        x = re.findall("\d+\.\d+", a)
        frate = 1000. / float(x[0])
        print('Detected frame rate:', frate)
        self.frate = frate

    # runs extraction for each fish for each plane assuming only one plane was imaged simultaneously
    def run_extraction_singleplane(self, diameter=15):

        if self.rec_crop[1] == 0:
            return

        # Check if the final files already exist, if so skip this plane.
        #self.set_current_fish(self.date, self.fish, rec=None)
        planepath = os.path.join(self.fpath, 'plane{0}'.format(self.plane))
        print(planepath, 'PLANEPATH')
        if not self.recompute:
            if os.path.exists(os.path.join(planepath, 'stat.npy')):

                print('{0} fish {1} plane {2} was already processed. Skipping.'.format(self.date, self.fish, self.plane))
                return

        self.read_frate_singleplane(planepath)
        self.do_register_roi_class_singleplane(planepath, diameter=diameter)
        return

    def do_register_roi_class_singleplane(self, datapath, diameter=15):

        ops = default_ops()
        db = {
            'two_step_registration': True,
            'look_one_level_down': False,
            'data_path': [datapath],
            'tau': 7,
            'fs': self.frate,
            'diameter': diameter,
            'reg_tif': True,
            'roidetect': True,
            'do_registration': 2,
            'nonrigid': True,
            'keep_movie_raw': True,
            'do_regmetrics': True,
            'block_size': [6, 6],
            'maxregshiftNR': 10,
            'smooth_sigma': 1.5,
            'snr_thresh': 1.1,
            'smooth_sigma_time': 1
        }
        ops = run_s2p(ops=ops, db=db)
        self.stat = np.load(os.path.join(datapath, 'suite2p/plane0/stat.npy'), allow_pickle=True)
        if self.classify:
        
            #lab classifier
            cl = classification.Classifier(self.classfile)
            cl.keys = ['npix_norm', 'compact', 'skew']
            iscell_lab = cl.run(self.stat)
            np.save(os.path.join(datapath, 'suite2p/plane0/iscell_lab.npy'), iscell_lab)

        self.frames_per_file = ops['frames_per_file'][0]
        self.traces = np.load(os.path.join(datapath, r'suite2p\plane0\F.npy'))
        self.vidpath = os.path.join(datapath, r'suite2p\plane0\reg_tif\file000_chan0.tif')

        return

    def run_extraction(self):

        self.dpath = os.path.join(self.path, self.date)

        ff = sorted([int(i.split('\\')[-1].strip('fish')) for i in glob.glob(self.dpath + '/fish*')])
        ff = [i for i in ff if i not in self.exf]
        if not self.recwise:
            f_iter = [(fish, None) for fish in ff]
        else:
            f_iter = [(fish, rec)
                      for fish in ff
                      for rec in range(self.rec_crop[0]+1, self.rec_crop[1]+1)
                      ]
        #TODO add forced overwrite for existing files
        for fish, rec in f_iter:
            print(fish, rec)
            if self.rec_crop[-1] == 0:
                continue

            self.set_current_fish(fish, rec=rec)
            if not self.recwise:
                recs_total = self.nplanes * (self.rec_crop[1] - self.rec_crop[0])
            else:
                recs_total = self.nplanes
            print(self.fpath)

            if self.delete:
                delete_old_data(self.fpath)
            all_merged_files = glob.glob(self.fpath + '/merge_plane*raw.tif')
            print("Found a total of {} processed recording files. Expecting {} processed recording files".format(len(all_merged_files), recs_total))

            if len(all_merged_files) < recs_total and self.do_preprocessing:

                if not self.recwise:
                    print('Starting preprocessing...')
                    self.run_preprocessing()
                else:
                    self.run_preprocessing_alt(self.fpath, self.date, fish)
            all_merged_files = glob.glob(self.fpath + '/merge_plane*raw.tif')
            all_merged_files.sort(key=lambda x:int(x.split('_')[-2].strip('rec')))
            # if len(all_merged_files) < recs_total:
            #     print(self.fpath, all_merged_files)
            #     sys.exit('Not enough data is available, stopping execution.')

            for plane in range(0, self.nplanes):
                self.plane = plane
                # Check if the final files already exist, if so skip this plane.
                if os.path.exists(os.path.join(self.fpath, 'plane{0}'.format(self.plane), 'stat.npy')):
                    print('{0} fish {1} plane {2} was already processed. Skipping.'.format(self.date, self.fish, self.plane))
                    continue

                self.init_temp_folders()
                if len(glob.glob(self.temp_path_1 + '/merge_plane{0}*raw.tif'.format(self.plane))) < 1:

                    # Moving merged files to the first temporary folder
                    merged_files = glob.glob(self.fpath + '/merge_plane{0}*raw.tif'.format(self.plane))
                    merged_files.sort(key=lambda x:int(x.split('_')[-2].strip('rec')))
                    if len(glob.glob(self.temp_path_1 + '/merge_plane{0}*raw.tif'.format(self.plane))) < (self.rec_crop[1] - self.rec_crop[0]):

                        # If the merged files do not exist yet, make them. Somewhat redundant now, see above
                        if self.recwise:

                            recs_plane_total = 1

                        else:

                            recs_plane_total = self.rec_crop[1] - self.rec_crop[0]

                        if len(merged_files) < (recs_plane_total):

                            sys.exit('Not enough data for plane {0} is available, stopping execution.'.format(plane))

                        for file in merged_files:
                            shutil.move(file, self.temp_path_1)

                # Transform tifffiles to binary files.
                if not os.path.exists(os.path.join(self.temp_path_1, 'suite2p', 'plane0', 'data.bin')):
                    print('doing make binary')
                    self.make_bin()

                # Moving filtered files to the temporary folder
                if len(glob.glob(self.temp_path_2 + '/merge_plane{0}*raw_filt.tif'.format(self.plane))) < 1:
                    filtered_files = glob.glob(self.fpath + '/merge_plane{0}*raw_filt.tif'.format(self.plane))

                    # If filtered files do not exist yet, make them.
                    if len(filtered_files) < 1:
                        print('Gaussian filtering files')
                        self.filter_plane(sigma=self.filter_sigma)
                        filtered_files = glob.glob(self.temp_path_2 + '/merge_plane{0}*raw_filt.tif'.format(self.plane))

                        if len(filtered_files) < 1:
                            print('No data for plane {0} is available, stopping execution.'.format(self.plane))
                            sys.exit()
                    else:
                        print('Moving filtered files to {0}'.format(self.temp_path_2))
                        for file in filtered_files:
                            shutil.move(file, self.temp_path_2)

                # Doing the motion correction on the filtered planes.
                if not os.path.exists(os.path.join(self.temp_path_2, 'suite2p', 'plane0', 'ops.npy')):
                    print('doing register on filter')
                    self.do_register_on_filter()

                # Applying the found shifts to the raw data.
                if not os.path.exists(os.path.join(self.temp_path_2, 'suite2p', 'plane0', 'reg_tif_chan2')):
                    print('doing apply shifts to raw')
                    self.apply_shift_to_raw()

                # Downsampling the data for ROI and signal extraction.
                if len(glob.glob(os.path.join(
                        str(self.temp_path_3), 'downsampled_{0:.2}Hz_registered*_plane{1}.tif'.format(
                            str(self.dsvolrate), str(self.plane))))) == 0:
                    print('downsampling')
                    self.downsample()

                # Extracting ROIs and traces.
                if not os.path.exists(os.path.join(self.temp_path_3, 'suite2p', 'plane0', 'stat.npy')):
                    print('ROI detect')
                self.do_ROI_detect()

                # Do cell classification with lab classifier.
                if not os.path.exists(os.path.join(self.temp_path_3, 'suite2p', 'plane0', 'iscell_lab.npy')):
                    print('Using lab classifier.')
                    classfile = 'C:/Users/kslangewal/Anaconda3/envs/suite2p/Lib/site-packages/suite2p/classifiers/EK_classifier16012020.npy'
                    self.stat = np.load(os.path.join(self.temp_path_3, 'suite2p/plane0/stat.npy'), allow_pickle=True)
                    iscell = classification.Classifier(self.classfile, keys=['npix_norm', 'compact', 'skew']).run(self.stat)
                    np.save(os.path.join(os.path.join(self.temp_path_3, 'suite2p/plane0'), 'iscell_lab.npy'), iscell)

                # Renaming the folder to the appropriate plane.
                os.rename(os.path.join(self.temp_path_3, 'suite2p', 'plane0'),
                          os.path.join(self.temp_path_3, 'suite2p', 'plane{0}'.format(self.plane)))

                files_to_move = glob.glob(self.temp_path_1 + '/merge_plane{0}*raw.tif'.format(self.plane)) + \
                                glob.glob(self.temp_path_2 + '/merge_plane{0}*raw_filt.tif'.format(self.plane)) + \
                                [os.path.join(self.temp_path_3, 'suite2p', 'plane{0}'.format(self.plane))] + \
                                [os.path.join(self.temp_path_3,
                                              'downsampled_{0:.2}Hz_registered_plane{1}.tif'.format(self.dsvolrate, self.plane))]

                # Moving the useful files back to the original folder and removing the temporary folders.
                print(files_to_move)
                for file in files_to_move:
                    if os.path.exists(file):
                        try:
                            shutil.move(file, self.fpath)
                        except shutil.Error as err:
                            if os.path.isfile(file):
                                os.remove(os.path.join(self.fpath, file.split('\\')[-1]))
                            elif os.path.isdir(file):
                                shutil.rmtree(os.path.join(self.fpath, file.split('\\')[-1]))
                            shutil.move(file, self.fpath)
                try:
                    shutil.rmtree(self.temp_path_1)
                    self.temp_path_1 = None
                except shutil.Error as e:
                    print("Error: %s : %s" % (self.temp_path_1, e.strerror))
                try:
                    shutil.rmtree(self.temp_path_2)
                    self.temp_path_2 = None
                except shutil.Error as e:
                    print("Error: %s : %s" % (self.temp_path_2, e.strerror))
                try:
                    shutil.rmtree(self.temp_path_3)
                    self.temp_path_3 = None
                except shutil.Error as e:
                    print("Error: %s : %s" % (self.temp_path_3, e.strerror))

        return

    def use_lab_classifier(self):
        for date in self.dates:
            self.dpath = os.path.join(self.path, date)

            #TODO add forced overwrite for existing files
            if not self.recwise:
                f_iter = [(fish, None) for fish in range(self.nfish)]
            else:
                f_iter = [(fish, rec)
                          for fish in range(self.nfish)
                          for rec in range(self.rec_crop[0]+1, self.rec_crop[1]+1)
                          ]
            #TODO add forced overwrite for existing files
            for fish, rec in f_iter:
                self.set_current_fish(date, fish+1, rec=rec)
                if self.rec_crop[1] == 0:
                    continue
                for plane in range(0, self.nplanes):
                    print('Using lab classifier: plane {0}'.format(plane))
                    self.stat = np.load(os.path.join(self.fpath, 'plane{0}/stat.npy'.format(plane)), allow_pickle=True)
                    #TODO give this as argument, might be elsewhere
                    classfile = 'C:/Users/jkappel/EK_classifier16012020.npy'
                    cl = classification.Classifier(classfile)
                    cl.keys = ['npix_norm', 'compact', 'skew']
                    iscell = cl.run(self.stat)
                    np.save(os.path.join(os.path.join(self.fpath, 'plane{0}'.format(plane)), 'iscell_lab.npy'), iscell)
        return

    def run_preprocessing(self):

        """"
        de-interleaves recording files according to imaging planes, corrects for bidirectional scan offsets, saves files
        :param dates: b
        :param path: String of file directory
        :param self.nplanes: Integer # of planes per recording
        :return:

        TODO:
         generalize function to automatically read # of imaging planes (self.nplanes) kwarg from metadata
         call motion correction & source extraction (cnmf) on time series from within function
        """

        # find all raw recording files in the folder,
        # format YYYYMMDD_F(fishnumber)_(recording number), eg 20190405_F1_00001.tif (fish #1, recording #1)
        if self.user=='martin':
            recs = set([int(i.split('_')[-2]) for i in os.listdir(self.fpath)])
        elif self.user=='lisa':
            recs = set([int(i.split('_')[-1].strip('.tiff')) for i in os.listdir(self.fpath) if
                        i.endswith('.tif') and '{0}_F{1}'.format(self.date.strip('suite'), str(self.fish)) in i and 'ZStack' not in i])
        else:
            recs = set([int(i.split('_')[-2]) for i in os.listdir(self.fpath) if i.endswith('.tif') and '{0}_F{1}'.format(self.date, str(self.fish)) in i and 'ZStack' not in i])

        # find all processed recordings in the folder
        recs_done = set([int(i.split('_')[-2].strip('rec')) for i in os.listdir(self.fpath) if i.endswith('raw.tif') and 'rec' in i])
        # generate list with recordings left to process
        recs_todo = sorted(list(recs - recs_done))

        print(self.fish, recs)
        print('recs done:', recs_done)
        print('recs TODO:', recs_todo)

        if len(recs_todo) == 0:
            return

        fn = self.fpath.replace('\\', '/')
        for rec in recs_todo:

            plane_ts = [list() for i in range(self.nplanes)]
            print('{0}_F{1}_{2}'.format(self.date, self.fish, str(rec).zfill(5)))
            # find all *.tif files of the recording. Note that fish number (fno) is 0-indexed in the code and
            # 1-indexed in the files because the first animal starts at 1 and not a zero ;)
            recfiles = sorted(
                [os.path.join(self.fpath, i) for i in os.listdir(self.fpath) if i.startswith('{0}_F{1}_{2}'.format(
                    self.date.strip('suite'), self.fish, str(rec).zfill(5)
                ))])
            if self.user=='martin':

                recfiles = glob.glob(self.fpath+'/*_{}_*.tif'.format(str(rec).zfill(5)))
            print('Found recording files: ', recfiles)
            # Each recording consists of several recording files, since the file size would be too massive otherwise
            # Several recordings make one session (normally 10 min per recording, and roughly 1h per session).
            # In case the recfile doesn't fit a round number of volumes (but for instance 937,5), roll makes
            # sure that the next recfile is started at the right index.
            plane_arr = np.arange(self.nplanes)
            for recfile in recfiles:
                print(recfile)
                with tiff.TiffFile(recfile, is_scanimage=False) as ts_file:

                    ts = np.array([i.asarray() for i in ts_file.pages])

                roll = int(ts.shape[0] - (self.nplanes * np.floor(ts.shape[0] / self.nplanes)))
                print(roll)
                # The interleaved time series is separated into planes and stored in a big list of lists
                # the following for-loop could maybe be written more concisely
                for inx in range(self.nplanes):
                    ts_raw = ts[inx::int(self.nplanes), :, :]
                    # The recording chunks are offset-corrected, an artifact from the bidirectional 2P scan
                    bidi_offset_correction_plane(ts_raw)
                    plane_ts[plane_arr[inx]].append(ts_raw.astype('int16'))
                plane_arr = np.roll(plane_arr, -roll)
                del ts

            # The individual chunks are concatenated into one recording file per plane and stored in a tiff
            for pno, ts_plane in enumerate(plane_ts):

                ts_full = np.concatenate(ts_plane, axis=0).astype('int16')
                ts_av = np.average(ts_full, axis=0).astype('int16')
                # Average frame is saved
                skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_av.tif'.format(
                    pno, rec)), ts_av.astype('int16'), plugin='tifffile')
                # This try/except expression is only present because of occasional network problems
                try:
                    skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_raw.tif'.format(
                        pno, rec)), ts_full.astype('int16'), plugin='tifffile')
                except OSError:
                    skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_raw.tif'.format(
                        pno, rec)), ts_full.astype('int16'), plugin='tifffile')

            del plane_ts
        return

    def run_preprocessing_alt(self):

        """"
        de-interleaves recording files according to imaging planes, corrects for bidirectional scan offsets, saves files
        :param dates: b
        :param path: String of file directory
        :param self.nplanes: Integer # of planes per recording
        :return:

        TODO:
         generalize function to automatically read # of imaging planes (self.nplanes) kwarg from metadata
         call motion correction & source extraction (cnmf) on time series from within function
        """

        # find all raw recording files in the folder,
        # format YYYYMMDD_F(fishnumber)_(recording number), eg 20190405_F1_00001.tif (fish #1, recording #1)
        recs = set([int(i.split('_')[-2]) for i in os.listdir(self.fpath) if i.endswith('.tif') and '{0}_F{1}'.format(self.date, self.fish) in i and 'ZStack' not in i])
        # find all processed recordings in the folder
        recs_done = set([int(i.split('_')[-2][-1]) for i in os.listdir(self.fpath) if i.endswith('raw.tif') and 'rec' in i])
        # generate list with recordings left to process
        recs_todo = sorted(list(recs - recs_done))

        print('recs done:', recs_done)
        print('recs TODO:', recs_todo)

        if len(recs_todo) == 0:

            return

        for rec in recs_todo:

            plane_ts = [list() for i in range(self.nplanes)]
            print('{0}_F{1}_{2}'.format(self.date, self.fish, str(rec).zfill(5)))
            # find all *.tif files of the recording. Note that fish number (fno) is 0-indexed in the code and
            # 1-indexed in the files because the first animal starts at 1 and not a zero ;)
            recfiles = sorted([os.path.join(self.fpath, i) for i in os.listdir(self.fpath) if i.startswith('{0}_F{1}_{2}'.format(
                self.date, self.fish, str(rec).zfill(5)
            ))])
            print('Found recording files: ', recfiles)
            # Each recording consists of several recording files, since the file size would be too massive otherwise
            # Several recordings make one session (normally 10 min per recording, and roughly 1h per session).
            # In case the recfile doesn't fit a round number of volumes (but for instance 937,5), roll makes
            # sure that the next recfile is started at the right index.
            plane_arr = np.arange(self.nplanes)
            for recfile in recfiles:
                print(recfile)
                with tiff.TiffFile(recfile, is_scanimage=False) as ts_file:

                    ts = np.array([i.asarray() for i in ts_file.pages])

                roll = int(ts.shape[0] - (self.nplanes * np.floor(ts.shape[0] / self.nplanes)))
                print(roll)
                # The interleaved time series is separated into planes and stored in a big list of lists
                # the following for-loop could maybe be written more concisely
                for inx in range(self.nplanes):
                    ts_raw = ts[inx::int(self.nplanes), :, :]
                    # The recording chunks are offset-corrected, an artifact from the bidirectional 2P scan
                    bidi_offset_correction_plane(ts_raw)
                    plane_ts[plane_arr[inx]].append(ts_raw)
                plane_arr = np.roll(plane_arr, -roll)
                del ts

            # The individual chunks are concatenated into one recording file per plane and stored in a tiff
            for pno, ts_plane in enumerate(plane_ts):

                ts_full = np.concatenate(ts_plane, axis=0)
                ts_av = np.average(ts_full, axis=0)
                # Average frame is saved
                skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_av.tif'.format(
                    pno, rec)), ts_av.astype('int16'), plugin='tifffile')
                # This try/except expression is only present because of occasional network problems
                try:
                    skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_raw.tif'.format(
                        pno, rec)), ts_full.astype('int16'), plugin='tifffile')
                except OSError:
                    skimage.io.imsave(os.path.join(self.fpath, 'merge_plane{0}_rec{1}_raw.tif'.format(
                        pno, rec)), ts_full.astype('int16'), plugin='tifffile')

            del plane_ts
        return
        
def delete_old_data(lpath):

    files = glob.glob(lpath + '/*.*')
    memmaps = [i for i in files if i.startswith('memmap')]
    print(memmaps)
    #[os.remove(m) for m in memmaps]
    estimates = [i for i in files if i.startswith('estimates')]
    print(estimates)
    # [os.remove(e) for e in estimates]
    return

if __name__ == '__main__':

    lpath = 'J:/Johannes Kappel/Imaging data/Theia'

    ### Shachar & Inbal data first try ###
    inputdict = {'20200930': {'nfish': 3, 'nplanes': [6] * 3, 'rec_crop': [(0, 0), (0, 6), (0, 0)]}}
    inputdict = {'20201019': {'nfish': 3, 'nplanes': [6] * 3, 'rec_crop': [(1, 6), (0, 6), (0, 6)]}}

    ### Shachar WFM analysis ###
    inputdict = {'20201118': {'nfish': 1, 'nplanes': [6] * 1, 'rec_crop': [(0, 4)]},
                 '20201119': {'nfish': 2, 'nplanes': [6] * 2, 'rec_crop': [(0, 2), (0, 3)]}}

    ### old data for re-analysis ###

    inputdict = {
                '20200402': {'nfish': 2, 'nplanes': [6] * 2, 'rec_crop': [(0, 7), (0, 7)]},
                '20200404': {'nfish': 1, 'nplanes': [6] * 1, 'rec_crop': [(0, 7)]},
                '20200405': {'nfish': 3, 'nplanes': [6] * 3, 'rec_crop': [(0, 7), (0, 8), (0, 8)]},
                '20200409': {'nfish': 2, 'nplanes': [6] * 2, 'rec_crop': [(0, 8), (0, 8)]}
                 }

    ### fine-scale thalamus imaging 7dpf ###

    inputdict = {

        '20201203': {'nfish': 1, 'nplanes': [6] * 1, 'rec_crop': [(0, 2)]},
        '20201203': {'nfish': 2, 'nplanes': [6] * 2, 'rec_crop': [(0, 2), (0, 2)]}
    }

    ### Inbal new data ###
    inputdict = {

        '20210225': {'nfish': 3, 'nplanes': [6] * 3, 'rec_crop': [(0, 2), (0, 2), (0, 2)]},
        '20210226': {'nfish': 3, 'nplanes': [6] * 3, 'rec_crop': [(0, 2), (0, 2), (0, 2)]},

    }
    inputdict = {

        '20210819': {'nfish': 2, 'nplanes': [1] * 2, 'rec_crop': [(0, 1), (0, 1)]},

    }
    ### NTR ablation data ###
    # inputdict = {
    #
    #     '20210224': {'nfish': 4, 'nplanes': [6] * 4, 'rec_crop': [(0, 2)]*4},
    #     '20210303': {'nfish': 7, 'nplanes': [6] * 7, 'rec_crop': [(0, 2)]*7},
    #
    # }
    #extraction = ROITracesExtraction(lpath, inputdict, ds_fac=6, recwise=True, delete=True)
    #extraction.run_extraction()
    #extraction.use_lab_classifier()
    # lpath = 'J:/Johannes Kappel/Ablation data/20211112'
    # extraction = ROITracesExtraction(lpath, {}, ds_fac=1, recwise=False, delete=True)
    # extraction.rec_crop = (0, 1)
    # extraction.plane = 0
    # for fpath in glob.glob(lpath+'/*'):
    #     if '27' in fpath:
    #         continue
    #     print(fpath)
    #     extraction.fpath = fpath
    #     if not os.path.exists(os.path.join(fpath, 'plane0')):
    #         continue
    #     extraction.run_extraction_singleplane(diameter=7)
    inputdict = {

        '20211031': {'nfish': 5, 'nplanes': [1] * 5, 'rec_crop': [(0, 7), (0, 6), (0, 2), (0, 7), (0, 5)]},

    }
    lpath = 'J:\_Projects\Inbal_JJ\Tg_lines_functional_ROIs\20211031_atf5b_7dpf\20211031'
    # for fno in range(inputdict['20211031']['nfish']):
    #
    #     for recpath in glob.glob(lpath+'/fish{}/rec*'.format(fno+1)):
    #
    #         print(recpath)
    #         recpath = os.path.join(recpath, '/plane0/suite2p')
    #         extraction = ROITracesExtraction(recpath, {}, ds_fac=1, recwise=False, delete=True)
    #         extraction.rec_crop = (0, 1)
    #         extraction.plane = 0
    #         extraction.run_extraction_singleplane(diameter=5)
    #
    # inputdict = {
    #
    #     '20211130': {'nfish': 10, 'nplanes': [1] * 10, 'rec_crop': [(0, 1)] * 10},
    #
    # }
    # lpath = 'J:\Johannes Kappel/Imaging data'
    # for fno in range(inputdict['20211130']['nfish']):
    #
    #     extraction = ROITracesExtraction(lpath, inputdict, ds_fac=1, recwise=False, delete=True)
    #     extraction.rec_crop = (0, 1)
    #     extraction.plane = 0
    #     extraction.date = '20211130'
    #     extraction.fish = fno+1
    #     extraction.run_extraction_singleplane(diameter=5)

    # inputdict = {
    #
    #     '20211209': {'nfish': 10, 'nplanes': [6] * 10, 'rec_crop': [(0, 1)] * 10},
    #
    # }
    # lpath = 'J:\Johannes Kappel/Imaging data/Theia'
    #
    # extraction = ROITracesExtraction(lpath, inputdict, ds_fac=5, recwise=False, delete=True)
    # extraction.run_extraction()
    # extraction.use_lab_classifier()
    #

    # inputdict = {
    #
    #     '20211112': {'nfish': 8, 'nplanes': {17:1, 18:1, 19:1, 20:1, 22:1, 24:1, 29:1, 31:1}, 'rec_crop': {17:(0, 1), 18:(0, 1), 19:(0, 1), 20:(0, 1), 22:(0, 1), 24:(0, 1), 29:(0, 1), 31:(0, 1)}}
    #
    # }
    # lpath = 'J:\Johannes Kappel/Ablation data'
    # #for fno in [17, 18, 19, 20, 22, 24, 29, 31]:
    # for fno in [31]:
    #     extraction = ROITracesExtraction(lpath, inputdict, ds_fac=1, recwise=False, delete=True)
    #     extraction.rec_crop = (0, 1)
    #     extraction.plane = 0
    #     extraction.date = '20211112'
    #     extraction.fish = fno
    #     extraction.dpath = os.path.join(extraction.path, extraction.date)
    #     extraction.fpath = os.path.join(extraction.dpath, 'fish{0}'.format(extraction.fish))
    #     extraction.run_extraction_singleplane(diameter=7)

#    extraction = ROITracesExtraction(path='J:/Johannes Kappel/Imaging data/Theia', date='20220203', nfish=8, ds_fac=5, diameter=7, recwise=False, delete=True)
#     extraction = ROITracesExtraction(path=r'J:/Johannes Kappel/Imaging data',
#                                      date='20211208',
#                                      nfish=1,
#                                      ds_fac=2,
#                                      diameter=7,
#                                      recwise=False,
#                                      delete=True,
#                                      nplanes=10,
#                                      rec_crop=[0, 4],
#                                      exf=range(1, 7),
#                                      frate=40,
#                                      user='martin'
#                                      )
    extraction = ROITracesExtraction(path=r'J:/Johannes Kappel/Imaging data/Theia',
                                     date='20230503',
                                     nfish=4,
                                     ds_fac=5,
                                     diameter=7,
                                     recwise=False,
                                     delete=True,
                                     nplanes=6,
                                     rec_crop=[0, 2],
                                     exf=[],
                                     frate=30,
                                     user='jj',
                                     preprocess=True

                                     )
    extraction.run_extraction()