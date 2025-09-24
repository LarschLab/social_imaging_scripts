import os
import sys
sys.path.insert(1, 'C:/Users/jkappel/PycharmProjects/SocialVisionSuite2p')
import Utils
from Preprocessing import bidi_offset_correction_plane
import tifffile as tiff
import numpy as np
from skimage.transform import rotate, resize
from skimage.feature import match_template
import pickle
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity, match_histograms
from matplotlib.colors import LinearSegmentedColormap
import glob
import nrrd
from scipy import ndimage
from scipy.ndimage import gaussian_filter

class ROIRegistration:

    def __init__(self, **kwargs):

        """
        :param lpath: str, path to zstacks
        :param datedict: dict, keys are dates to process in format YYYYMMDD,
                            values are list of tuples which recordings to use for each fish of the date,
                            0-indexed, eg (0, 6) yields recording 1 to 7
        :param scale_ratios: list of iterables containing floats, dates->fish, contains the scaling factor between the
                                recording FOV and ZStack FOV per fish (Recording ZOOM / ZStack ZOOM)
        :param angle: float, angle in degree which recording averages have to be rotated by to match the rotated zstack
                        (ZStacks have been rotated at another point manually to fit orientation of 'standard brain')
                        TODO: automate ZStack rotation according to standard brain orientation
        :param coarse_step: int, step size for finding the correct xyz location of the frame in the zstack
        :return: reg_dict
        """

        self.impath = kwargs.get('impath', None)
        self.alpath = kwargs.get('alpath', None)

        self.csv_path = kwargs.get('csv_path', '')
        self.coarse_step = kwargs.get('coarse_step', 3)
        self.centroid = kwargs.get('centroid', False)
        self.reg_dict = kwargs.get('reg_dict', {})
        self.exdates = kwargs.get('exdates', [])
        self.fkeys =  kwargs.get('fkeys', None)
        self.opt_scaling = kwargs.get('opt_scaling', True)
        self.main_db = kwargs.get('main_db', True)
        self.resize = kwargs.get('resize', True)
        self.rotate_first = kwargs.get('rotate_first', False)
        self.zfilt = kwargs.get('zfilt', False)
        self.resc_z = kwargs.get('resc_z', (75, 95))
        self.resc_im = kwargs.get('resc_im', (0.1, 99.9))
        self.gamma = kwargs.get('gamma', False)
        self.zlim = kwargs.get('zlim', False)
        self.recompute = kwargs.get('recompute', False)
        self.reextract = kwargs.get('reextract', True)
        self.db_stim = kwargs.get('db_stim', [])
        self.mdict = kwargs.get('mdict', {})
        self.recwise = kwargs.get('recwise', False)
        self.reverse_z = kwargs.get('reverse_z', False)
        self.match_hist = kwargs.get('match_hist', False)
        self.use_maxim = kwargs.get('use_maxim', False)

        self.folder = None
        self.nplanes = None
        self.ops = None
        self.date = None
        self.age = None
        self.fno = None
        self.fish = None
        self.fkey = None
        self.fpath = None
        self.f_alpath = None
        self.rec_crop = None
        self.rec_iter = None
        self.xys_df = None
        self.dims = None
        self.zoom = None
        self.zstack_zoom = None
        self.res_scaling = None
        self.zoom_scaling = None

        if self.main_db:

            self.reg_db = pd.read_excel(

                self.csv_path,
                sheet_name='FISHES_OVERVIEW',
                engine='openpyxl',
                dtype={
                    'Date': str,
                    'Rot_Angle': float,
                    'Zoom': str,
                    'Zstack_Zoom': str,
                    'Recs_Checked': str

                }
            )
            return

        else:

            self.reg_db = pd.read_excel(

                self.csv_path,
                sheet_name='FISH_OVERVIEW',
                engine='openpyxl',
                dtype={
                    'date': str,
                    'rot_angle': float,
                    'zoom': float,
                    'zstack_zoom': float,
                    'recs_final': str

                }
            )
            return

    def iter_dbase(self):

        for index, ops in self.reg_db.iterrows():

            if np.isnan(float(ops.date)):

                continue

            self.ops = ops
            print(self.ops)

            self.date = self.ops.date
            if self.date in self.exdates:

                continue

            self.fno = int(self.ops.fno)
            self.fish = 'fish{}'.format(int(self.fno))
            self.fkey = '{}_F{}'.format(self.date, self.fno)
            self.nplanes = int(self.ops.nplanes)
            self.fpath = os.path.join(str(self.impath), self.date, str(self.fish))
            self.f_alpath = os.path.join(
                    self.alpath,
                    self.date,
                    self.fish
            )
            if not os.path.exists(self.f_alpath):
                os.makedirs(self.f_alpath)
            self.rec_crop = tuple([int(self.ops['recs_final'].strip(',')[0]), int(self.ops['recs_final'].split(',')[1])])
            self.recwise = bool(self.ops.recwise)


            self.zoom = float(self.ops.zoom)
            self.zstack_zoom = float(self.ops.zstack_zoom)
            print(self.rec_crop)
            if self.rec_crop[1] == 0:

                continue

            if self.recwise:

                self.rec_iter = [int(rec) for rec in range(self.rec_crop[0], self.rec_crop[1]+1)]

            else:

                self.rec_iter = [None]

            #self.register_imaging_planes()
            self.extract_xys()

    def iter_dbase_main(self):

        for index, ops in self.reg_db.iterrows():

            if np.isnan(float(ops.Date)):

                continue

            if ops.Status == 'Bad':

                continue

            # if ops.Status == 'Process':
            #
            #     continue
            #
            # if ops.Recs_Checked == 'None':
            #
            #     continue

            if ops.Stim not in self.db_stim:

                continue

            if not ops.Gcamp == '6s':

                continue

            self.ops = ops
            print(self.ops)

            self.date = self.ops.Date
            if self.date in self.exdates:

                continue

            self.fno = int(self.ops.FNO)
            self.fish = 'fish{}'.format(int(self.ops.FNO))
            self.age = int(self.ops.Age)
            self.fkey = '{}_F{}'.format(self.date, self.ops.FNO)
            self.nplanes = int(self.ops.Nplanes)
            if self.fkeys is not None:

                if self.fkey not in self.fkeys:
                    print('Continuing...')
                    continue
            self.impath = self.ops.Path
            self.fpath = os.path.join(str(self.impath), self.date, str(self.fish))
            self.f_alpath = os.path.join(
                    self.alpath,
                    self.date,
                    self.fish
            )
            if self.age < 10:
                self.folder = 'one_week'
            elif 9 < self.age < 17:
                self.folder = 'two_weeks'
            else:
                self.folder = 'three_weeks'

            self.rot_angle = float(self.ops.Rot_Angle)

            # self.generate_c2()
            # continue
            if not os.path.exists(self.f_alpath):
                os.makedirs(self.f_alpath)

            self.rec_crop = tuple([int(i) for i in self.ops['Recs_Checked'].strip('()').split(',')])
            if self.ops.Zoom == 'False':
                self.read_zoom()
            else:
                self.zoom = float(self.ops.Zoom)
                self.zstack_zoom = float(self.ops.Zstack_Zoom)


            print(self.rec_crop)
            if self.rec_crop[1] == 0:

                continue

            if self.recwise:

                self.rec_iter = [int(rec) for rec in range(self.rec_crop[0]+1, self.rec_crop[1]+1)]
                print('REC ITER', self.rec_iter)
            else:

                self.rec_iter = [None]

            zfile_nrrd = os.path.join(self.alpath, self.folder,
                                      r'{}_F{}_2P_GCaMP6s_stack.nrrd'.format(self.date, self.fno))
            if not os.path.exists(zfile_nrrd):
                isz = self.generate_nrrd()

                if not isz:
                    continue
            #self.rewrite_nrrd()
            self.register_imaging_planes()
            self.extract_xys()

    def read_zoom(self):

        print('Reading zoom params from metadata...')
        tsp = glob.glob(self.fpath + '/{}_F{}_00001_00001.tif'.format(self.date, self.ops.FNO))[0]
        ts = tiff.TiffFile(tsp)
        self.zoom = float([i for i in ts.pages[0].software.split('\n') if 'Zoom' in i][0].split('=')[1])
        if os.path.exists(os.path.join(self.fpath, 'altzstackchannel')):
            print('Yes')
            tsp = glob.glob(self.fpath + '/altzstackchannel/*stack_00001_00001.tif')[0]
        else:
            tsp = glob.glob(self.fpath + '/*stack*.tif')[0]
        ts = tiff.TiffFile(tsp)
        self.zstack_zoom = float([i for i in ts.pages[0].software.split('\n') if 'Zoom' in i][0].split('=')[1])

        print('Zoom: ', self.zoom, ' Zstack Zoom: ', self.zstack_zoom)

    def rewrite_nrrd(self):

        zfile_nrrd = os.path.join(self.alpath, self.folder, r'{}_F{}_2P_GCaMP6s_stack.nrrd'.format(self.date, self.fno))
        print('Rewriting NRRD ', zfile_nrrd)
        zstack_rot, header = nrrd.read(zfile_nrrd)
        if self.age > 16:

            sd = np.array([
                [1.5, 0, 0],
                [0, 1.5, 0],
                [0, 0, 2]]).astype(float)

        elif self.age < 10:
            # for tectal ablations
            sd = np.array([
                [1.5, 0, 0],
                [0, 1.5, 0],
                [0, 0, 2]]).astype(float)
        else:

            sd = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]).astype(float)

        header = {
            'space dimension': 3,
            'space directions': sd,
            'space units': ["microns", "microns", "microns"],
            'type': 'uint8',
            'PixelType': 'uint8'
        }
        nrrd.write(zfile_nrrd, zstack_rot, header=header)
        return

    def generate_nrrd(self):

        zfiles = glob.glob(os.path.join(self.fpath, 'zstack/*'))
        zfiles = [i for i in zfiles if not 'rotated' in i]
        if len(zfiles) > 1:
            zfiles = [i for i in zfiles if not 'rotated' in i]
            if len(zfiles) > 1:
                print('Found more than one zstack. Using {}'.format(zfiles[0]))

        if len(zfiles) == 0:

            zfiles = glob.glob(os.path.join(self.fpath, '*stack*.*'))
            if len(zfiles) == 0:
                print('Found no zstack. Returning')
                return False
            if len(zfiles) > 1:
                zfiles = [i for i in zfiles if not 'rotated' in i]
                if len(zfiles) > 1:

                    print('Found more than one zstack. Using {}'.format(zfiles[0]))

        zstack = tiff.imread(zfiles[0])
        if len(zstack.shape) == 4:
            print('Reshaping ZSTACK!', zstack.shape)
            zstack = np.reshape(zstack, (zstack.shape[0]*zstack.shape[1], zstack.shape[2], zstack.shape[3]))
        if 'bidicorr' not in zfiles[0]:
            bidi_offset_correction_plane(zstack, maxoff=20, zrange=(int(zstack.shape[0]/2), int(zstack.shape[0]/2)+10))
        zstack = rescale_intensity(zstack,
                                   in_range=(
                                   np.percentile(zstack, 0), np.percentile(zstack, 99.99)),
                                   out_range=(0, 255))
        zstack = zstack.astype(np.uint8)

        if not os.path.exists(self.f_alpath):
            os.makedirs(self.f_alpath)
        zfile_tiff = os.path.join(
            self.f_alpath,
            r'{}_F{}_2P_GCaMP6s_stack_norot.tif'.format(self.date, self.fno))

        tiff.imwrite(zfile_tiff, zstack)
        zstack_rot = ndimage.rotate(zstack, axes=(1, 2), angle=self.rot_angle, reshape=self.resize, cval=0)
        zfile_tiff = os.path.join(
            self.f_alpath,
            r'{}_F{}_2P_GCaMP6s_stack.tif'.format(self.date, self.fno))
        tiff.imwrite(zfile_tiff, zstack_rot)
        if self.age > 16:

            sd = np.array([
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2]]).astype(float)

        elif self.age < 10:
            # for tectal ablations
            sd = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]).astype(float)
        else:

            sd = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]).astype(float)

        header = {
            'space dimension': 3,
            'space directions': sd,
            'space units': ["microns", "microns", "microns"],
            'type': 'uint8',
            'PixelType': 'uint8'
        }
        zstack_rot = np.moveaxis(zstack_rot, 0, -1)
        zstack_rot = np.moveaxis(zstack_rot, 0, 1)

        zfile_nrrd = zfile_tiff.split('.')[0] + '.nrrd'
        nrrd.write(zfile_nrrd, zstack_rot, header=header)

        zfile_nrrd = os.path.join(self.alpath, self.folder, r'{}_F{}_2P_GCaMP6s_stack.nrrd'.format(self.date, self.fno))
        nrrd.write(zfile_nrrd, zstack_rot, header=header)
        return True

    def generate_c2(self):

        cpath = glob.glob(self.fpath + r'/altzstackchannel/*C2*.tif')
        zstack = tiff.imread(cpath[0])
        bidi_offset_correction_plane(zstack, maxoff=20, zrange=(int(zstack.shape[0]/2), int(zstack.shape[0]/2)+10))
        zstack = rescale_intensity(zstack,
                                   in_range=(
                                   np.percentile(zstack, 0), np.percentile(zstack, 99.99)),
                                   out_range=(0, 255))
        zstack = zstack.astype(np.uint8)
        print(self.resize, 'Resize BOOL')
        zstack_rot = ndimage.rotate(zstack, axes=(1, 2), angle=self.rot_angle, reshape=self.resize, cval=0)
        print(zstack.shape, zstack_rot.shape)
        zstack_rot = np.moveaxis(zstack_rot, 0, -1)
        zstack_rot = np.moveaxis(zstack_rot, 0, 1)
        nrrdpath = os.path.join(self.alpath, self.folder,
            r'{}_F{}_2P_GCaMP6s_stack.nrrd'.format(self.date, self.fno))
        nrrd_c1, header = Utils.trans_nrrd(nrrd.read(nrrdpath), header=True)
        zstack_rot = resize(zstack_rot, nrrd_c1.shape[::-1], preserve_range=True)
        nrrd.write(nrrdpath.replace('.nrrd', '_c2.nrrd'), zstack_rot, header=header)
        return

    def register_imaging_planes(self, plot=True):

        """
        Function for creating imaging plane averages across recordings, rotating and downsampling for matching to ZStack
        """

        for rno, rec in enumerate(self.rec_iter):

            print(self.date, self.fish, 'rec ', rec, ' processing')
            if self.rec_crop[1] == 0:
                print('{} will not be processed'.format(self.fpath))
                continue

            if rno == 0:

                self.reg_dict[self.fkey] = {}
            print('RECWISE', self.recwise)
            if self.recwise:

                self.fpath = os.path.join(self.impath, self.date, self.fish, 'rec{0}'.format(rec))
                self.f_alpath = os.path.join(self.alpath, self.date, self.fish, 'rec{0}'.format(rec))
                if not os.path.exists(self.f_alpath):
                    os.makedirs(self.f_alpath)

                self.reg_dict[self.fkey][rec] = {}
            if os.path.exists(os.path.join(self.alpath, self.date, self.fish, 'reg_shifts_{}_{}.p'.format(self.date, self.fish))):

                if not self.recompute:

                    self.reg_dict = pickle.load(
                        open(os.path.join(self.alpath, self.date, self.fish, 'reg_shifts_{}_{}.p'.format(self.date, self.fish)), 'rb'))
                    print('Registration shifts file already exists.')
                    continue

            if self.rotate_first:

                zstack = tiff.imread(
                    os.path.join(
                        self.alpath, self.date, self.fish,
                        r'{}_F{}_2P_GCaMP6s_stack.tif'.format(self.date, self.fno)))
            else:

                zfile_norot = os.path.join(
                        self.alpath, self.date, self.fish,
                        r'{}_F{}_2P_GCaMP6s_stack_norot.tif'.format(
                            self.date, self.fno))

                if not os.path.exists(zfile_norot):
                    self.generate_nrrd()

                zstack = tiff.imread(zfile_norot)

            if self.zfilt:

                zstack = gaussian_filter(zstack, (self.zfilt[0], self.zfilt[1], self.zfilt[2]))

            zstack = rescale_intensity(zstack,
                                       in_range=(np.percentile(zstack, self.resc_z[0]), np.percentile(zstack, self.resc_z[1])),
                                       out_range=(0, 255)
            )
            if self.gamma:

                zstack = zstack ** self.gamma[0]
                zstack = rescale_intensity(zstack,
                                           in_range=(np.percentile(zstack, 0),
                                                     np.percentile(zstack, 100)),
                                           out_range=(0, 255),
                                           )
            if self.fkey in self.mdict.keys():
                if self.recwise:
                    pad = self.mdict[self.fkey][rec]
                else:
                    pad = self.mdict[self.fkey]
            else:
                pad = [False] * self.nplanes

            for plane in range(int(self.nplanes)):

                self.zoom_scaling = float(self.zstack_zoom)/float(self.zoom)
                if self.use_maxim:

                    ds_mean_file = glob.glob(os.path.join(self.fpath, 'downsampled*registered_plane{}_av_max.tif'.format(plane)))
                    ds_almean_file = glob.glob(os.path.join(self.f_alpath, 'downsampled*registered_plane{}_av_max.tif'.format(plane)))

                else:

                    ds_mean_file = glob.glob(
                        os.path.join(self.fpath, 'downsampled*registered_plane{}_av_std.tif'.format(plane)))
                    ds_almean_file = glob.glob(
                        os.path.join(self.f_alpath, 'downsampled*registered_plane{}_av_std.tif'.format(plane)))

                if len(ds_mean_file) > 0:
                    av_im = tiff.imread(ds_mean_file[0])

                elif len(ds_almean_file) > 0:
                    av_im = tiff.imread(ds_almean_file[0])

                else:

                    ds_file = glob.glob(os.path.join(self.fpath, 'downsampled*registered_plane{}.tif'.format(plane)))
                    with tiff.TiffFile(
                            ds_file[0],
                            is_scanimage=False) as ts_file:

                        ts = np.array([i.asarray() for i in ts_file.pages])[:1500]
                    if self.use_maxim:

                        av_im = np.max(ts, axis=0)
                        dsf = ds_file[0].replace('\\', '/').split('/')[-1].split('.tif')[0] + '_av_max.tif'

                    else:

                        av_im = np.std(ts, axis=0)
                        dsf = ds_file[0].replace('\\', '/').split('/')[-1].split('.tif')[0] + '_av_std.tif'

                    av_im = rescale_intensity(av_im,
                                              in_range=(np.percentile(av_im, 0.001),
                                                        np.percentile(av_im, 99.99)),
                                              out_range=(0, 2 ** 8 - 1)
                                              )
                    tiff.imsave(os.path.join(self.f_alpath, dsf), av_im.astype(np.uint8))

                if self.gamma:

                    av_im = av_im ** self.gamma[1]

                if self.rotate_first:

                    av_im_rot = rotate(av_im, angle=self.rot_angle, resize=self.resize, preserve_range=True)

                else:

                    av_im_rot = av_im

                # This is a temorary solution, it would be better to know the exact xy resolution in um/px for stack
                # and imaging plane
                self.res_scaling = zstack.shape[-1]/av_im_rot.shape[-1]
                scaling = self.res_scaling * self.zoom_scaling
                av_im_res = resize_im(av_im_rot, scaling)
                print(scaling, self.zoom_scaling, self.res_scaling, int(round(av_im_rot.shape[0] * scaling)))
                av_im_res = rescale_intensity(av_im_res,
                                           in_range=(np.percentile(av_im_res, self.resc_im[0]),
                                                     np.percentile(av_im_res, self.resc_im[1])),
                                           out_range=(0, 255)
                                           )
                av_im_res = av_im_res.astype(np.uint8)

                if pad[plane]:

                    if isinstance(pad[plane], int):

                        maxmatch = match_template(zstack[pad[plane]],
                                       av_im_res,
                                       pad_input=True,
                                       mode='minimum')
                        xyshifts = np.unravel_index(maxmatch.argmax(), maxmatch.shape)
                        xyshifts = tuple(max(i - d // 2, 0) for i, d in zip(xyshifts, av_im_res.shape))
                        shifts = (pad[plane], xyshifts[0], xyshifts[1])
                    else:
                        shifts = pad[plane]

                else:

                    shifts = find_xyz(
                        zstack,
                        av_im_res,
                        coarse_step=self.coarse_step,
                        rf=.1,
                        match_hist=self.match_hist
                    )
                if self.opt_scaling:

                    scalematches = {}
                    for step in np.arange(shifts[0]-8, shifts[0]+8, 1):

                        if step >= zstack.shape[0]:
                            continue
                        zframe = zstack[step]
                        av_im_rot = rescale_intensity(av_im_rot,
                                                      in_range=(np.percentile(av_im_rot, self.resc_im[0]),
                                                                np.percentile(av_im_rot, self.resc_im[1])),
                                                      out_range=(0, 255)
                                                      )

                        opt_scale, av_im_res, opt_shifts, maxmatch = find_scaling(
                            zframe,
                            av_im_rot,
                            scales=np.arange(scaling-.02, scaling+.02, 0.005),
                            match_hist=self.match_hist
                        )
                        shifts = (step, int(opt_shifts[0]), int(opt_shifts[1]))
                        print('Optimized scaling:', opt_scale, maxmatch, shifts)
                        scalematches[maxmatch] = (opt_scale, av_im_res, shifts)
                    scaling, av_im_res, shifts = scalematches[np.max([i for i in scalematches.keys()])]

                if not isinstance(pad[plane], int):
                    shifts = pad[plane]

                if plot:

                    template = np.empty((zstack.shape[1:]))
                    try:
                        print(shifts)
                        zframe = zstack[shifts[0]]
                        template[shifts[1]:shifts[1] + av_im_res.shape[0], shifts[2]:shifts[2] + av_im_res.shape[1]] = \
                            av_im_res[:template.shape[0]-shifts[1], :template.shape[1]-shifts[2]]
                        template[np.where(template == 0.)] = np.nan

                    except ValueError:

                        template = np.empty((zstack.shape[1:]))
                        print('Matching was outside frame?', shifts)

                    cmap1 = LinearSegmentedColormap.from_list('cmap1', ['black', 'green'])
                    cmap2 = LinearSegmentedColormap.from_list('cmap2', ['black', 'magenta'])
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

                    ax.imshow(zframe, cmap='inferno', alpha=1, interpolation='none', clim=(0, 255))
                    ax.imshow(template, cmap='viridis', alpha=.5, interpolation='none', clim=(0, 255))

                    ax.set_axis_off()

                    plt.savefig(os.path.join(self.f_alpath, '{}_alignment_plane{}_std.png'.format(self.fkey, plane+self.nplanes*rno)), bbox_inches='tight')
                    plt.savefig(os.path.join(self.alpath, self.folder, '{}_alignment_plane{}_std.png'.format(self.fkey, plane+self.nplanes*rno)), bbox_inches='tight')

                    plt.close()

                if bool(self.recwise):

                    self.reg_dict[self.fkey][rec][plane] = (shifts, scaling)

                else:

                    self.reg_dict[self.fkey][plane] = (shifts, scaling)

            if rec is None or rec == self.rec_crop[1]:

                pickle.dump(self.reg_dict,
                            open(os.path.join(self.alpath, self.date, self.fish, 'reg_shifts_{}_{}.p'.format(self.date, self.fish)), 'wb'))

                pickle.dump(self.reg_dict,
                            open(os.path.join(self.alpath, self.folder, 'reg_shifts_{}_{}.p'.format(self.date, self.fish)), 'wb'))
        return

    def extract_xys(

            self,
            plot=True,
            step=2

    ):
        """
        Loading xy centers from spatial weights (ROIs), rotating, scaling and shifting to fit onto respective zstack
        :param lpath: str, file directory
        :param alpath: str, file directory for output shifted xyz coordinates
        :param datedict:  dict, keys are dates to process in format YYYYMMDD,
                            values are list of tuples which recordings to use for each fish of the date,
                            0-indexed, eg (0, 6) yields recording 1 to 7
        :param scale_ratios: list of iterables containing floats, dates->fish, contains the scaling factor between the
                                recording FOV and ZStack FOV per fish (Recording ZOOM / ZStack ZOOM)
        :param dims: tuple of ints, FOV dimensions
        :param depth_factor: float, scaling factor in z for zstacks, eg sf=2 and z-position=frame#100 is equal to 200 microns
                # TODO: should be adjusted individually for each zstack
        :param nplanes: int, # of imaging planes
        :param plot: bool, whether to plot the aligned imaging frames
        :param iteration: str, '1st' or '2nd'. Specifies the caiman cnmf iteration that is used for getting ROI center xys
        :param micron_scale: float, scale that was used in the moving .nrrd file
        :return:
        """

        nids = []
        pnos = []
        all_reg_xyz = []

        regdictpath = os.path.join(self.alpath, self.date, self.fish, 'reg_shifts_{}_{}.p'.format(self.date, self.fish))
        roipath = os.path.join(self.alpath, self.date, self.fish, r'{}_F{}_shifted_xyzs_df.csv'.format(self.date, self.fno))

        if os.path.exists(roipath):

            if not self.recompute and not self.reextract:

                print('ROIs were already extracted')
                return

            else:

                self.reg_dict = pickle.load(open(regdictpath, 'rb'))

        zfile = os.path.join(
            self.alpath, self.folder,
            '{}_F{}_2P_GCaMP6s_stack.nrrd'.
                format(
                self.date,
                self.fno
            )
        )
        print(zfile)
        zstack, header = Utils.trans_nrrd(nrrd.read(zfile), header=True)
        micron_scaling = [header['space directions'][i].max() for i in range(3)]
        zfile_norot = os.path.join(
            self.alpath, self.date, self.fish,
            '{}_F{}_2P_GCaMP6s_stack_norot.tif'.
                format(
                self.date,
                self.fno
            )
        )
        zstack_norot = tiff.imread(zfile_norot)

        for rno, rec in enumerate(self.rec_iter):

            if self.recwise:

                self.fpath = os.path.join(self.impath, self.date, self.fish, 'rec{0}'.format(rec))
                self.f_alpath = os.path.join(self.alpath, self.date, self.fish, 'rec{0}'.format(rec))

            for plane in range(self.nplanes):

                if self.recwise:
                    print(self.reg_dict)
                    print(self.fkey, rec, plane)
                    self.fpath = os.path.join(self.impath, self.date, self.fish, 'rec{0}'.format(rec))
                    shifts, scaling = self.reg_dict[self.fkey][rec][plane]
                else:
                    shifts, scaling = self.reg_dict[self.fkey][plane]
                stats = np.load(os.path.join(self.fpath, 'plane{0}/stat.npy'.format(plane)), allow_pickle=True)
                if self.centroid:

                    xys = np.zeros((len(stats), 2))
                    for neuron in range(len(stats)):
                        xys[neuron, 0] = stats[neuron]['med'][1]
                        xys[neuron, 1] = stats[neuron]['med'][0]
                    xys_df = pd.DataFrame(xys)

                else:

                    xcoords = []
                    ycoords = []
                    labels = []

                    for nno, neuron in enumerate(range(len(stats))):
                        xcoords.extend(list(stats[neuron]['xpix']))
                        ycoords.extend(list(stats[neuron]['ypix']))
                        labels.extend([nno] * len(list(stats[neuron]['ypix'])))

                    xys_df = pd.DataFrame(np.array([labels, xcoords, ycoords]).T, columns=['labels', 'x', 'y'])

                # The xys are saved as a df, but generate_roi_xys returns the matrix (without first column containing
                # indexes). To not mess up other callings of generate_roi_xys, this is fixed here.
                if xys_df.shape[1] > 2:

                    xys = xys_df.values[:, -2:]
                shifts = (shifts[0], shifts[1], shifts[2])
                print('scaling', scaling)

                ds_mean_file = glob.glob(
                    os.path.join(self.f_alpath, 'downsampled*registered_plane{}_av_std.tif'.format(plane)))

                av_im = tiff.imread(ds_mean_file[0])
                if self.rotate_first:

                    rotated_xys, scaled_xys, shifted_xyz = rotate_scale_shift(
                        xys,
                        shifts,
                        scaling=scaling,
                        angle=-int(self.rot_angle),
                        dims=zstack_norot.shape[1:],
                        rotate_first=True
                    )
                    reg_xyz = shifted_xyz

                else:

                    scaled_xys, shifted_xys, rotated_xyz, (ox, oy) = scale_shift_rotate(
                        xys,
                        shifts,
                        scaling=scaling,
                        angle=-int(self.rot_angle),
                        dims=zstack_norot.shape[1:],
                        rotate_first=False
                    )

                    reg_xyz = rotated_xyz

                    reg_xyz[:, 0] += (zstack.shape[1]-zstack_norot.shape[1])/2
                    reg_xyz[:, 1] += (zstack.shape[2]-zstack_norot.shape[2])/2
                    if self.reverse_z:
                        reg_xyz[:, 2] = zstack.shape[0] - reg_xyz[:, 2]
                if plot:

                    bg_s = zstack[int(shifts[0])]
                    av_im = resize_im(av_im, scaling)
                    template = np.empty(zstack_norot.shape[1:])
                    try:
                        template[shifts[1]:shifts[1] + av_im.shape[0], shifts[2]:shifts[2] + av_im.shape[1]] = match_histograms(av_im, bg_s)
                    except ValueError:

                        print('Matching was outside frame?')
                        template = np.empty(zstack_norot.shape[1:])

                    av_im = template
                    av_im = rotate(av_im, int(self.rot_angle), resize=self.resize, preserve_range=True)

                    # av_im = rescale_intensity(av_im,
                    #                           in_range=(np.percentile(av_im, 1), np.percentile(av_im, 99)))
                    av_im[np.where(av_im == 0.)] = np.nan

                    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                    bg_s = bg_s.max()-bg_s
                    ax.imshow(bg_s, clim=(np.percentile(bg_s, .2), np.percentile(bg_s, 99.8)), cmap='Greys')
                    ax.imshow(av_im, cmap='inferno', alpha=.2, clim=(10, 200)) # Something not right, has to be shifted further
                    ax.scatter(reg_xyz[:, 0][::step], reg_xyz[:, 1][::step], c=xys_df.labels.values[::step], cmap='Dark2', s=.1, alpha=.5, edgecolors='none')
                    #plt.axis('off')
                    plt.savefig(os.path.join(self.f_alpath, '{}_aligned_rois_plane_{}.png'.format(self.fkey, plane+self.nplanes*rno)), bbox_inches='tight')
                    plt.savefig(os.path.join(self.alpath, self.folder, '{}_aligned_rois_plane_{}.png'.format(self.fkey, plane+self.nplanes*rno)), bbox_inches='tight')

                    plt.close()

                all_reg_xyz.append(reg_xyz)
                if not self.centroid:

                    nids.append(xys_df.labels.values)
                    pnos.append([plane + self.nplanes * rno] * reg_xyz.shape[0])
        print('Micron scale: ', micron_scaling)
        print('Saving {}_F{}'.format(self.date, self.fno))
        all_reg_xyz = np.concatenate(all_reg_xyz, axis=0)
        if not self.centroid:

            nids = np.concatenate(nids, axis=0)
            pnos = np.concatenate(pnos, axis=0)

        # Micron scaling has to be applied twice, once before tranformation into micron dimensions, and once
        # after trasformation back into pixel dimensions!
        # It can't be left out in Z just because fixed/moving have the same resolution!
        for i in range(3):

            all_reg_xyz[:, i] *= micron_scaling[i]

        header = "x, y, z"
        np.savetxt(os.path.join(self.alpath, self.folder, '{}_F{}_shifted_xyzs.csv'.format(self.date, self.fno)), all_reg_xyz, delimiter=',', header=header)

        xyz_df = pd.DataFrame(all_reg_xyz, columns=('x', 'y', 'z'))
        xyz_df.to_csv(os.path.join(self.alpath, self.date, self.fish, r'{}_F{}_shifted_xyzs_df.csv'.format(self.date, self.fno)), encoding='utf-8')
        if not self.centroid:

            np.save(os.path.join(self.alpath, self.date, self.fish, r'{}_nids.npy'.format(self.fkey)), nids)
            np.save(os.path.join(self.alpath, self.date, self.fish, r'{}_plabels.npy'.format(self.fkey)), pnos)

            np.save(os.path.join(self.alpath, self.folder,  r'{}_nids.npy'.format(self.fkey)), nids)
            np.save(os.path.join(self.alpath, self.folder,  r'{}_plabels.npy'.format(self.fkey)), pnos)

        return


def find_xyz(

        ref,
        moving_image,
        coarse_step=6,
        rf=.25,
        match_hist=False

):
    """
    Find the matching depth of a 2D image in a 3D stack, using match_template of skimage

    :param ref: array-like, must be a stack with the slicing direction along axis=0,
                and with the same pixel size as moving_image
    :param moving_image: array-like, 2D image to register to ref
    :param coarse_step: int, step-size for z-direction when finding the correct z-plane
    :return:
    """

    mi = resize_im(moving_image, rf)
    if match_hist:

        matches_c = np.stack(tuple(match_template(resize_im(plane, rf),
                                                  match_histograms(mi,resize_im(plane, rf)),
                                                  pad_input=True,
                                                  mode='minimum')
                                   for plane in ref[::coarse_step, :, :]), axis=0)
    else:

        matches_c = np.stack(tuple(match_template(resize_im(plane, rf),
                                                  mi,
                                                  pad_input=True,
                                                  mode='minimum')
                                   for plane in ref[::coarse_step, :, :]), axis=0)

    optimal = matches_c.max(axis=(1, 2)).argmax() * coarse_step
    print('Optimal coarse: ', optimal)

    matches_fine = np.stack(tuple(match_template(plane,
                                                 moving_image,
                                                 pad_input=True,
                                                 mode='minimum')
                                  for plane in
                                  ref[(optimal - coarse_step):(optimal + coarse_step), ...]),
                            axis=0)
    optimal_plane = matches_fine.max(axis=(1, 2)).argmax()
    print('Optimal plane: ', optimal_plane)
    shifts = np.unravel_index(matches_fine[optimal_plane, ...].argmax(),
                              matches_fine.shape[1:])

    return (optimal_plane - coarse_step + optimal, *tuple(max(i - d // 2, 0) for i, d in zip(shifts, moving_image.shape)))


def find_scaling(

        zframe,
        im_res,
        scales=np.arange(0.9, 1.2, 0.05),
        rf=1,
        match_hist=False

):

    zf = resize_im(zframe, rf)
    scaled_ims = [resize_im(im_res, scale*rf) for scale in scales]

    if match_hist:

        matches_fine = np.stack(tuple(match_template(zf,
                                                     match_histograms(im, zf),
                                                     pad_input=True,
                                                     mode='minimum')
                                      for im in scaled_ims),
                                axis=0)
    else:

        matches_fine = np.stack(tuple(match_template(zf,
                                                     im,
                                                     pad_input=True,
                                                     mode='minimum')
                                      for im in scaled_ims),
                                axis=0)

    opt = matches_fine.max(axis=(1, 2)).argmax()
    opt_scale = scales[opt]
    opt_im = scaled_ims[opt]
    shifts = np.unravel_index(matches_fine[opt, ...].argmax(),
                              matches_fine.shape[1:])
    return opt_scale, opt_im, tuple(max(i // rf - d // 2, 0) for i, d in zip(shifts, opt_im.shape)), matches_fine.max()


def rotate_scale_shift(

        xys,
        shifts,
        scaling=1.,
        angle=135,
        dims=(512, 512),
        rotate_first=False
):
    """Generates xyz coordinates from xy coordinates from a single imaging plane"""

    rotated_xys = pixel_rotate(xys, angle=angle, dims=dims, rotate_first=rotate_first)
    scaled_xys = rotated_xys * scaling
    print(scaled_xys)

    # X and Y shifts are inverted, due to ROW/COL confusion for np arrays and xy coordinate system,
    # or maybe bug in XY extraction?
    z = np.array([shifts[0]]*xys.shape[0]).reshape(-1, 1) #* depth_factor I think we want to stay in pixel units
    x = np.array(scaled_xys[:, 0] + shifts[2]).reshape(-1, 1) #- 725./2. # *1.4????
    y = np.array(scaled_xys[:, 1] + shifts[1]).reshape(-1, 1) #- 725./2.

    shifted_xyz = np.concatenate((x, y, z), axis=1)

    return rotated_xys, scaled_xys, shifted_xyz


def resize_im(

        im,
        scaling

):
    return resize(im, (int(round(im.shape[0] * scaling)),
                       int(round(im.shape[1] * scaling))), preserve_range=True)


def scale_shift_rotate(

        xys,
        shifts,
        scaling=1.,
        angle=135,
        dims=(512, 512),
        rotate_first=False
):
    """Generates xyz coordinates from xy coordinates from a single imaging plane"""

    sc_xy = xys * scaling

    # X and Y shifts are inverted, due to ROW/COL confusion for np arrays and xy coordinate system,
    # or maybe bug in XY extraction?
    sh_x = np.array(sc_xy[:, 0] + shifts[2]).reshape(-1, 1)
    sh_y = np.array(sc_xy[:, 1] + shifts[1]).reshape(-1, 1)

    sh_xy = np.concatenate([sh_x, sh_y], axis=1)
    r_xy, (ox, oy) = pixel_rotate(sh_xy, angle=angle, dims=dims, rotate_first=rotate_first)
    z = np.array([shifts[0]]*xys.shape[0]).reshape(-1, 1) #* depth_factor I think we want to stay in pixel units

    r_xyz = np.concatenate([r_xy[:, 0].reshape(-1, 1), r_xy[:, 1].reshape(-1, 1), z], axis=1)

    return sc_xy, sh_xy, r_xyz, (ox, oy)


def pixel_rotate(

        points,
        angle=135,
        dims=(512, 512),
        rotate_first=False

):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    """
    angle = np.deg2rad(angle)
    ox, oy = dims[0]/2., dims[1]/2.
    corners = [[0, 0], [0, dims[1]], [dims[0], 0], [dims[0], dims[1]]]
    print(ox, oy)

    px, py = points[:, 0], points[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)

    if rotate_first:

        qx_min = min([ox + np.cos(angle) * (i - ox) - np.sin(angle) * (j - oy) for i, j in corners])
        qy_min = min([oy + np.sin(angle) * (i - ox) + np.cos(angle) * (j - oy) for i, j in corners])

        qx = qx - qx_min
        qy = qy - qy_min

    return np.concatenate((qx.reshape(-1, 1), qy.reshape(-1, 1)), axis=1), (ox, oy)


def calc_anglediff(

    unit1,
    unit2,
    theta=np.pi

):

    if unit1 < 0:
        unit1 += 2 * theta

    if unit2 < 0:
        unit2 += 2 * theta

    phi = abs(unit2 - unit1) % (theta * 2)
    sign = 1
    # used to calculate sign
    if not ((unit1 - unit2 >= 0 and unit1 - unit2 <= theta) or (
            unit1 - unit2 <= -theta and unit1 - unit2 >= -2 * theta)):
        sign = -1
    if phi > theta:
        result = 2 * theta - phi
    else:
        result = phi

    return result * sign


if __name__ == "__main__":

    mdict = {
        #'20210224_F1': (51, 53, 58, 59, 63, 72),
        #'20210224_F3': (81, False, False, False, False, False),
        #'20210303_F2': ((89, 224, 238), 93, 100, 104, False, False),
        #'20210303_F5': (61, 65, 71, 76, False, False),
        #'20210303_F6': (89, False, False, False, False, False),
        #'20210303_F7': (67, 72, 73, 81, 87, 90)
    }
    #mdict = {'20210120_F2': {1: (126, 147, 160, 173, 186, 197), 2:(137, 154, 167, 183, 190, 201)}}
    #mdict = {'20211209_F8': (119, 133, 139, 140, 142, 144)}

    rr = ROIRegistration(
        csv_path='J:/_Projects/JJ_Katja/FISH_DATABASE1.xlsx',
        impath='J:/Johannes Kappel/Imaging data/Theia',
        alpath='J:/Johannes Kappel//Alignment',
        #fkeys=['20211209_F{}'.format(i) for i in [8]],
        recompute=False,
        reextract=True,
        recwise=False,
        reverse_z=False,
        resize=True, #Had this param at false for all 81c Exp. since 20211209
        match_hist=False,
        use_maxim=False,
        #zfilt=(1, 1, 1),
        resc_z=(1, 95),
        resc_im=(1, 95),
        gamma=(.5, 1),
        #mdict=mdict,
        # zlim=(50,120),
        db_stim=['struct_1026']
    )

    rr.iter_dbase_main()


