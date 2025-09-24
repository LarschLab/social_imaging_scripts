import numpy as np
from scipy import interpolate
import os
import glob
import tifffile as tiff
import skimage.io
from skimage.filters import gaussian
from suite2p import registration, io
from skimage.transform import rotate, resize
from skimage.exposure import rescale_intensity
from scipy import ndimage
import time
import pandas as pd
import shutil
from scipy.signal import resample, convolve
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'C:/Users/jlarsch/Documents/social_imaging_scripts/analysis')
import tuninganalysis as Tuninganalysis
import pickle
import re
from sklearn.linear_model import LinearRegression


def take_one_slice():
    ts = tiff.imread(r'G:/Imaging_data/20200507/merged_planes_F1/merge_plane0_rec1_raw.tif')
    ts_frame = ts[0, :, :]
    skimage.io.imsave(r'G:/Imaging_data/20200507/merged_planes_F1/one_frame_raw', ts_frame.astype('int16'),
                      plugin='tifffile')
    return


def get_fish_age(date, fno):

    if (date == '20200623' and (fno == 1 or fno == 2)) or date == '20200728':
        age = 6
    elif (date == '20200521' and fno == 1) or date == '20200528' or date == '20200603' or (date == '20200617' and fno == 1) or (date == '20200701' and fno == 1) or (date == '20200708' and (fno == 1 or fno == 2)):
        age = 7
    elif (date == '20200611' and fno == 1) or (date == '20200619' and fno == 1) or (date == '20200813' and (fno == 1 or fno == 3)):
        age = 8
    elif (date == '20200611' and (fno == 3 or fno == 4) ) or (date == '20200623' and fno == 3):
        age = 13
    elif (date == '20200617' and (fno == 2 or fno == 3) ) or (date == '20200701' and fno == 3) or (date == '20200708' and fno == 6) or (date == '20200805') or (date == '20200812'):
        age = 14
    elif date == '20200813' and fno == 2:
        age = 15
    elif date == '20200617' and fno == 4:
        age = 19
    elif date == '20200610' or date == '20200625' or (date == '20200721' and (fno == 2 or fno == 5)):
        age = 20
    elif (date == '20200611' and fno == 5) or (date == '20200619' and (fno == 2 or fno == 3)) or (date == '20200701' and fno == 2) or (date == '20200708' and fno == 5):
        age = 21
    elif date == '20200723':
        age = 22
    elif date == '20200721' and fno == 8:
        age = 27
    elif date == '20190402':
        age = 'ask jj for age'
    else:
        age = 'unknown'
    return age


def apply_shifts_to_binary_new(ops, offsets, read_filename):
    """ apply registration shifts computed on one binary file to another

    Parameters
    ----------
    offsets : list of arrays
        shifts computed from reg_file_align/raw_file_align,
        rigid shifts in Y are offsets[0] and in X are offsets[1],
        nonrigid shifts in Y are offsets[3] and in X are offsets[4]
    raw_file_align : string
        file to read raw binary from (if not empty)
    """

    nfr = 0
    with io.BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], read_filename=read_filename) as f:

        data = np.empty(shape=(ops['nframes'], ops['Lx'], ops['Ly']), dtype='int16')
        for indices, frame in f.iter_frames():

            data[indices[0]] = frame.astype('int16')

        data = np.array(data)
        # get shifts
        nframes = data.shape[0]
        iframes = nfr + np.arange(0, nframes, 1, int)
        ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)

        # apply shifts
        if ops['bidiphase'] != 0 and not ops['bidi_corrected']:
            registration.register.bidiphase.shift(data, int(ops['bidiphase']))

        # if bidiphase_value != 0 and not bidi_corrected:
        #     registration.register.rigid.bidiphase.shift(data, bidiphase_value)
        for frame, dy, dx in zip(data, ymax, xmax):
            frame[:] = registration.register.rigid.shift_frame(frame=frame, dy=dy, dx=dx)
        #registration.register.rigid.shift_data(data, ymax, xmax)

        if ops['nonrigid']:
            ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]
            data = registration.register.nonrigid.transform_data(
                data,
                nblocks=ops['nblocks'],
                xblock=ops['xblock'],
                yblock=ops['yblock'],
                ymax1=ymax1,
                xmax1=xmax1
            )
        return data


def apply_shifts_to_binary(ops, offsets, reg_file_alt, raw_file_alt):
    ''' apply registration shifts to binary data'''
    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch
    ix = 0
    meanImg = np.zeros((Ly, Lx))
    k=0
    t0 = time.time()
    print('test123')
    if len(raw_file_alt) > 0:
        reg_file_alt = open(reg_file_alt, 'wb')
        raw_file_alt = open(raw_file_alt, 'rb')
        raw = True
    else:
        reg_file_alt = open(reg_file_alt, 'r+b')
        raw = False
    while True:
        print(raw)
        if raw:
            buff = raw_file_alt.read(nbytesread)
        else:
            buff = reg_file_alt.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        print(data.shape)
        buff = []
        if (data.size==0) | (ix >= ops['nframes']):
            print('wat')
            break
        data = np.reshape(data[:int(np.floor(data.shape[0]/Ly/Lx)*Ly*Lx)], (-1, Ly, Lx))
        nframes = data.shape[0]
        iframes = ix + np.arange(0,nframes,1,int)
        # get shifts
        ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)
        ymax1,xmax1 = [],[]
        if ops['nonrigid']:
            ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]
        # apply shifts
        for frame, dy, dx in zip(data, ymax, xmax):
            frame[:] = registration.register.rigid.shift_frame(frame=frame, dy=dy, dx=dx)

        if ops['nonrigid']:
            ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]
            data = registration.register.nonrigid.transform_data(
                data,
                nblocks=ops['nblocks'],
                xblock=ops['xblock'],
                yblock=ops['yblock'],
                ymax1=ymax1,
                xmax1=xmax1
            )
        data = np.minimum(data, 2**15 - 2)
        meanImg += data.mean(axis=0)
        data = data.astype('int16')
        # write to binary
        if not raw:
            reg_file_alt.seek(-2*data.size,1)
        reg_file_alt.write(bytearray(data))
        # write registered tiffs
        if ops['reg_tif_chan2']:
            write(data, ops, k, False)
        ix += nframes
        k+=1
    if ops['functional_chan']!=ops['align_by_chan']:
        ops['meanImg'] = meanImg/k
    else:
        ops['meanImg_chan2'] = meanImg/k
    print('Registered second channel in %0.2f sec.'%(time.time()-t0))
    reg_file_alt.close()
    if raw:
        raw_file_alt.close()
    return ops


def write(data, ops, k, ichan):
    """ writes frames to tiffs
    Parameters
    ----------
    data : int16
        frames x Ly x Lx
    ops : dictionary
        requires 'functional_chan', 'align_by_chan'
    k : int
        number of tiff
    ichan : bool
        channel is ops['align_by_chan']
    """
    print('test456')
    if ichan:
        if ops['functional_chan']==ops['align_by_chan']:
            tifroot = os.path.join(ops['save_path'], 'reg_tif')
            wchan = 0
        else:
            tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
            wchan = 1
    else:
        if ops['functional_chan']==ops['align_by_chan']:
            tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
            wchan = 1
        else:
            tifroot = os.path.join(ops['save_path'], 'reg_tif')
            wchan = 0
    if not os.path.isdir(tifroot):
        os.makedirs(tifroot)
    fname = 'file%0.3d_chan%d.tif'%(k,wchan)
    with tiff.TiffWriter(os.path.join(tifroot, fname)) as tif:
        for i in range(data.shape[0]):
            tif.save(data[i])


def generate_multiplane_tiff(

        lpath,
        date,
        fno,
        rec='',
        crop=(0, 1500),
        angle=135,
        nplanes=6,
        ds_factor=1,
        save_ind=False,
        destpath=None
):

    ts_row1 = []
    ts_row2 = []
    print(destpath)
    print(os.listdir(os.path.join(lpath, date, 'fish{}'.format(fno + 1), rec)))
    tiffs = glob.glob(os.path.join(lpath, date, 'fish{}'.format(fno + 1), rec, 'downsampled*registered_plane[0-9].tif'))
    print(lpath, date, tiffs)
    if len(tiffs) != nplanes:
        print('Number of downsampled tiffs does not match number of planes! Exiting.')
        return

    for tno, tiffpath in enumerate(sorted(tiffs)):
        print(tiffpath)
        ts = tiff.imread(tiffpath)
        ts = ts[crop[0]:crop[1]]
        if ds_factor != 1:
            ts = resize(ts, (int(ts.shape[0] / ds_factor), int(ts.shape[1] / ds_factor), int(ts.shape[2] / ds_factor)),
                        preserve_range=True)
        print(ts.dtype)
        # ts = rescale_intensity(ts,
        #                          in_range=(
        #                              np.percentile(ts, 0.01), np.percentile(ts, 99.99)),
        #                          out_range=(0, 255)).astype(np.uint8)
        #ts = ndimage.rotate(ts, axes=(1, 2), angle=angle, reshape=True, cval=0).astype(np.uint8)

        ts = np.flip(ts, axis=1)
        print(ts.shape, ts.dtype)

        if save_ind:

            tiff.imsave(os.path.join(tiffpath.split('.')[0]), '_dsrot.tif', ts)

        if tno < 2:

            ts_row1.append(ts)

        elif tno < 4:

            ts_row2.append(ts)

        else:

            ts_row3.append(ts)

        del ts

    ts_row1 = np.concatenate(ts_row1, axis=1)
    ts_row2 = np.concatenate(ts_row2, axis=1)
    ts_row3 = np.concatenate(ts_row3, axis=1)

    ts_full = np.concatenate([ts_row1, ts_row2], axis=2)

    if destpath is None:
        destpath = lpath
    if not os.path.exists(os.path.join(destpath, date,  'fish{}'.format(fno+1))):

        os.makedirs(os.path.join(destpath, date,  'fish{}'.format(fno+1)))

    tiff.imsave(os.path.join(destpath, date,  'fish{}'.format(fno+1), rec, 'ds_rot_crop_full.tif'), ts_full)
    return


def transfer_database(db_stim='speed_fast_acc_ccw', save_dsfile=False, recwise=False):

    db = pd.read_excel('J:/_Projects/JJ_Katja/FISH_DATABASE1.xlsx', sheet_name='FISHES_OVERVIEW',
                            dtype={'Age': int, 'FNO': int, 'Date': str, 'Path': str})

    #cond = (db.Status == 'Done') & (db.Registration_Status == 'Done') & (db.Stim == db_stim) & (db.Age.values > 18)
    cond = (db.Stim == db_stim) & (db.Status != 'Bad')
    imdest = 'C:/Johannes Kappel/Imaging data/Theia'
    stimdest = 'C:/Johannes Kappel/Stimuli data/Theia'

    for idx, row in db[cond].iterrows():

        print(row.Date, row.FNO)
        # if row.Date.startswith('2019'):
        #
        #     continue
        # Imaging data
        fdest = os.path.join(imdest, row.Date, 'fish{}'.format(row.FNO))

        try:
            os.makedirs(fdest)
        except FileExistsError:
            pass

        fpath = os.path.join(row.Path, row.Date, 'fish{}'.format(row.FNO))
        try:
            os.makedirs(fdest)
        except FileExistsError:
            pass
        for plane in range(int(row.Nplanes)):

            if recwise:

                if row.Nplanes > 1:

                    pdests = [os.path.join(fdest, 'rec{}'.format(rno.split('\\')[-1].strip('rec')), 'plane{}'.format(plane))
                                for rno in glob.glob(fpath+'/rec*')]

                    ppaths = [os.path.join(fpath, 'rec{}'.format(rno.split('\\')[-1].strip('rec')), 'plane{}'.format(plane))
                                for rno in glob.glob(fpath+'/rec*')]

                else:
                    pdests = [os.path.join(fdest, 'rec{}'.format(rno.split('\\')[-1].strip('rec')), 'suite2p', 'plane0')
                              for rno in glob.glob(fpath + '/*')]

                    ppaths = [os.path.join(fpath, 'rec{}'.format(rno.split('\\')[-1].strip('rec')), 'suite2p', 'plane0')
                                for rno in glob.glob(fpath + '/*')]


            else:

                pdests = [os.path.join(fdest, 'plane{}'.format(plane))]
                ppaths = [os.path.join(fpath, 'plane{}'.format(plane))]

            for ppath, pdest in zip(ppaths, pdests):

                try:
                    os.makedirs(pdest)
                except FileExistsError:
                    pass

                pfiles = [i for i in os.listdir(ppath) if not 'data' in i
                          ]
                print(pfiles)
                for pfile in pfiles:
                    if not '.npy' in pfile:
                        continue
                    pfdest = os.path.join(pdest, pfile)
                    if os.path.exists(pfdest):
                        print('Data already transfered..')
                        pass
                    else:
                       shutil.copyfile(os.path.join(ppath, pfile), pfdest)

            if save_dsfile:

                    ds_file = glob.glob(os.path.join(fpath, 'downsampled*registered_plane{}.tif'.format(plane)))
                    print('Generating ds_file')
                    with tiff.TiffFile(
                            ds_file[0],
                            is_scanimage=False) as ts_file:

                        ts = np.array([i.asarray() for i in ts_file.pages])[:1000]
                    av_im = np.std(ts, axis=0).astype(np.float32)

                    tiff.imsave(os.path.join(fdest, 'downsampled_registered_plane{}_av_std.tif'.format(plane)), av_im)
            try:
                shutil.copytree(os.path.join(fpath, 'zstack'), os.path.join(fdest, 'zstack'))
            except FileNotFoundError:
                pass

            # generate_multiplane_tiff(
            #
            #     lpath,
            #     row.Date,
            #     row.FNO-1,
            #     rec='',
            #     crop=(0, -1),
            #     angle=135,
            #     nplanes=6,
            #     ds_factor=4,
            #     save_ind=False,
            #     destpath=imdest
            # )

            #Stimuli data
            sfdest = os.path.join(stimdest, row.Date, 'fish{}'.format(row.FNO))
            try:
                os.makedirs(sfdest)
            except FileExistsError:
                pass

            sfpath = os.path.join(row.StimPath, row.Date, 'fish{}'.format(row.FNO))

            sfiles = [i for i in os.listdir(sfpath) if 'stimuli' in i or 'GCaMP6s' in i]
            print(sfiles)
            try:
                [shutil.copyfile(os.path.join(sfpath, i), os.path.join(sfdest, i)) for i in sfiles]
            except FileExistsError:
                pass


def transfer_data():

    impath = 'J:/Johannes Kappel/Imaging data/Theia'
    imdest = 'D:/Johannes Kappel/Imaging data/Theia'

    stimpath = 'J:/Johannes Kappel/Stimuli data/Theia'
    stimdest = 'D:/Johannes Kappel/Stimuli data/Theia'
    # for date in ['20201203', '20201210', '20210120']:
    #
    #     dpath = os.path.join(impath, date)
    #     for fish in os.listdir(dpath):
    #
    #         fpath = os.path.join(dpath, fish)
    #         print(fpath)
    #
    #         for rno, rpath in enumerate(glob.glob(os.path.join(fpath, 'rec[0-9]'))):
    #             print(rpath)
    #             rdest = os.path.join(imdest, date, fish, 'rec{}'.format(rno+1))
    #             # dsfile = glob.glob(rpath + '/ds_rot_crop_full.tif')[0]
    #             # print(dsfile, rdest)
    #             # shutil.copyfile(dsfile, os.path.join(rdest, 'ds_rot_crop_full.tif'))
    #             # continue
    #             for plane in range(6):
    #
    #                 pdest = os.path.join(imdest, date, fish, 'rec{}'.format(rno+1), 'plane{}'.format(plane))
    #                 print(pdest)
    #                 try:
    #                     os.makedirs(pdest)
    #                 except FileExistsError:
    #                     pass
    #                 ppath = os.path.join(rpath, 'plane{}'.format(plane))
    #                 print(ppath)
    #                 pfiles = [i for i in os.listdir(ppath) if not 'data' in i]
    #                 print(pfiles)
    #                 [shutil.copyfile(os.path.join(ppath, i), os.path.join(pdest, i)) for i in pfiles]

    for date in ['20201203', '20201210', '20210120']:

        dpath = os.path.join(stimpath, date)
        for fish in os.listdir(dpath):

            fpath = os.path.join(dpath, fish)
            print(fpath)
            for rno, rpath in enumerate(glob.glob(os.path.join(fpath, 'rec[0-9]'))):
                print(rpath)
                rdest = os.path.join(stimdest, date, fish, 'rec{}'.format(rno+1))
                try:
                    os.makedirs(rdest)
                except FileExistsError:
                    pass
                pfiles = [i for i in os.listdir(rpath) if not i.endswith('mp4')]
                print(pfiles)
                [shutil.copyfile(os.path.join(rpath, i), os.path.join(rdest, i)) for i in pfiles]

def make_stimspeed_regressor(

        stimpath='C:/Users/jkappel/Downloads/nat_stim_experiment_2min.txt',
        imaging_len=3600,
        single_stim_min=2,
        stim_delay_sec=30,
        fps=30,
        stim_fps=60,
        dish_size=[500, 500]

):
    stim_raw = np.loadtxt(stimpath)[::int(stim_fps / fps), :]
    fr_delay = stim_delay_sec * fps
    fr_stim = single_stim_min * 60 * fps
    fr_full_stim = fr_delay + fr_stim
    stim_xy = np.zeros((len(np.where(stim_raw[:, 0] == -550)[0]) * (
            fr_delay + np.where(stim_raw[:, 0] == -550)[0][1]), 2))
    for s_idx, s_val in enumerate(np.where(stim_raw[:, 0] == -550)[0]):

        stim_xy[fr_delay + s_idx * fr_full_stim:(s_idx + 1) * fr_full_stim] = \
            stim_raw[s_val:s_val + fr_stim]

    speed = np.sqrt((stim_xy[1:, 0] - stim_xy[:-1, 0]) ** 2 + (stim_xy[1:, 1] - stim_xy[:-1, 1]) ** 2)
    speed[np.where(stim_xy[:-1, 0] > dish_size[0])] = 0
    speed[np.where(stim_xy[:-1, 0] < 0)] = 0
    speed[np.where(stim_xy[:-1, 1] > dish_size[1])] = 0
    speed[np.where(stim_xy[:-1, 1] < 0)] = 0
    speed[fr_delay - 1::fr_full_stim] = 0
    speed[fr_full_stim - 1::fr_full_stim] = 0
    stim_xy = stim_xy[:imaging_len * fps, :]
    speed = speed[:imaging_len * fps]
    speed_res = resample(speed, 3600)
    speed_regressor = convolve_ts(speed_res)
    return speed_regressor


def trans_nrrd(tup, header=False):

    im = tup[0].astype(np.uint8)
    im = np.moveaxis(im, 2, 0)
    im = np.moveaxis(im, 1, 2)
    if header:
        return im, tup[1]
    else:
        return im


def convolve_ts(

        ts,
        sampling_interval=1,
        toff=7,
        delay=30,

):
    t = np.linspace(0, delay, int(delay / sampling_interval))
    t = np.hstack((-np.ones(int(delay / sampling_interval)), t))
    e = np.exp(-t / toff) * (1 + np.sign(t)) / 2
    e = e / np.max(e)

    return np.convolve(ts, e, mode='same') / np.max(convolve(ts, e, mode='same'))


def get_opticflow_regs(mreg):

    # GCaMP6s kernel
    si = 1
    toff = 7
    delay = 15

    # stimulus onset times
    stn = np.array(
        [20, 62, 104, 146, 207, 249, 292, 333, 385, 427, 469, 511, 573, 615, 657, 698, 751, 793, 834, 876, 938, 980,
         1022, 1064])

    fs = np.zeros((1200, stn.size + 1))
    fs_all = np.zeros((1200))

    nregs = mreg.values.shape[0]
    fsm = np.zeros((1200, nregs))

    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    for i in range(stn.size):
        fs[stn[i]:stn[i] + 13, i] = 1
        fs_all[stn[i]:stn[i] + 13] = 1
        fs[:, i] = convolve_ts(fs[:, i], si, toff, delay)
        ax.plot(fs[:, i] + i)

    fs_all = convolve_ts(fs_all, si, toff, delay)
    ax.plot(fs_all + stn.size + 1)
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 10), dpi=150)
    for i in range(nregs):

        boolidx = np.concatenate([
            np.where(mreg.values[i, 1:])[0],
            np.where(mreg.values[i, 1:])[0] + 8,
            np.where(mreg.values[i, 1:])[0] + 16], axis=0)
        regidxs = stn[boolidx]
        for idx in regidxs:
            fsm[idx:idx + 13, i] = 1
        fsm[:, i] = convolve_ts(fsm[:, i], si, toff, delay)
        ax.plot(fsm[:, i] + i)
    ax.set_yticks(range(nregs))
    ax.set_yticklabels(mreg.values[:, 0])
    plt.close()

    return fs, fs_all, fsm


def analyse_database(ages, db_stim, ds_fac=5., frate=30.):

    dbfile = 'J:/_Projects/JJ_Katja/FISH_DATABASE1.xlsx'
    #dbfile = 'C:/Users/jkappel/Downloads/FISH_DATABASE.xlsx'
    sumdict = pd.DataFrame({})
    db = pd.read_excel(dbfile, sheet_name='FISHES_OVERVIEW', engine='openpyxl',
                          dtype={'Age': int, 'FNO': int, 'Date': str, 'Path': str})

    cond = (db.Status == 'Done') & \
           (db.Registration_Status == 'Done') & \
           (db.Stim == db_stim) & \
           (ages[0] < db.Age) & \
           (db.Age < ages[1])

    # cond = (db.Stim == db_stim) & \
    #        (ages[0] < db.Age) & \
    #        (db.Age < ages[1])

    fid = -1
    cells_total = 0
    for idx, row in db[cond].iterrows():
        if not str(row.Date)=='20200721':
            continue
        if not int(row.FNO)==5:
            continue
        print(row.Date, row.FNO, row.Stim)
        # Imaging data
        # if 'dim' in str(row.Notes) or 'high' in str(row.Notes):
        #
        #     print(row.Notes, ' excluded')
        #     continue
        print(row.Notes)

        fid += 1
        path = 'E:/Johannes Kappel/Imaging data/Theia'
        stimpath = 'E:/Johannes Kappel/Stimuli data/Theia'

        generate_multiplane_tiff(

                row.Path,
                row.Date,
                row.FNO-1,
                rec='',
                crop=(0, -1),
                angle=135,
                nplanes=6,
                ds_factor=1,
                save_ind=False,
                destpath=r'J:/Johannes Kappel/Imaging data/'
        )
        continue
        se = Tuninganalysis.StimuliExtraction(stimpath, path, frate=frate, ds_fac=ds_fac, clip_resp_scores=2.)
        rec_crop = tuple([int(row.Recs_Checked.strip('()').split(',')[0]),
                         int(row.Recs_Checked.strip('()').split(',')[-1])])
        print(rec_crop)
        se.set_date_and_fish(row.Date, row.FNO, row.Nplanes, rec_crop, 600)  # TODO generalize rec_offsets
        se.extract_stim()
        fimpath = os.path.join(path, row.Date, 'fish{}'.format(row.FNO))

        for plane in range(int(row.Nplanes)):

            traces = np.load(os.path.join(fimpath, 'plane{}'.format(plane), 'F.npy'))

            tmin, tmax = traces.min(axis=1).reshape(-1, 1), traces.max(axis=1).reshape(-1, 1)
            traces = (traces - tmin) / (tmax - tmin)

            _ = se.score_neurons(traces)
            stats, iscell, iscell_lab = [np.load(
                        os.path.join(
                            se.impath,
                            '{0}/fish{1}/plane{2}/{3}.npy'.format(se.date, se.fno, plane, i)
                        ), allow_pickle=True) for i in ['stat', 'iscell', 'iscell_lab']]

            nd = {

                'date': [se.date] * stats.shape[0],
                'fno': [row.FNO] * stats.shape[0],
                'age': [row.Age] * stats.shape[0],
                'plane': [plane] * stats.shape[0],
                'peak_cw': np.median(se.peak_t[:, :int(len(se.unique_p) / 2)], axis=1),
                'peak_ccw': np.median(se.peak_t[:, int(len(se.unique_p) / 2):], axis=1),
                'x_coor': [stats[neuron]['med'][1] for neuron in range(stats.shape[0])],
                'y_coor': [stats[neuron]['med'][0] for neuron in range(stats.shape[0])],
                'fid': [fid] * stats.shape[0],
                'nid': [i for i in range(stats.shape[0])],
                'unid': [i for i in range(cells_total, cells_total+stats.shape[0])],
                'iscell': iscell[:, 0].astype(bool),
                'iscell_lab': iscell_lab[:, 0].astype(bool)

            }
            print(stats.shape, se.peak_t.shape)
            for sno, stim in enumerate(se.unique_p):

                nd['rs_{}'.format(stim)] = se.resp_scores[:, sno]
                nd['m_{}'.format(stim)] = se.magnitudes[:, sno]
            nd = pd.DataFrame(nd)
            sumdict = pd.concat([sumdict, nd], axis=0)
            cells_total += stats.shape[0]
    pickle.dump(sumdict, open('{}_sumdict.p'.format(db_stim), 'wb'))
    print('_')


def rewrite_nrrd_hcr(zfile_nrrd):

    import nrrd
    zstack, header = nrrd.read(zfile_nrrd)
    print(header)
    header = {
        'space dimension': 3,
        'space directions': np.array([
            [2., 0, 0],
            [0, 2., 0],
            [0, 0, 2.]]).astype(float),
        'space units': ["microns", "microns", "microns"],
        'type': 'uint8',
        'PixelType': 'uint8'
    }
    # zfile_inv = zfile_nrrd.split('.')[0] + '_aligned_inv_zres.nrrd'
    # nrrd.write(zfile_inv, zstack[:, :, ::-1], header=header)
    nrrd.write(zfile_nrrd, zstack, header=header)
    return

def read_frate_singleplane(planepath):

    txtfile = [i for i in os.listdir(planepath) if i.endswith('metadata.txt')][0]
    txt = open(os.path.join(planepath, txtfile), 'rb')
    a = [str(i) for i in txt.readlines() if 'D3Step' in str(i)][0]
    print(a, planepath)
    x = re.findall("\d+\.\d+", a)
    frate = 1000. / float(x[0])
    print('Detected frame rate:', frate)
    return frate

def inbal_ds():

    recfiles = glob.glob(r'J:\Inbal Shainer\itpr1b_calcium_imaging_20220328_spatial_frequency\*itpr1b*/*.tif')
    for recfile in recfiles:
        print(recfile)
        print(r'J:/Johannes Kappel/Imaging data/' + '/'.join(recfile.split('\\')[-2:]))
        with tiff.TiffFile(recfile, is_scanimage=False) as ts_file:

            ts = np.array([i.asarray() for i in ts_file.pages])
        ts_ds = np.array([np.mean(ts[i:i + 6], axis=0) for i in np.arange(0, ts.shape[0], 6)])
        skimage.io.imsave(r'J:/Johannes Kappel/Imaging data/' + '/'.join(recfile.split('\\')[-2:]), ts_ds.astype('int16'), plugin='tifffile')

if __name__ == '__main__':

    #analyse_database([17, 23], 'bout_cont_18dim')
    transfer_database(db_stim='dot_wf', save_dsfile=False, recwise=True)
    #transfer_database(db_stim='dots_grating_loom_dim_abl', save_dsfile=True)

    #transfer_database(db_stim='dots_grating_loom_nat_abl', save_dsfile=True)
    #transfer_database(db_stim='speed_acc_cw')
    #transfer_database(db_stim='nat_circ_flic_cow')
    # nrrdfs = glob.glob('J:/_Projects/social_HCR/32_animals_wt/nrrd_all/C*_aligned_ref_mc2.nrrd')
    # for nrrdf in nrrdfs:
    #     print(nrrdf)
    #     rewrite_nrrd_hcr(nrrdf)
    #transfer_data()
    # for fno in [1,2]:
    #     for rec in ['rec1', 'rec2']:
    #         generate_multiplane_tiff(
    #
    #             'J:/Johannes Kappel/Imaging data/Theia',
    #             '20210225',
    #             fno,
    #             rec=rec,
    #             crop=(0, 1500),
    #             angle=135,
    #             nplanes=6,
    #             ds_factor=4,
    #             save_ind=False,
    #             destpath=None
    #         )