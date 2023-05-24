import nrrd
import numpy as np
import os
import tifffile as tiff
import pandas as pd
import sys
sys.path.insert(1, 'C:/Users/jkappel/PycharmProjects/SocialVisionSuite2p')
import Utils
import Tuninganalysis
import pickle
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class BoutPreferenceAnalysis:

    def __init__(self, **kwargs):

        self.dbfile = kwargs.get('dbfile', 'J:/_Projects/JJ_Katja/FISH_DATABASE1.xlsx')
        self.db_stim = kwargs.get('db_stim', 'bout_cont_18dim')
        self.alpath = kwargs.get('alpath', 'J:/Johannes Kappel/Alignment')
        self.impath = kwargs.get('impath', None)
        self.stimpath = kwargs.get('stimpath', None)
        self.reganalysis = kwargs.get('reganalysis', False)
        self.anatomy = kwargs.get('anatomy', True)
        self.multirec = kwargs.get('multirec', False)
        self.roitags = kwargs.get('roitags', ['_hcr', '_scaled_hcrref'])
        self.zspath = kwargs.get('zstack', 'J:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd')
        self.age = kwargs.get('age', (0, 99))
        self.read_frate = kwargs.get('read_frate', False)
        self.frate = kwargs.get('frate', 30)
        self.ds_fac = kwargs.get('ds_fac', 5.)
        self.micdelay = kwargs.get('micdelay', 0)
        self.norm_traces = kwargs.get('norm_traces', True)
        self.foldertag = kwargs.get('foldertag', '')
        self.filetag = kwargs.get('filetag', '')
        self.tau = kwargs.get('tau', 7.)
        self.regexzero = kwargs.get('regexzero', False)

        self.regs = None
        self.regscores = None
        self.unique_p = None
        self.traces = None
        self.stimparams = None
        self.fpath = None
        self.delays= None
        self.zstack = None
        self.roistack = None
        self.roistack_f = None
        self.scores = None

        self.sd = {}
        self.roi_df = pd.DataFrame()
        self.roi_df_hcr = pd.DataFrame()
        self.micron_scaling = (1, 1, 1)

    def construct_regressors(self):

        fparams = sorted([[i] for i in self.unique_p])
        self.regs = np.zeros((self.traces.shape[1], len(fparams))) * np.nan
        for rno, regparams in enumerate(fparams):

            reg = np.zeros(self.traces.shape[1])
            start_stop = np.array([i[1:] for i in self.stimparams if i[0] in regparams])

            for j in start_stop:
                reg[j[0]:j[1]] = 1

            reg_conv = Utils.convolve_ts(

                reg,
                sampling_interval=1.,
                toff=self.tau,
                delay=self.delays[0],

            )

            reg_conv /= reg_conv.max()
            reg_conv[np.isnan(reg_conv)] = 0

            self.regs[:, rno] = reg_conv.reshape(-1)
        return

    def get_regscores(self, reg, lim=None):

        rbool = np.where(reg!=0)
        if self.regexzero:
            lreg = LinearRegression().fit(reg.T.reshape(-1, 1)[rbool], self.traces.T[rbool])
        else:
            lreg = LinearRegression().fit(reg.T.reshape(-1, 1), self.traces[:, :lim].T)

        rsme = np.nanmean(np.sqrt((self.traces[:, :lim] - reg) ** 2), axis=1).reshape(-1, 1)
        v = ((reg - reg.mean()) ** 2).sum()
        r2 = 1 - (rsme / v)
        r2[np.isinf(r2)] = 0
        coefs = lreg.coef_
        scr = coefs * r2
        scores = np.concatenate([scr, coefs], axis=1)
        return scores

    def get_regscores_multi(self):

        self.regscores = np.zeros((self.traces.shape[0], self.regs.shape[1], 2)) * np.nan
        for rno in range(self.regs.shape[1]):

            scores = self.get_regscores(self.regs[:, rno])
            self.regscores[:, rno] = scores

        return self.regscores

    def iter_dbase(self):

        if self.anatomy:

            if self.zspath.endswith('tif'):
                self.zstack = tiff.imread(self.zspath)
            elif self.zspath.endswith('.nrrd'):
                self.zstack, header = Utils.trans_nrrd(nrrd.read(self.zspath), header=True)
                self.micron_scaling = np.array([header['space directions'][i].max() for i in range(3)]).astype(float)
                print('Micron scale: ', self.micron_scaling)
            else:
                print('Unknown file format.')
                return None

        sumdict = pd.DataFrame()

        db = pd.read_excel(self.dbfile, sheet_name='FISHES_OVERVIEW', engine='openpyxl',
                              dtype={'Age': int, 'FNO': int, 'Date': str, 'Path': str})
        self.db = db
        print('DBSTIM', self.db_stim)
        if self.db_stim == 'bout_cont_18dim':
            cond = (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'speed_acc_cw':
            cond = (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'nat_circ_flic_cow':
            cond = (db.Stim == self.db_stim) & \
                   (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'social_nat_bout_2f':
            cond = (db.Status != 'Bad') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'dot_wf':
            cond = (db.Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'dots_grating_loom_nat_abl':
            cond = (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'dots_grating_loom_dim_abl':
            cond = (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])
        elif self.db_stim == 'dot_wf_abl':
            cond = (db.Stim == self.db_stim) & \
                   (db.Status == 'Done')
        elif self.db_stim == 'dot_wf_cyt':
            cond = (db.Stim == self.db_stim) & \
                   (db.Status == 'Done')
        elif self.db_stim == 'dot_wf_iso':
            cond = (db.Stim == self.db_stim) & \
                   (db.Status == 'Done')
        elif self.db_stim == 'pizza_pth2':
            cond = (db.Stim == self.db_stim) & \
                   (db.Status == 'Done')
        elif self.db_stim == 'dot_wf_juv':
            cond = (db.Status == 'Done') & \
                   (db.Registration_Status == 'Done') & \
                   (db.Stim == self.db_stim) & \
                   (self.age[0] < db.Age) & \
                   (db.Age < self.age[1])

        fid = -1
        cells_total = 0
        unid = []
        lam = []
        print(db[cond].Date.unique())
        for idx, row in db[cond].iterrows():

            print(row.Date, row.FNO, row.Stim)
            print(row.Notes)

            fid += 1

            if self.impath is None:
                self.impath = row.Path
            if self.stimpath is None:
                self.stimpath = row.StimPath

            self.fpath = os.path.join(self.impath, row.Date, 'fish{}'.format(row.FNO))
            self.alpath_f = os.path.join(self.alpath, row.Date, 'fish{}'.format(row.FNO))
            self.fkey = '{}_F{}'.format(row.Date, row.FNO)

            if self.multirec:
                mrec = list(range(int(row.Multirec.strip('()').split(',')[0])+1,
                                  int(row.Multirec.strip('()').split(',')[-1])+1))
            else:
                mrec = [None]

            if self.read_frate and not self.multirec:

                self.frate = Utils.read_frate_singleplane(os.path.join(self.fpath, 'plane0')) # hacky solution for single plane imaging on femtonics
                print('Detected frame rate: ', self.frate)

            se = Tuninganalysis.StimuliExtraction(
                self.stimpath,
                self.impath,
                frate=self.frate,
                ds_fac=self.ds_fac,
                micdelay=self.micdelay,
                clip_resp_scores=2.,
                recs=mrec,
                reganalysis=True # defaulting to True for now, assuming that including delay increases noise
            )

            rec_crop = tuple([int(row.Recs_Checked.strip('()').split(',')[0]),
                              int(row.Recs_Checked.strip('()').split(',')[-1])])
            se.set_date_and_fish(row.Date, row.FNO, row.Nplanes, rec_crop,
                                 int(row.Rec_Offset))

            if not self.multirec:

                if self.db_stim == 'dot_wf_iso':
                    if row.Nplanes == 1:
                        se.plane = 0
                se.extract_stim()
                self.unique_p, self.stimparams = se.unique_p, se.stimparams

            else:

                stimdicts = se.extract_stim_multirec(read_frate=self.read_frate, fpath=self.fpath)
                self.unique_p = se.unique_p
                self.stimparams = [stim for sd in sorted(stimdicts.keys()) for stim in stimdicts[sd]['stimparams']]

            self.delays = np.concatenate([se.protocols[i].delay.values for i in range(len(se.protocols))], axis=0)

            if row.Age < 10:

                self.folder = 'one_week'

            elif 9 < row.Age < 17:

                self.folder = 'two_weeks'

            else:

                self.folder = 'three_weeks'

            if self.anatomy:
                # Anatomy
                nids = np.load(os.path.join(self.alpath, self.folder, '{}_nids.npy'.format(self.fkey)))
                plabels = np.load(os.path.join(self.alpath, self.folder, '{}_plabels.npy'.format(self.fkey)))
                for roitag in self.roitags:

                    xyzpath = os.path.join(self.alpath, self.folder, '{}_shifted_xyzs_aligned{}.csv'.format(self.fkey, roitag))
                    xyz = pd.read_csv(xyzpath).values
                    zyx = np.round(
                        np.concatenate([xyz[:, 2:3], xyz[:, 1:2], xyz[:, 0:1]], axis=1)).astype('int')

                    roi_df = pd.DataFrame({
                        'x': zyx[:, 2],
                        'y': zyx[:, 1],
                        'z': zyx[:, 0],
                        'plane': plabels,
                        'nid': nids,
                        'fid':[fid] * nids.shape[0]
                    })

                    if roitag == '_hcr':
                        self.roi_df_hcr = pd.concat([self.roi_df_hcr, roi_df], axis=0)
                    elif roitag == '_scaled_hcrref' or  roitag == '_ref' or roitag == '':
                        self.roi_df = pd.concat([self.roi_df, roi_df], axis=0)


            if self.multirec:
                p_iter = [[plane, rec] for rec in mrec for plane in range(int(row.Nplanes))]
            else:
                p_iter = [[plane, None] for plane in range(int(row.Nplanes))]
            for plane, rec in p_iter:

                print('Plane ', plane)
                if rec is not None:

                    self.fpath = os.path.join(self.fpath, 'rec{}'.format(rec))
                    se.unique_p = stimdicts[rec]['unique_p']
                    se.stimparams = stimdicts[rec]['stimparams']

                # self.traces, stats, iscell, iscell_lab = [np.load(
                #     os.path.join(self.fpath, '{0}plane{1}/{2}.npy'.format(self.foldertag, plane, i)
                #     ), allow_pickle=True) for i in ['F', 'stat', 'iscell', 'iscell_lab']]
                self.traces, stats, iscell = [np.load(
                    os.path.join(self.fpath, '{0}plane{1}/{2}.npy'.format(self.foldertag, plane, i)
                    ), allow_pickle=True) for i in ['F', 'stat', 'iscell']]

                if self.norm_traces:

                    tmin, tmax = self.traces.min(axis=1).reshape(-1, 1), self.traces.max(axis=1).reshape(-1, 1)
                    self.traces = (self.traces - tmin) / (tmax - tmin)

                self.traces[np.isnan(self.traces)] = 0.
                self.snr = np.median(np.abs(np.diff(self.traces, axis=1)), axis=1)/np.sqrt(se.volrate)

                _ = se.score_neurons(self.traces)
                print(self.unique_p)

                if self.anatomy:

                    # Anatomy
                    if self.multirec:

                        unid.extend(roi_df[roi_df.plane == (plane+(rec-1)*row.Nplanes)].nid.values + cells_total)

                    else:

                        unid.extend(roi_df[roi_df.plane == plane].nid.values + cells_total)


                    lam.extend([j for i in range(stats.shape[0]) for j in stats[i]['lam']])

                if self.reganalysis:

                    self.construct_regressors()
                    self.get_regscores_multi()

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
                    'snr': self.snr,
                    #'iscell_lab': iscell_lab[:, 0].astype(bool),
                    'm_bl': np.nanmean(se.baselines, axis=1)

                }
                if self.reganalysis:
                    for sno, stim in enumerate(sorted(self.unique_p)):

                        nd['regsc_{}'.format(stim)] = self.regscores[:, sno, 0]
                        nd['regcoef_{}'.format(stim)] = self.regscores[:, sno, 1]

                print(stats.shape, se.peak_t.shape)
                for sno, stim in enumerate(se.unique_p):

                    nd['rs_{}'.format(stim)] = se.resp_scores[:, sno]
                    nd['m_{}'.format(stim)] = se.magnitudes[:, sno]

                if rec is not None:

                    self.fpath = os.path.join(self.impath, row.Date, 'fish{}'.format(row.FNO))
                    nd['rec'] = [rec]* stats.shape[0]

                if self.db_stim == 'dots_grating_loom_nat_abl':
                    if row.Notes == '81C':
                        is81c = 1
                    else:
                        is81c = 0
                    nd['is81c'] = [is81c] * stats.shape[0]

                nd = pd.DataFrame(nd)
                sumdict = pd.concat([sumdict, nd], axis=0)
                cells_total += stats.shape[0]

                self.sd = sumdict

        pickle.dump(sumdict, open('{}_sumdict{}.p'.format(self.db_stim, self.filetag), 'wb'))

        if self.anatomy:

            self.roi_df['unid'] = unid
            self.roi_df['lam'] = lam

            self.roi_df_hcr['unid'] = unid
            self.roi_df_hcr['lam'] = lam

            if '_scaled_hcrref' in self.roitags:
                pickle.dump(self.roi_df, open('{}_roidf_scaled_hcrref.p'.format(self.db_stim), 'wb'))
            if '_hcr' in self.roitags:
                pickle.dump(self.roi_df_hcr, open('{}_roidf_hcr.p'.format(self.db_stim), 'wb'))
            if '_ref' or '' in self.roitags:
                pickle.dump(self.roi_df, open('{}_roidf{}.p'.format(self.db_stim, self.filetag), 'wb'))

        #self.create_roistack(tag='merge_20211206')

        return self.sd, self.roi_df

    def get_traces(self, sdt):

        alltraces = []
        allunids = []
        for fid in sdt.fid.unique():

            date = sdt[sdt.fid==fid].date.unique()[0]
            fno = sdt[sdt.fid==fid].fno.unique()[0]
            path = self.db[(self.db.Date==date) & (self.db.FNO==fno)].Path.unique()[0]
            planes = sdt[sdt.fid == fid].plane.unique()
            fpath = os.path.join(path, date, 'fish{}'.format(fno))

            for plane in planes:

                nids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].nid
                unids = sdt[(sdt.fid == fid) & (sdt.plane == plane)].unid
                traces = np.load(os.path.join(fpath, 'plane{}'.format(plane), 'F.npy'))[nids.values.astype(int)]
                tmin, tmax = traces.min(axis=1).reshape(-1, 1), traces.max(axis=1).reshape(-1, 1)
                traces = (traces - tmin) / (tmax - tmin)
                traces[np.isnan(traces)] = 0.

                alltraces.append(traces)
                allunids.append(unids)

        return alltraces, allunids


def create_roistack(

        zstack,
        micron_scaling,
        roi_df_t,
        unids_t,
        sd,
        scoring='bpi'

):
    roistack = np.zeros(shape=(zstack.shape[0], zstack.shape[1], zstack.shape[2])) * np.nan

    zyx_t = roi_df_t.values[:, :3][:, ::-1].astype(float)
    lam_t = roi_df_t.lam.values

    for i in range(3):
        zyx_t[:, i] *= (1. / micron_scaling[i])

    for i in range(3):
        shape_bool = zyx_t[:, i] < roistack.shape[i]
        zyx_t = zyx_t[shape_bool]
        lam_t = lam_t[shape_bool]
        unids_t = unids_t[shape_bool]

    zyx_t = np.round(zyx_t, 0).astype(int)
    if scoring == 'bpi':
        # BPI coloring
        scores = sd.bpi

    elif scoring == 'mcont':

        scores = (sd['m_True_60.0_1.6'] + sd['m_False_60.0_1.6']) / 2

    elif scoring == 'mbout':

        scores = (sd['m_True_1.5_1.6'] + sd['m_False_1.5_1.6']) / 2

    elif scoring == 'mbl':

        scores = sd['m_bl']
    scores_t = scores.values[unids_t]

    scores_t[np.isnan(scores_t)] = 0
    scores_t[np.isinf(scores_t)] = 0
    stackcolor = scores_t.reshape(-1)

    roistack[zyx_t[:, 0], zyx_t[:, 1], zyx_t[:, 2]] = stackcolor
    return roistack


def plot_dv_slices(

        zstack,
        axes,
        roistack,
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.8,
        dims=(3, 3),
        slicerange=None,
        mask=None,
        fs=(15, 15)

):

    if slicerange is None:
        slicerange = np.array([92, 202])

    if dims == (1, 1):

        axes = [axes]
        slices = np.array(slicerange)

    else:

        axes = axes.reshape(-1)
        slices = np.linspace(slicerange[0], slicerange[1], axes.shape[0] + 1)

    for ax, sl in zip(axes, range(slices.shape[0])):

        start, stop = int(round(slices[sl])), int(round(slices[sl + 1]))
        print(start, stop)
        maxz = np.nanmax(zstack[start: stop]).max()
        rslc = np.nanmax(roistack[start: stop], 0)

        ax.imshow(np.nanmax(zstack[start: stop], 0), cmap='Greys', origin='lower', alpha=1, clim=(50, 400), interpolation='none')

        mp = ax.imshow(rslc, cmap=cmap, clim=clim, origin='lower', alpha=roialpha, interpolation='none')
        if mask is not None:
            ax.imshow(mask, alpha=1, cmap='inferno', clim=(0, 150), origin='lower', interpolation='none')

        ax.set_axis_off()
        #         ax.set_xlim(75, 650)
        #         ax.set_ylim(650, 75)
        ax.set_xlim(150, 575)
        ax.set_ylim(575, 150)
        if not dims==(1, 1):
            ax.text(180, 180, '{} μm'.format(start * 2 + 40), fontdict={
                'size': 12})  # 40 is start of skin at average brain 21 dpf, *2 because of depth micron scaling
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    return mp


def plot_dv_slices_scatter(

        zstack,
        sd,
        cval='bpi',
        clim=(-1, 1),
        cmap='coolwarm',
        roialpha=.5

):
    fig, axes = plt.subplots(3, 3, figsize=(5, 5), dpi=300)
    axes = axes.reshape(-1)
    slices = np.linspace(92, 202, axes.shape[0] + 1)

    if isinstance(cval, str):

        cval = sd[cval]
    else:

        cval = cval

    for ax, sl in zip(axes, range(slices.shape[0])):
        start, stop = int(round(slices[sl])), int(round(slices[sl + 1]))
        rois = sd[(start < sd.z) & (sd.z < stop)]
        colorvals = cval[(start < sd.z) & (sd.z < stop)]

        print(start, stop)
        maxz = np.nanmax(zstack[start: stop]).max()

        ax.imshow(np.nanmax(zstack[start: stop], 0), cmap='Greys', origin='lower', alpha=1, clim=(0, 500))
        print(cval.shape, rois.shape)
        ax.scatter(rois.x, rois.y, s=.2, c=colorvals, cmap=cmap, vmin=clim[0], vmax=clim[1], alpha=roialpha)

        ax.set_axis_off()
        ax.set_xlim(50, 550)
        ax.set_ylim(600, 150)
        ax.text(180, 180, '{} μm'.format(start * 2 + 40), fontdict={
            'size': 12})  # 40 is start of skin at average brain 21 dpf, *2 because of depth micron scaling
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()
    return


def plot_orthogonal_views(

        zstack,
        roistack,
        zyx_t,
        maxz=True,
        roialpha=.5,
        cmap='coolwarm',
        clim=(-1, 1)

):
    fig = plt.figure(constrained_layout=True, figsize=(15, 15), dpi=300)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=(1, zstack.shape[0] / zstack.shape[1]),
                             height_ratios=(zstack.shape[0] / zstack.shape[1], 1))

    ax1 = fig.add_subplot(spec[0, 0])
    ax3 = fig.add_subplot(spec[1, 0], sharex=ax1)
    ax4 = fig.add_subplot(spec[1, 1], sharey=ax3)

    start, stop = zyx_t[:, 0].min(), zyx_t[:, 0].max()
    print(start, stop)

    for axno, ax in zip([0, 1, 2], [ax3, ax1, ax4]):

        if maxz:

            rslc = np.nanmax(roistack[:], axno)

        else:

            rslc = np.nanmean(roistack[:], axno)

        if axno == 0:

            av = np.nanmax(zstack[120: 180], axno)

        elif axno == 1:

            start, stop = zyx_t[:, 1].min(), zyx_t[:, 1].max()
            av = np.nanmax(zstack[:, start:stop, :], axno)

        elif axno == 2:

            start, stop = zyx_t[:, 2].min(), zyx_t[:, 2].max()
            av = np.nanmax(zstack[:, :, start:stop], axno)

            av = av.T
            rslc = rslc.T

        ax.imshow(av, cmap='Greys', origin='lower', alpha=1, aspect='auto', clim=(0, 500))
        ax.imshow(rslc, cmap=cmap, clim=clim, origin='lower', alpha=roialpha, aspect='auto')

        ax.set_axis_off()

    ax3.set_xlim(75, 650)
    ax3.set_ylim(650, 75)
    ax4.set_xlim(zstack.shape[0], 0)
    ax1.set_ylim(zstack.shape[0], 0)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def calc_matrix_nt(sumdict, nc=9, norm_sum=True):

    X = sumdict[sorted([i for i in sumdict.keys() if i.startswith('m_False') or i.startswith('m_True')])].values

    if norm_sum:
        X = normalize_sum(X)

    tn = X[sumdict.peak_cw < sumdict.peak_ccw][:, :nc].copy()
    nt = X[sumdict.peak_cw < sumdict.peak_ccw][:, nc:].copy()

    X[sumdict.peak_cw < sumdict.peak_ccw][:, :nc] = nt
    X[sumdict.peak_cw < sumdict.peak_ccw][:, nc:] = tn

    return X


def normalize_sum(X):
    X_min = X.min(axis=1)[:, np.newaxis]
    X = X - X_min
    X_sum = X.sum(axis=1)[:, np.newaxis]
    X = X / X_sum

    return X


def calc_bpi(X):

    bout_r = np.mean(X[:, :5] + X[:, 9:14], axis=1)
    cont_r = np.mean(X[:, 5:9] + X[:, 14:], axis=1)
    bpi = (bout_r - cont_r) / (bout_r + cont_r)
    return bpi


def thresh_rs(

        sd_t,
        fixed_thresh=None,
        std_thresh=1
):
    if fixed_thresh is None:
        # Response score threshold per fish
        tvals = np.zeros(sd_t.fid.max())
        for fid in sd_t.fid.unique():
            tvals[fid - 1] = np.nanmean(sd_t[sd_t.fid == fid].rs_mean.values) + np.nanstd(
                sd_t[sd_t.fid == fid].rs_mean.values) * std_thresh

        rs_thresh = sd_t.rs_mean > tvals[sd_t.fid.values - 1]

    else:

        rs_thresh = sd_t.rs_mean > fixed_thresh

    return rs_thresh, sd_t[rs_thresh]


def scatter_lowdim(

        lowdims,
        metric,
        clim,
        cmap='coolwarm',
        dims=(0, 1),
        save=False,
        alpha=.5,
        size=3,
        tag=''

):
    for ino, i in enumerate(lowdims):

        fig, ax = plt.subplots(figsize=(1, 1), dpi=200)
        ax.scatter(i[:, dims[0]], i[:, dims[1]],
                   c=metric, cmap=cmap, vmin=clim[0], vmax=clim[1], s=size, lw=0, alpha=alpha, edgecolors='none')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect(1)

        if save:
            plt.savefig('nt_metric_lowdim_{}.png'.format(ino, tag), bbox_inches='tight')

        plt.show()


def scatter_lowdim_metric(

        ax,
        lowdim,
        metric,
        clim,
        cmap='coolwarm',
        dims=(0, 1),
        save=False,
        alpha=.5,
        size=3,
        tag=''

):

    ax.scatter(lowdim[:, dims[0]], lowdim[:, dims[1]],
               c='grey', s=.5, lw=0, alpha=alpha, edgecolors='none')
    ax.scatter(lowdim[:, dims[0]][metric], lowdim[:, dims[1]][metric],
               c='red', s=1, lw=0, alpha=alpha, edgecolors='none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1)
    return


def plot_pca_variance(pca, n_components=18):

    fig, ax = plt.subplots(figsize=(2, 2), dpi=200)
    var = ax.plot(range(1, n_components * 2 + 1), pca.explained_variance_ratio_, marker='o', markersize=2)
    varcumsum = ax.plot(range(1, n_components * 2 + 1), pca.explained_variance_ratio_.cumsum(), marker='o', markersize=2)
    ax.set_title('PCA Explained variance')
    ax.set_xticks(range(1, n_components * 2 + 1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.legend(var, varcumsum, labels=['per component', 'cumulative'])
    plt.tight_layout()
    plt.show()

    colors = ['green', 'red', 'blue']
    fig, ax = plt.subplots(1, 3, figsize=(6, 2), dpi=200)
    for i in range(3):
        ax[i].plot(range(1, n_components * 2 + 1), pca.components_[i], lw=1, c=colors[i], marker='o', linestyle='-', markersize=2)
        ax[i].set_xticks(range(1, n_components * 2 + 1))
    plt.tight_layout()
    plt.show()


def get_traces(df, impath='E:/Johannes Kappel/Imaging data/Theia'):
    alltraces = []
    for fish in df.fid.unique():

        dft = df[df.fid == fish]
        date, fno = dft.date.unique()[0], dft.fno.unique()[0]
        for plane in sorted(dft.plane.unique()):
            dfp = dft[dft.plane == plane]
            traces = np.load(os.path.join(impath, date, 'fish{}'.format(fno), 'plane{}'.format(plane), 'F.npy'))
            traces[np.isnan(traces)] = 0
            traces[np.isinf(traces)] = 0
            tmin, tmax = traces.min(axis=1).reshape(-1, 1), traces.max(axis=1).reshape(-1, 1)
            traces = (traces - tmin) / (tmax - tmin)
            alltraces.extend([i for i in traces[dfp.nid.values.astype(int)]])
    return alltraces


def plot_matrix(

        X,
        unique_p,
        save=False

):

    # Sorted matrix after bout-like vs. continuous
    bpi = calc_bpi(X)
    bpi_sort = bpi.argsort()[::-1]
    X_sorted = X[bpi_sort]

    matrices = [X[:, :9] + X[:, 9:], X_sorted[:, :9] + X_sorted[:, :9]]

    for bno, barcode in enumerate(matrices):

        bc_mean = barcode.mean()
        bc_std = np.std(barcode)

        fig, ax = plt.subplots(figsize=(20, 6))
        clr = ax.imshow(
            barcode.T,
            aspect='auto',
            cmap=cc.cm.CET_L3,
            clim=(0, 1.),
            #interpolation='none'
        )
        ax.set_yticks(range(9))
        ax.set_yticklabels(
            ['{0} Hz, {1} mm'.format(i.split('_')[1], int(float(i.split('_')[2]) * 2.5)) for i in unique_p[:9]]
            , rotation=46, ha='right')

        plt.colorbar(clr)
        if save:
            plt.savefig('Barcode_vectors_{}.png'.format(bno), bbox_inches='tight')
        plt.show()


def project_lowdim(

        X,
        n_neighbors=12,
        min_dist=0.03,

):

    pca = PCA(whiten=False, n_components=18)
    X_pca = pca.fit_transform(X)

    umap_mag = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, metric='euclidean')
    X_umap = umap_mag.fit_transform(X)

    return pca, X_pca, umap_mag, X_umap


def project_lowdim(

        X,
        n_neighbors=12,
        min_dist=0.03,

):

    pca = PCA(whiten=False, n_components=18)
    X_pca = pca.fit_transform(X)

    umap_mag = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, metric='euclidean')
    X_umap = umap_mag.fit_transform(X)

    return pca, X_pca, umap_mag, X_umap


def plot_tuning_cluster_behavior(
        X_cluster,
        save=False,
        n_components=9,
        tag=''
):
    green = '#00cd09'
    magenta = '#cd00c4'
    red = '#c70039'

    behavior_tuning = pd.read_csv(
        'C:/Users/jkappel/Downloads/ForJJ.csv')  # can also be found in J:/Johannes Kappel/Stimuli data
    attraction_fs = 1 / behavior_tuning['xax'][[7, 5, 4, 3, 2]]
    attraction = behavior_tuning['si.2'][[7, 5, 4, 3, 2]]
    attraction_er = behavior_tuning['error'][[7, 5, 4, 3, 2]]

    freqs = np.array([0.75, 1.5, 1.5, 1.5, 3., 6., 60., 60., 60.])
    medsize = np.array([0, 2, 4, 5, 7])
    dsize = np.array([1, 3, 6, 8])

    print(X_cluster.shape)
    tuning = np.mean(X_cluster, axis=0)
    bidituning = (tuning[:n_components] + tuning[n_components:]) / 2
    std = np.std(X_cluster, axis=0)
    bidistd = (std[:n_components] + std[n_components:]) / 2

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3, 7), dpi=300)

    # ax[0].plot(attraction_fs, attraction, marker='o', markersize=5, color='black')
    # ax[0].fill_between(attraction_fs,
    #                    attraction + attraction_er,
    #                    attraction - attraction_er,
    #                    facecolor='grey',
    #                    alpha=0.2)

    ax0 = ax[0]#.twinx()
    ax0.plot(freqs[medsize], bidituning[medsize], color=red, marker='o', markersize=5)
    ax0.scatter(freqs[dsize],
                bidituning[dsize],
                color=red,
                marker='o',
                edgecolors='none',
                s=[i ** 2 for i in [2.5, 10, 2.5, 10]],
                alpha=.5)
    ax0.fill_between(freqs[medsize],
                     bidituning[medsize] + bidistd[medsize],
                     bidituning[medsize] - bidistd[medsize],
                     facecolor=red,
                     alpha=0.1)

    ax[1].plot(freqs[medsize], tuning[medsize], color=green, marker='o', markersize=5)
    ax[1].scatter(freqs[dsize],
                  tuning[dsize],
                  color=green,
                  marker='o',
                  edgecolors='none',
                  s=[i ** 2 for i in [2.5, 10, 2.5, 10]],
                  alpha=.5)
    ax[1].fill_between(freqs[medsize],
                       tuning[medsize] + std[medsize],
                       tuning[medsize] - std[medsize],
                       facecolor=green,
                       alpha=0.1)
    ax[2].plot(freqs[medsize], tuning[medsize + n_components], color=magenta, marker='o', markersize=5)
    ax[2].scatter(freqs[dsize],
                  tuning[dsize + n_components],
                  color=magenta,
                  marker='o',
                  edgecolors='none',
                  s=[i ** 2 for i in [2.5, 10, 2.5, 10]],
                  alpha=.5)
    ax[2].fill_between(freqs[medsize],
                       tuning[medsize + n_components] + std[medsize + n_components],
                       tuning[medsize + n_components] - std[medsize + n_components],
                       facecolor=magenta,
                       alpha=0.1)

    ax0.set_ylabel('dF/F')
    ax[1].set_ylabel('dF/F n-t', color=green)
    ax[2].set_ylabel('dF/F t-n', color=magenta)
    #ax[0].set_ylabel('Shoaling index')
    plt.setp(ax, xscale='log')

    plt.setp(ax0,
             xticks=[0.75, 1.5, 3., 6., 60.],
             xticklabels=[0.75, 1.5, 3., 6., 60.],
             xlim=(0.65, 70),
             ylim=[-.3, 1.6]
             )
    plt.setp(ax[1:],
             xticks=[0.75, 1.5, 3., 6., 60.],
             xticklabels=[0.75, 1.5, 3., 6., 60.],
             xlim=(0.65, 70),
             yticks=np.arange(0., 0.7, 0.2),
             ylim=[-.01, 0.9],

             )

    plt.xlabel('Bout frequency')
    if save:
        plt.savefig('tuning_cluster_{}.png'.format(tag), bbox_inches='tight', transparent=True)

    plt.show()


def plot_bpn_tuning(

        X_cluster,
        save=False,
        n_components=9,
        tag='',
        fs=(5, 5)
):
    red = '#c70039'

    freqs = np.array([0.75, 1.5, 1.5, 1.5, 3., 6., 60., 60., 60.])
    medsize = np.array([0, 2, 4, 5, 7])
    dsize = np.array([1, 3, 6, 8])

    tuning = np.mean(X_cluster, axis=0)
    bidituning = (tuning[:n_components] + tuning[n_components:]) / 2
    std = np.std(X_cluster, axis=0)
    bidistd = (std[:n_components] + std[n_components:]) / 2

    fig, ax = plt.subplots(figsize=fs)

    ax.plot(freqs[medsize], bidituning[medsize], color=red, marker='o', markersize=5)
    ax.scatter(freqs[dsize],
               bidituning[dsize],
               color=red,
               marker='o',
               edgecolors='none',
               s=[i ** 2 for i in [2.5, 10, 2.5, 10]],
               alpha=.5)
    ax.fill_between(freqs[medsize],
                    bidituning[medsize] + bidistd[medsize],
                    bidituning[medsize] - bidistd[medsize],
                    facecolor=red,
                    alpha=0.1)

    plt.setp(ax,
             xscale='log',
             xticks=[0.75, 1.5, 3., 6., 60.],
             xticklabels=[0.75, 1.5, 3., 6., 60.],
             xlim=(0.65, 70),
             ylim=[-.3, 1.6],
             ylabel='dF/F',
             xlabel='Bout frequency'
             )

    if save:
        plt.savefig('tuning_cluster_{}.png'.format(tag), bbox_inches='tight', transparent=True)

    plt.show()


if __name__ == "__main__":

    kwargs_inbal = {
        'db_stim': 'dot_wf',
        'age': (6, 99),
        'multirec': True,
        'reganalysis': True,
        'anatomy': True,
        'ds_fac': 5,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '_test_null',
        'regexzero': True,
        'roitags': ['_hcr', '_scaled_hcrref']
    }

    kwargs_inbal_cyt = {
        'db_stim': 'dot_wf_cyt',
        'age': (6, 99),
        'multirec': True,
        'reganalysis': True,
        'anatomy': False,
        'ds_fac': 1,
        'read_frate': True,
        'micdelay': 5,
        'norm_traces': True,
        'foldertag': 'suite2p/',
        'filetag': '_2023118',
        'regexzero': True,
        'roitags': ['_ref']
    }
    kwargs = {
        'db_stim': 'dot_wf_iso',
        'age': (6, 99),
        'multirec': False,
        'reganalysis': True,
        'anatomy': False,
        'ds_fac': 1,
        'read_frate': True,
        'micdelay': 5,
        'norm_traces': False,
        'foldertag': 'plane0/suite2p/',
        'filetag': '_raw_20220106',
        'roitags': ['_ref']
    }
    kwargs = {
        'db_stim': 'dot_wf_abl',
        'age': (6, 99),
        'multirec': False,
        'reganalysis': True,
        'anatomy': False,
        'ds_fac': 1,
        'read_frate': True,
        'micdelay': 5,
        'norm_traces': True,
        'foldertag': 'plane0/suite2p/',
        'filetag': '_norm_20220124',
        'roitags': ['_ref'],
        'impath': 'E:/Johannes Kappel/Ablation data/',
        'stimpath': 'E:/Johannes Kappel/Stimuli data/'
    }


    kwargs_main = {
        'db_stim': 'bout_cont_18dim',
        'age': (17, 99),
        'multirec': False,
        'reganalysis': True,
        'anatomy': True,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '_norm_20220202_snr',
        'roitags': ['_ref'],
        'impath': 'E:/Johannes Kappel/Imaging data/Theia',
        'stimpath': 'E:/Johannes Kappel/Stimuli data/Theia',
        'zstack': 'E:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd',
        'alpath': 'E:/Johannes Kappel/Alignment'
    }
    drive = 'J'
    kwargs_81cjuv = {
        'db_stim': 'dot_wf_juv',
        'age': (17, 99),
        'multirec': False,
        'reganalysis': True,
        'anatomy': True,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '_norm_20220218',
        'roitags': ['_ref'],
        'impath': '{}:/Johannes Kappel/Imaging data/Theia'.format(drive),
        'stimpath': '{}:/Johannes Kappel/Stimuli data/Theia'.format(drive),
        'zstack': '{}:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd'.format(drive),
        'alpath': '{}:/Johannes Kappel/Alignment'.format(drive)
    }
    kwargs_81clarva = {
        'db_stim': 'dots_grating_loom_nat_abl',
        'age': (0, 9),
        'multirec': False,
        'reganalysis': True,
        'anatomy': True,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '_norm_20220218',
        'roitags': ['_ref'],
        'impath': '{}:/Johannes Kappel/Imaging data/Theia'.format(drive),
        'stimpath': '{}:/Johannes Kappel/Stimuli data/Theia'.format(drive),
        'zstack': '{}:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd'.format(drive),
        'alpath': '{}:/Johannes Kappel/Alignment'.format(drive)
    }
    kwargs_81clarva_b = {
        'db_stim': 'dots_grating_loom_dim_abl',
        'age': (0, 9),
        'multirec': False,
        'reganalysis': True,
        'anatomy': True,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '_norm_20220218',
        'roitags': [''],
        'impath': '{}:/Johannes Kappel/Imaging data/Theia'.format(drive),
        'stimpath': '{}:/Johannes Kappel/Stimuli data/Theia'.format(drive),
        'zstack': '{}:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd'.format(drive),
        'alpath': '{}:/Johannes Kappel/Alignment'.format(drive)
    }

    kwargs_johannes = {
        'db_stim': 'pizza_pth2',
        'age': (17, 99),
        'multirec': False,
        'reganalysis': True,
        'anatomy': False,
        'read_frate': False,
        'micdelay': 0,
        'norm_traces': True,
        'foldertag': '',
        'filetag': '',
        'roitags': ['_ref'],
        'impath': 'J:/Johannes Kappel/Imaging data/Theia',
        'stimpath': 'J:/Johannes Kappel/Stimuli data/Theia',
        'zstack': 'J:/Johannes Kappel/Alignment/21dpf_AVG_H2BGCaMP6s.nrrd',
        'alpath': 'J:/Johannes Kappel/Alignment'
    }

    bpa = BoutPreferenceAnalysis(
            dbfile='J:/_Projects/JJ_Katja/FISH_DATABASE1.xlsx',
            **kwargs_johannes
    )

    bpa.iter_dbase()