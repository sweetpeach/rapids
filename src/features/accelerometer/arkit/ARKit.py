from collections import OrderedDict
from numpy import linalg as LA
from scipy import signal
from scipy import fftpack
import pandas as pd
import numpy as np
import pandas as pd
import datetime
import numpy as np
import pytz
import scipy
import time


class ARKit:

    def __init__(self, acc_data, day_segment, arkit_params):
        # assert stuff here
        self.arkit_time_features = arkit_params['ARKIT_TIME_FEATURES']
        self.arkit_frequency_features = arkit_params['ARKIT_FREQUENCY_FEATURES']
        self.window_size_in_minutes = arkit_params['ARKIT_WINDOW_SIZE']
        self.mode = arkit_params['ARKIT_MODE']
        acc_data = acc_data.rename(columns={'double_values_0':'double_x', 'double_values_1':'double_y', 'double_values_2':'double_z'})
        if day_segment != "daily":
            acc_data = acc_data[acc_data["local_day_segment"] == day_segment]
        if not acc_data.empty:
            # deduplicate data
            arkit_feaures = pd.DataFrame()
            acc_data = acc_data[['timestamp', 'double_x', 'double_y', 'double_z']]
            df_dedups = acc_data[acc_data.timestamp.duplicated(keep=False)].groupby(['timestamp'])[['double_x','double_y','double_z']].mean().reset_index()
            acc_data = acc_data.drop_duplicates(subset=['timestamp'], keep=False)
            acc_data = acc_data.append(df_dedups)
            acc_data = acc_data.sort_values(by=['timestamp']).reset_index(drop=True)
            acc_data = acc_data.reset_index(drop=True)
            # windowize data
            acc_data['time'] = pd.to_datetime(acc_data.timestamp, unit='ms')
            acc_data['window'] = ((acc_data.time - acc_data.time[0]).dt.total_seconds()/ (self.window_size_in_minutes * 60)).sort_values().round(0).astype(int)
            self.acc_data = acc_data

    def run(self):
        return self.acc_data.groupby(['window'])[['double_x', 'double_y', 'double_z', 'timestamp']].apply(self.arkit_featurize)


    """
    modes
    0 - time domain only
    1 - time + frequency domain
    2 - time + frequency + stats
    3 - statistical methods only
    """

    def arkit_featurize(self, df_fw):
        local_dict = OrderedDict()
        if self.mode > 0 and self.mode < 3:
            if df_fw.index.size >= (30*self.window_size_in_minutes):
                df_fw.double_x = df_fw.double_x.replace({0:1e-08})
                df_fw.double_y = df_fw.double_y.replace({0:1e-08})
                df_fw.double_z = df_fw.double_z.replace({0:1e-08})
                f_x = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_x)
                f_y = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_y)
                f_z = scipy.interpolate.interp1d(df_fw.timestamp, df_fw.double_z)
                r = (np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2)).replace({0:1e-08})
                f_r = scipy.interpolate.interp1d(df_fw.timestamp, r)
                xnew = []
                step = (df_fw.timestamp.iloc[-1] - df_fw.timestamp.iloc[0]) /df_fw.index.size
                if int(step)==0:
                    print(df_fw)
                for ti in range(df_fw.timestamp.iloc[0], df_fw.timestamp.iloc[-1], int(step)):
                    xnew.append(ti)

                f_fs = self.window_size_in_minutes * 60 / df_fw.index.size
                L = 512 # change it to 512
                local_dict.update({'skip_fft':False, 'fx': f_x(xnew), 'fy': f_y(xnew), 'fz': f_z(xnew), 'fr': f_r(xnew), 'fs': f_fs, 'L': L})
            else:
                local_dict.update({'skip_fft':True})
            if df_fw.index.size == 0:
                local_dict['skip_td'] = True
            else:
                local_dict['skip_td'] = False
        if self.mode == 0:
            local_dict['skip_fft'] = True
            if df_fw.index.size == 0:
                local_dict['skip_td'] = True
            else:
                local_dict['skip_td'] = False
        if self.mode == 3:
            local_dict['skip_fft'] = True
            local_dict['skip_td'] = True
        feat_dict = {}
        #window information:
        if df_fw.index.size > 0:
            feat_dict.update({'start_timestamp':df_fw.timestamp.iloc[0]})
            feat_dict.update({'end_timestamp':df_fw.timestamp.iloc[0] + 6*10**3})
            feat_dict.update({'sample_count':df_fw.index.size})
        else:
            feat_dict.update({'start_timestamp': np.nan})
            feat_dict.update({'end_timestamp':np.nan})
            feat_dict.update({'sample_count':np.nan})
        for feature in self.arkit_time_features:
            if feature == 'int_desc':
                if not local_dict['skip_td']:
                    int_desc = np.sqrt((df_fw.double_x ** 2).describe() + (df_fw.double_y **2).describe() + (df_fw.double_z ** 2).describe())
                    feat_dict.update({'int_mean': int_desc[1], 'int_std': int_desc[2],
                                      'int_min': int_desc[3],'int_25': int_desc[4], 'int_50': int_desc[5],'int_75': int_desc[6]})
                else:
                    feat_dict.update({'int_mean': np.nan, 'int_std': np.nan,
                                      'int_min': np.nan,'int_25': np.nan, 'int_50': np.nan,'int_75': np.nan})
            elif feature == 'int_rms':
                if not local_dict['skip_td']:
                    int_rms = np.sqrt((df_fw.double_x**2).sum() + (df_fw.double_y**2).sum() + (df_fw.double_z**2).sum()) / np.sqrt(df_fw.index.size)
                    feat_dict.update({'int_rms':int_rms})
                else:
                    feat_dict.update({'int_rms': np.nan})
            elif feature == 'mag_desc':
                if not local_dict['skip_td']:
                    mag_desc = np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2).describe()
                    feat_dict.update({'mag_mean': mag_desc[1], 'mag_std': mag_desc[2], 'mag_min': mag_desc[3],
                                      'mag_25': mag_desc[4], 'mag_50': mag_desc[5],'mag_75': mag_desc[6]})
                else:
                    feat_dict.update({'mag_mean': np.nan, 'mag_std': np.nan, 'mag_min': np.nan,
                      'mag_25': np.nan, 'mag_50': np.nan,'mag_75': np.nan})
            elif feature == 'pear_coef':
                if not local_dict['skip_td']:
                    cov_matrix =  np.cov(np.stack((df_fw.double_x,df_fw.double_y, df_fw.double_z), axis=0))
                    pear_coef_xy = cov_matrix[0,1] / (df_fw.double_x.std() * df_fw.double_y.std())
                    pear_coef_yz = cov_matrix[1,2] / (df_fw.double_y.std() * df_fw.double_z.std())
                    pear_coef_xz = cov_matrix[0,2] / (df_fw.double_x.std() * df_fw.double_z.std())
                    feat_dict.update({'pear_coef_xy':pear_coef_xy, 'pear_coef_yz':pear_coef_yz,'pear_coef_xz':pear_coef_xz })
                else:
                    feat_dict.update({'pear_coef_xy':np.nan, 'pear_coef_yz':np.nan,'pear_coef_xz':np.nan})
            elif feature == 'sma':
                if not local_dict['skip_td']:
                    sma = (np.abs(df_fw.double_x.to_numpy()).sum() + np.abs(df_fw.double_y.to_numpy()).sum() + np.abs(df_fw.double_z.to_numpy()).sum()) / df_fw.index.size
                    feat_dict.update({'sma':sma})
                else:
                    feat_dict.update({'sma':np.nan})
            elif feature == 'svm':
                if not local_dict['skip_td']:
                    svm = np.sqrt(df_fw.double_x**2 + df_fw.double_y**2 + df_fw.double_z**2).sum() / df_fw.index.size
                    feat_dict.update({'svm':svm})
                else:
                    feat_dict.update({'svm':np.nan})
        for feature in self.arkit_frequency_features:
            if feature == 'fft':
                if not local_dict['skip_fft']:
                    L = local_dict['L']
                    dfx = fftpack.fft(local_dict['fx'], 512)
                    dfy = fftpack.fft(local_dict['fy'], 512)
                    dfz = fftpack.fft(local_dict['fz'], 512)
                    dfr = fftpack.fft(local_dict['fr'], 512)
                    # DC component
                    # Remove the L part!
                    feat_dict.update({'fdc_x': np.mean(np.real(dfx)), 'fdc_y': np.mean(np.real(dfy)),
                                      'fdc_z':  np.mean(np.real(dfz)), 'fdc_r':  np.mean(np.real(dfr))})
                    # Energy
                    feat_dict.update({'feng_x': (np.sum(np.real(dfx)**2 + np.imag(dfx)**2)) / L, 'feng_y': (np.sum(np.real(dfy)**2 + np.imag(dfy)**2)) / L,
                                      'feng_z':  (np.sum(np.real(dfz)**2 + np.imag(dfz)**2)) / L, 'feng_r':  (np.sum(np.real(dfr)**2 + np.imag(dfr)**2)) / L})
                    # Entropy
                    ck_x = np.sqrt(np.real(dfx)**2  + np.imag(dfx)**2)
                    cj_x = ck_x / np.sum(ck_x)
                    e_x = np.sum(cj_x * np.log(cj_x))

                    ck_y = np.sqrt(np.real(dfy)**2  + np.imag(dfy)**2)
                    cj_y = ck_y / np.sum(ck_y)
                    e_y = np.sum(cj_y * np.log(cj_y))

                    ck_z = np.sqrt(np.real(dfz)**2  + np.imag(dfz)**2)
                    cj_z = ck_z / np.sum(ck_z)
                    e_z = np.sum(cj_z * np.log(cj_z))

                    ck_r = np.sqrt(np.real(dfr)**2  + np.imag(dfr)**2)
                    cj_r = ck_r / np.sum(ck_r)
                    e_r = np.sum(cj_r * np.log(cj_r))

                    feat_dict.update({'fent_x': e_x, 'fent_y':  e_y,'fent_z':  e_z, 'fent_r': e_r})

                    # Correlation
                    # Fix the length, should be FFT wndow size 512

                    fcorr_xy = np.dot(np.real(dfx) / L, np.real(dfy) / L)
                    fcorr_xz = np.dot(np.real(dfx) / L, np.real(dfz) / L)
                    fcorr_yz = np.dot(np.real(dfy) / L, np.real(dfz) / L)

                    feat_dict.update({'fcorr_xy': fcorr_xy,'fcorr_xz':  fcorr_xz, 'fcorr_yz': fcorr_yz})

                else:
                    feat_dict.update({'fdc_x': np.nan, 'fdc_y':  np.nan,'fdc_z':  np.nan, 'fdc_r': np.nan})
                    feat_dict.update({'feng_x':  np.nan, 'feng_y':  np.nan, 'feng_z':   np.nan, 'feng_r':   np.nan})
                    feat_dict.update({'fent_x': np.nan, 'fent_y':  np.nan,'fent_z':  np.nan, 'fent_r': np.nan})
                    feat_dict.update({'fcorr_xy': np.nan,'fcorr_xz':  np.nan, 'fcorr_yz': np.nan})
            elif feature == 'psd':
                if not local_dict['skip_fft']:
                    fs = local_dict['fs']
                    psd_window = signal.get_window('boxcar', len(local_dict['fx'])) # do not pass this window
                    freqs_x, pxx_denx = signal.periodogram(local_dict['fx'], window=psd_window, fs=fs)
                    freqs_y, pxx_deny = signal.periodogram(local_dict['fy'], window=psd_window, fs=fs)
                    freqs_z, pxx_denz = signal.periodogram(local_dict['fz'], window=psd_window, fs=fs)
                    freqs_r, pxx_denr = signal.periodogram(local_dict['fr'], window=psd_window, fs=fs)
                    feat_dict.update({'psd_mean_x': np.mean(pxx_denx), 'psd_mean_y': np.mean(pxx_deny),
                                      'psd_mean_z': np.mean(pxx_denz), 'psd_mean_r': np.mean(pxx_denr)})

                    feat_dict.update({'psd_max_x': np.max(pxx_denx),
                                      'psd_max_y': np.max(pxx_deny),
                                      'psd_max_z': np.max(pxx_denz),
                                      'psd_max_r': np.max(pxx_denr)})


                    freqs_05_3_x = np.argwhere((freqs_x >= 0.5) & (freqs_x <= 3))
                    freqs_05_3_y = np.argwhere((freqs_y >= 0.5) & (freqs_y <= 3))
                    freqs_05_3_z = np.argwhere((freqs_z >= 0.5) & (freqs_z <= 3))
                    freqs_05_3_r = np.argwhere((freqs_r >= 0.5) & (freqs_r <= 3))


                    # max b/w 0.3 - 3Hz
                    # 0.5 - 3 Hz if missing, maybe not 0.0
                    feat_dict.update({'psd_max_x_05_3': np.max(pxx_denx[freqs_05_3_x]) if freqs_05_3_x.any() else 0.0,
                      'psd_max_y_05_3': np.max(pxx_deny[freqs_05_3_y]) if freqs_05_3_y.any() else 0.0,
                      'psd_max_z_05_3': np.max(pxx_denz[freqs_05_3_z]) if freqs_05_3_z.any() else 0.0,
                      'psd_max_r_05_3': np.max(pxx_denr[freqs_05_3_r]) if freqs_05_3_r.any() else 0.0})
                else:
                    feat_dict.update({'psd_mean_x': np.nan, 'psd_mean_y':np.nan,
                                      'psd_mean_z': np.nan, 'psd_mean_r': np.nan})
                    feat_dict.update({'psd_max_x': np.nan,
                                      'psd_max_y': np.nan,
                                      'psd_max_z': np.nan,
                                      'psd_max_r': np.nan})
                    feat_dict.update({'psd_max_x_05_3': np.nan,
                      'psd_max_y_05_3': np.nan,
                      'psd_max_z_05_3': np.nan,
                      'psd_max_r_05_3': np.nan})
            elif feature == 'lmbs':
                if not local_dict['skip_td']:
                    lmb_f_05_3 = np.linspace(0.5, 3, 100)
                    lmb_psd_x = signal.lombscargle(df_fw.timestamp, df_fw.double_x, lmb_f_05_3, normalize=False)
                    lmb_psd_y = signal.lombscargle(df_fw.timestamp, df_fw.double_y, lmb_f_05_3, normalize=False)
                    lmb_psd_z = signal.lombscargle(df_fw.timestamp, df_fw.double_z, lmb_f_05_3, normalize=False)

                    feat_dict.update({'lmb_psd_max_x_05_3': np.max(lmb_psd_x) if lmb_psd_x.any() else 0.0,
                      'lmb_psd_max_y_05_3': np.max(lmb_psd_y) if lmb_psd_y.any() else 0.0,
                      'lmb_psd_max_z_05_3': np.max(lmb_psd_z) if lmb_psd_z.any() else 0.0})
                else:
                    feat_dict.update({'lmb_psd_max_x_05_3': np.nan,
                      'lmb_psd_max_y_05_3': np.nan,
                      'lmb_psd_max_z_05_3': np.nan})
        return pd.Series(feat_dict)
