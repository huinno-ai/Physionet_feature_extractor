# @coded by sichon
import os

from ecgdetectors import Detectors
import neurokit2 as nk
import matplotlib.pyplot as plt
from biosppy import storage
from biosppy.signals import ecg
import numpy as np
import antropy as ant
import pandas as pd
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features


def between(a,b,c):
    if a <= b and b < c:
        return 0
    if b < a:
        return -1
    if b >=c:
        return 1

def Find_between_idx(lst,ref_idx,s,e,finddepth):
    if ref_idx < 0 or ref_idx > len(lst) or finddepth<0:
        return -1
    if lst [ ref_idx ] >= s and lst [ref_idx] < e :
        return ref_idx
    elif lst [ ref_idx ] >= e:
        return Find_between_idx(lst,ref_idx-1, s,e, finddepth-1)
    elif lst [ ref_idx ] < s:
        return Find_between_idx(lst, ref_idx +1 , s, e, finddepth - 1)


def extract_interval(signal0,rpeaks,peak1,peak2, type = 0):
    #type == 0 >> 순방향 segment r p1 p2 r p1 p2 r
    #type == 1 >> r-idx 가 차이가 있을경우  p1 r p2  p1 r p2

    len_r = len(rpeaks)
    Qualify_p1 = np.zeros(len_r,dtype=int)
    Qualify_p2 = np.zeros(len_r,dtype=int)
    lst_p1p2_interval = np.zeros(len_r, dtype=int)

    # case p1 - r - p1 >> remove first p1 to make r -p1 - r
    peak1_ = peak1
    if peak1[0] < rpeaks[0] :
        peak1_ = peak1[1:]

    # case p2 - r - p2 >> remove first p2 to make r -p2 - r
    peak2_ = peak2
    if peak2[0] < rpeaks[0]:
        peak2_ = peak2[1:]
    # case  r - p1 - r

    j = 0
    k = 0
    # signal Qualify feature ( Checking p-r-p )
    for i  in range(len_r-1):
        rp = rpeaks[i]
        rp_next= rpeaks[i+1]
        p1 = peak1_[j]
        b = between(rp,p1,rp_next)
        if b == 0:
            Qualify_p1[i] = 1
            j+=1
        elif b == 1:
            Qualify_p1[i] = 0
        elif b == -1:
            if i is not 0:
                Qualify_p1[i-1] = 2
            Qualify_p1[i] = 0
            i -= 1
            j += 1
    for i in range(len_r-1):
        rp = rpeaks[i]
        rp_next = rpeaks[i + 1]
        p2 = peak2_[k]
        b2 = between(rp, p2, rp_next)
        if b2 == 0:
            Qualify_p2[i] = 1
            k += 1
        elif b2 == 1:
            Qualify_p2[i] = 0
        elif b2 == -1:
            if i is not 0:
                Qualify_p2[i - 1] = 2
            Qualify_p2[i] = 0
            i -= 1
            k += 1
        #case p1 - r - p1
    #if type
    if type == 0:
        p1_idx=0
        p2_idx=0
        for i in range(len_r-1):
            rp = rpeaks[i]
            rp_next = rpeaks[i + 1]
            p1_idx_next = Find_between_idx(peak1_,p1_idx, rp, rp_next, 3)
            p2_idx_next = Find_between_idx(peak2_, p2_idx, rp, rp_next, 3)

            p1p2_interval = peak2_[p2_idx_next] - peak1_[p1_idx_next]
            p1_idx = p1_idx_next
            p2_idx = p2_idx_next
            lst_p1p2_interval[i] = p1p2_interval
    if type == 1:
        p1_idx = 0
        p2_idx = 1
        for i in range(len_r-2):
            rp = rpeaks[i]
            rp_next = rpeaks[i + 1]
            rp_nextnext = rpeaks[i + 2]

            p1_idx_next = Find_between_idx(peak1_, p1_idx, rp, rp_next, 3)
            p2_idx_next = Find_between_idx(peak2_, p2_idx, rp_next, rp_nextnext, 3)

            p1p2_interval = peak2_[p2_idx_next] - peak1_[p1_idx_next]
            p1_idx = p1_idx_next
            p2_idx = p2_idx_next
            lst_p1p2_interval[i] = p1p2_interval
    return lst_p1p2_interval, Qualify_p1 , Qualify_p2

def extract_Amplitude(signal0,peaks):
    # we assume isoElectrical line is y=0
    l = len (peaks)
    lstamp = np.zeros(l, dtype=float)
    for i,peak in enumerate(peaks):
        lstamp[i] = signal0[peak]
    return lstamp

def extract_Area(signal0,peaks,threshold):
    # we assume isoElectrical line is y=0
    l = len(peaks)
    lstarea = np.zeros(l, dtype=float)
    for idx,peak in enumerate(peaks):
        iso_pre = peak-threshold
        iso_post = peak+threshold
        for i in range(threshold+1):
            if peak - i < 0 :
               iso_pre = 0
               break
            if signal0[peak-i] *signal0[peak] <= 0:
                iso_pre = peak-i
                break
        for i in range(threshold+1):
            if peak + i >= len(signal0):
                iso_post= peak+i-1
                break
            if signal0[peak+i] * signal0[peak] <= 0:
                iso_pre = peak+i
                break
        lstarea[idx] = np.sum(np.abs(signal0[iso_pre:iso_post]))
    return lstarea

def extract_AreaQRS(signal0,peaksR,peaksQ,peaksS):
    # we assume isoElectrical line is y=0
    l = len(peaksR)
    p1_idx = 0
    p2_idx = 1
    lstarea = np.zeros(l, dtype=float)
    for i in range(l - 2):
        rp = peaksR[i]
        rp_next = peaksR[i + 1]
        rp_nextnext = peaksR[i + 2]
        p1_idx_next = Find_between_idx(peaksQ, p1_idx, rp, rp_next, 3)
        p2_idx_next = Find_between_idx(peaksS, p2_idx, rp_next, rp_nextnext, 3)

        if p1_idx_next == -1 or p2_idx_next == -1:
            lstarea[i] = 0
            p1_idx = i+1
            p2_idx = i+2
            continue
        lstarea[i] = np.sum(np.abs(signal0[peaksQ[p1_idx_next]:peaksS[p2_idx_next]]))
    return lstarea


def rpeaks_to_RRinterval(peaks):
    l = len(peaks)
    # RRinterval = np.zeros(l-1, dtype=float)
    peaks2 = peaks[1:l]
    RRinterval = peaks2 -peaks[0:l-1]
    return RRinterval

def Flat_Concatenate(args):
    lst = []

    for arg in args:
        # print(arg)
        if isinstance(arg, dict):
            for value in arg.values():
                if np.isnan(value) == True:
                    pass
                    # print ('it is nan')
                else:
                    lst  = np.append(lst,value)
        elif len(lst) == 0:
            lst = arg
        else:
            if isinstance(arg,np.ndarray) == False:
                lst  = np.append(lst,value)
            else:
                lst = np.concatenate((lst,arg))
    return lst

def valid_check(lst):
    if isinstance(lst, dict):
        for value in lst.values():
            if np.isnan(value) == True:
                print('it is nan')
                return 0
        return 1
    else :
        if np.isnan(np.sum(lst)) == True:
            return 0
        return 1


def Extract_DerivedFeature(lst,fet_name):
    # (validity median std min max entropy fractal)
    Dervied_feature = np.empty( 9 )
    Dervied_feature[:] = np.nan

    Dervied_feature[1] = valid_check(lst)
    if Dervied_feature[1] == 1 :
        Dervied_feature[2] = np.median(lst)
        Dervied_feature[0] = Dervied_feature[2]
        Dervied_feature[3] = np.std(lst)
        Dervied_feature[4] = np.min(lst)
        Dervied_feature[5] = np.max(lst)

        if len(lst) <= 5:
            Dervied_feature[6] = 'nan'
            Dervied_feature[7] = 'nan'
        else:
            Dervied_feature[6] = ant.perm_entropy(lst, normalize=True)
            Dervied_feature[7] = ant.sample_entropy(lst)

        # Fractal dimension
        Dervied_feature[8] = ant.petrosian_fd(lst)
    else :
        pass
    return Dervied_feature
        # print(lst)
def export_feature(df,np_feature,headerpath):
    idx =  headerpath.rfind('/', 0, len(headerpath))
    idx2 = headerpath.rfind('.', 0, len(headerpath))
    path = headerpath[0:idx]
    filename = headerpath[idx+1:idx2]
    #
    # pull = path
    newpath = headerpath.replace('hea','fet')
    newpath2 = headerpath.replace('hea', 'pdfet')
    # newpath = 'feature/' + filename + '.feat'
    df.to_csv(newpath2)
    np.savetxt(newpath,np_feature,delimiter=',',fmt="%f")
    # np_feature.tofile(newpath,',')
#

def export_matrial(matrials,headerpath):
    # print(matrials)
    newpath = headerpath.replace('hea', 'matrial')
    if os.path.exists(newpath):
        os.remove(newpath)
    with open(newpath, 'a') as f:
        for arg in matrials:
            # with open(newpath, 'w') as f:
            print(arg, file=f)

def Feature_Extractor(recording,rate,strlabel,age,sex,plotoption = False,headerpath='features/1.hat'):

    if plotoption == True:
        fig,axs = plt.subplots(2,1)
        fig.suptitle('lead1,lead2,avf,v1,v2 ' + strlabel)
        axs[0].plot(recording[1])
        axs[1].plot(recording[1])

    idx = [i for i in range(len(recording))]

    detectors = Detectors(rate)
    for i in range(1,2):
        r_peaks = detectors.two_average_detector(recording[idx[i]])
        try:
            rpeaks = ecg.ecg(signal=recording[idx[i]], sampling_rate=500., show=False, )[2]
            # if not rpeaks or  (len(rpeaks)) < 5:
            #     return
            _, waves_peak = nk.ecg_delineate(recording[idx[i]], rpeaks, sampling_rate=500)
        except:
            import sys
            print(sys.exc_info()[0])
            return 0

        if waves_peak['ECG_Q_Peaks'] is not None:
            s = 0
            for val in waves_peak.values():
                 s = s +  sum(val)
            # print(s)
            if np.isnan(s) == True :
                # QRST 에 nan 이 있는경우
                return 0
                pass
            else :
                lstecg_q = recording[idx[i]][waves_peak['ECG_Q_Peaks']]
                lstecg_r = recording[idx[i]][rpeaks]
                lstecg_s = recording[idx[i]][waves_peak['ECG_S_Peaks']]
                lstecg_p = recording[idx[i]][waves_peak['ECG_P_Peaks']]
                lstecg_t = recording[idx[i]][waves_peak['ECG_T_Peaks']]
                ############plot pqrst  ##############
                if plotoption == True:
                    lstecg=[]
                    for r_peak in r_peaks:
                        lstecg.append(recording[i][r_peak])
                    # print(lstecg)
                    axs[i].scatter(r_peaks,lstecg,color = 'r')
                    axs[i].scatter(waves_peak['ECG_Q_Peaks'], lstecg_q, color='r')
                    axs[i].scatter(waves_peak['ECG_S_Peaks'], lstecg_s, color='g')
                    axs[i].scatter(waves_peak['ECG_P_Peaks'], lstecg_p, color='black')
                    axs[i].scatter(waves_peak['ECG_T_Peaks'], lstecg_t, color='y')
                    plt.show()

        # Extract matrials (p,q,r,s,t positions and p,q,r,s,t Amplitude)
        matrials = (waves_peak['ECG_P_Peaks'], waves_peak['ECG_Q_Peaks'], rpeaks, waves_peak['ECG_S_Peaks'],
                    waves_peak['ECG_T_Peaks'],lstecg_p, lstecg_q, lstecg_r, lstecg_s, lstecg_t)
        export_matrial(matrials, headerpath)
        # PR / QRS / ST / QT * cycles extract
        lst_intervalPR, lst_QualifyA1, lst_QualifyP = extract_interval(recording[i] , rpeaks,waves_peak['ECG_P_Peaks'], rpeaks ,type= 1)     #PR
        lst_intervalQRS, lst_QualifyQ, lst_QualifyS = extract_interval(recording[i] , rpeaks,waves_peak['ECG_Q_Peaks'],waves_peak['ECG_S_Peaks'],type= 1 )     #QRS
        lst_intervalST, PQRST_QualifyA3, lst_QualifyT = extract_interval(recording[i], rpeaks,waves_peak['ECG_S_Peaks'],waves_peak['ECG_T_Peaks'],type= 0)     #ST
        lst_intervalQT, PQRST_QualifyA4, PQRST_QualifyB4 = extract_interval(recording[i], rpeaks,waves_peak['ECG_Q_Peaks'],waves_peak['ECG_T_Peaks'],type= 1)     #QT

        # Finding isoelectrical line Start
            #we Assume isoelectrical line zero
            #Must be modified for better method
        # Finding isoelectrical line End

        # P Q R S T amplitude
        lst_PAmp = extract_Amplitude(recording[i], waves_peak['ECG_P_Peaks'])
        lst_QAmp = extract_Amplitude(recording[i], waves_peak['ECG_Q_Peaks'])
        lst_RAmp = extract_Amplitude(recording[i], rpeaks)
        lst_SAmp = extract_Amplitude(recording[i], waves_peak['ECG_S_Peaks'])
        lst_TAmp = extract_Amplitude(recording[i], waves_peak['ECG_T_Peaks'])

        # P area , QRS area , T area
        lst_PArea = extract_Area(recording[i], waves_peak['ECG_P_Peaks'], threshold = int(50 * (rate/ 500)))
        lst_QArea = extract_AreaQRS(recording[i],rpeaks, waves_peak['ECG_Q_Peaks'], waves_peak['ECG_S_Peaks'])
        lst_TArea = extract_Area(recording[i], waves_peak['ECG_T_Peaks'], threshold = int(50 * (rate/ 500)))


        #HRV features @ref https://github.com/Aura-healthcare/hrv-analysis
        rr_intervals_list = rpeaks_to_RRinterval(rpeaks)
            # This remove outliers from signal
        rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals_list,low_rri=300, high_rri=2000)
            # This replace outliers nan values with linear interpolation
        interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,interpolation_method="linear")
            # This remove ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
        time_domain_features = get_time_domain_features(rr_intervals_list)
        # print (time_domain_features)

        # rr ,nn,pr ...  >>  extract corresponding   (validity median std min max entropy fractal)  features
        cols = ['value','validity','median','std','min','max','per_entropy','sam_entorpy','fractal']
        rows = ['rr','nn','pr','qrs','st','qt','pAmp','qAmp','rAmp','sAmp','tAmp','pArea','qrsArea','TArea','HRV_mean_nni','HRV_sdnn','HRV_sdsd','HRV_nni50','HRV_pnni50',\
                'HRV_nni20','HRV_pnni20','HRV_rmssd','HRVmedian_nni','HRVrange_nni','HRVcvsd','HRVcvnni','HRVmean_hr','HRVmax_hr','HRVmin_hr','HRVstd_hr','age','sex','KLT','SQI',\
                 'Lead_Relative']

        #aggregate features to dataframe
        df = pd.DataFrame(index=rows, columns=cols)
        df.iloc[0] = Extract_DerivedFeature(rr_intervals_list,'rr')
        df.iloc[1] = Extract_DerivedFeature(nn_intervals_list,'nn')

        df.iloc[2] = Extract_DerivedFeature(lst_intervalPR,'pr')
        df.iloc[3] = Extract_DerivedFeature(lst_intervalQRS,'qrs')
        df.iloc[4] = Extract_DerivedFeature(lst_intervalST,'st')
        df.iloc[5] = Extract_DerivedFeature(lst_intervalQT,'qt')

        df.iloc[6] = Extract_DerivedFeature(lst_PAmp,'pAmp')
        df.iloc[7] = Extract_DerivedFeature(lst_QAmp, 'qAmp')
        df.iloc[8] = Extract_DerivedFeature(lst_RAmp,'rAmp')
        df.iloc[9] = Extract_DerivedFeature(lst_SAmp,'sAmp')
        df.iloc[10] = Extract_DerivedFeature(lst_TAmp,'tAmp')

        df.iloc[11] = Extract_DerivedFeature(lst_PArea,'pArea')
        df.iloc[12] = Extract_DerivedFeature(lst_QArea,'qrsArea')
        df.iloc[13] = Extract_DerivedFeature(lst_TArea,'TArea')

        ct = 0
        for val in list(time_domain_features.values()):
            df.iloc[14+ct,0] = val
            ct +=1
        df.iloc[14+ct,0] = age
        ct += 1
        df.iloc[14+ct,0] = sex

        #flatten pandas features > n x 1 feature list
        np_feature_flat = Flat_Concatenate(df.iloc[:,:].to_numpy().astype(float))
        export_feature(df,np_feature_flat,headerpath)

        return np_feature_flat
# print(r_peaks)
