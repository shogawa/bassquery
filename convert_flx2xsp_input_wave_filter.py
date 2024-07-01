from joblib import Parallel, delayed
import subprocess
import sys
import time

import numpy as np
from astropy import constants as const


filter_info = {
    'J': [12500, 12375, 12625],
    'H': [16500, 16335, 16665],
    'K': [22000, 21780, 22220],
    'L': [34500, 34165, 34835],
    'M': [47000, 46530, 47470],
    'N': [100000, 99000, 101000],
    'Q': [200000, 198000, 202000],
    '4.5m': [45000, 44550, 45450],
    '5.5m': [55000, 54450, 55550],
    '18m': [180000, 178200, 181800],
    '20m': [200000, 198000, 202000],
    '25m': [250000, 247500, 252500],
    '30m': [300000, 297000, 303000],
    'spec8': [80000, 79200, 80800],
    'spec9': [90000, 89100, 90900],
    'spec9.7': [97000, 96030, 97970],
    'spec10': [100000, 99000, 101000],
    'spec11': [110000, 108900, 111100],
    'spec12': [120000, 118800, 121200],
    'spec12.5': [125000, 123750, 126250],
    'spec12.7': [127000, 125830, 128170],
    'spec12.8': [128000, 126720, 129280],
    'spec20': [200000, 198000, 202000],
    'spec30': [300000, 297000, 303000],
    'spec40': [400000, 396000, 404000],
    'TReCS_Si2': [87147.86, 81340.00, 92770.00],
    'TReCS_N': [99730.24, 74251.36, 133730.00],
    'TReCS_Qa': [183203.38, 172608.48, 195853.05],
    'Michelle_Np': [112385.82, 97878.18, 127916.89],
    'Michelle_Si5': [116708.68, 109000.00, 125000.00],
    'Michelle_Si6': [124701.08, 116286.36, 133500.00],
    'Michelle_Qa': [180762.16, 167951.30, 195285.34],
    'OSCIR_N': [107051.29, 80405.15, 135869.66],
    'OSCIR_IHW18': [180941.52, 169203.33, 194206.92],
    'NICMOS1_F110W': [11233.62, 7848.73, 14335.73],
    'NICMOS2_F160W': [16036.90, 13646.65, 18419.13],
    'NICMOS2_F110W': [11234.92, 7868.98, 14336.38],
    'NICMOS2_F222M': [22175.27, 20498.00, 23990.72],
    'IRAS.25': [230702.01, 160159.57, 310833.33],
    'IRAS.60': [581903.91, 300000.00, 863617.02],
    'IRAS.100': [995377.63, 700000.00, 1400000.0],
    'NACO_H': [16495.13, 14400.00, 18800.00],
    'NACO_J': [12578.34, 10600.00, 14400.00],
    'NACO_Ks': [21365.81, 19000.00, 24200.00],
    'NACO_Lp': [37914.49, 34000.00, 42700.00],
    'NACO_Mp': [47708.58, 43900.00, 52100.00],
    'NACO_IB2.06': [20629.72, 20160.00, 21100.00],
    'NACO_IB2.42': [24309.84, 23840.00, 24780.00],
    'NACO_NB4.05': [40556.88, 39877.80, 41242.20],
    'GTC_CC_Si2': [86566.48, 78968.39, 93992.17],
    'VISIR_PAH1': [85943.73, 81290.31, 91285.69],
    'VISIR_ArIII': [89948.23, 88562.67, 91405.32],
    'VISIR_SIV': [104556.22, 102924.83, 106228.32],
    'VISIR_NeII_1': [122712.31, 121170.00, 124660.13],
    'VISIR_NeII': [127940.96, 125563.60, 130534.64],
    'VISIR_NeII_2': [130454.35, 127994.33, 132853.41],
    'VISIR_PAH2': [112625.02, 105979.79, 120284.24],
    'VISIR_PAH2_2': [117394.14, 114810.00, 120533.70],
    'VISIR_Q1': [177246.68, 167624.26, 187998.49],
    'VISIR_Q2': [187422.58, 177621.42, 196407.23],
    'FORCAST_F315': [313925.39, 275674.77, 404018.88],
    'herschel.pacs.70': [707696.38, 556706.37, 977403.23],
    'herschel.pacs.160': [1.62e+69, 1.18e+6, 2.44e6]
}

def get_filter_info(filter_name):
    if filter_name in filter_info:
        info = filter_info[filter_name]
        return [info[2], info[1]]
    else:
        center = filter_wavelengths.get(filter_name, None)
        if center:
            lower = center * 0.99
            upper = center * 1.01
            return [upper, lower]
        else:
            return -1

filter_wavelengths = {
    'J': 12500, 'H': 16500, 'K': 22000, 'L': 34500, 'M': 47000,
    'N': 100000, 'Q': 200000, '4.5m': 45000, '5.5m': 55000, '18m': 180000,
    '25m': 250000, '30m': 300000, 'spec8': 80000, 'spec9': 90000,
    'spec9.7': 97000, 'spec10': 100000, 'spec11': 110000, 'spec12': 120000,
    'spec12.5': 125000, 'spec12.7': 127000, 'spec12.8': 128000,
    'spec20': 200000, 'spec30': 300000, 'spec40': 400000
}

def make_input_dat(objects, wave):
    flux = objects[3::2].astype(float)
    ferr = objects[4::2].astype(float)

    a = [i for i in range(wave.size) if flux[i]!=-9999.0]
    wave_ul = [get_filter_info(str(wave[i]))for i in a]

    flux1 = np.array([flux[i] if ferr[i]!=-1.0 else flux[i]/2 for i in a])
    ferr1 = np.array([ferr[i] if ferr[i]!=-1.0 else flux[i]/2 for i in a])
    #mJy to Jy
    flux = flux1 * 1e-3
    ferr = ferr1 * 1e-3

    print('generating {} file'.format(objects[0]))
    #np.savetxt('{}_filter_flx2xsp_input.dat'.format(objects[0]), np.column_stack([np.flip(wave_ul)[:,0], np.flip(wave_ul)[:,1], np.flip(flux),np.flip(ferr)]), fmt=["%5.3f","%5.3f","%2.18e","%2.18e"])
    for i in range(len(wave_ul)):
        np.savetxt('{0}_filter_flx2xsp_input_photo{1:02d}.dat'.format(objects[0], i), np.column_stack([np.flip(wave_ul)[:,0][i], np.flip(wave_ul)[:,1][i], np.flip(flux)[i],np.flip(ferr)[i]]), fmt=["%5.3f","%5.3f","%2.18e","%2.18e"])
        command = ['ftflx2xsp',
                    'infile={0}_filter_flx2xsp_input_photo{1:02d}.dat'.format(objects[0], i),
                    'phafile={0}_filter_photo{1:02d}.pha'.format(objects[0], i),
                    'rspfile={0}_filter_photo{1:02d}.rsp'.format(objects[0], i),
                    'xunit=angstrom',
                    'yunit=Jy',
                    'clobber=yes']
        subprocess.run(command)
    print('generated {} file'.format(objects[0]))

def make_input_dat_spec(objects, wave):
    flux = objects[3::].astype(float)

    a = [i for i in range(wave.size) if flux[i]!=-9999.0]

    wave_ul = [get_filter_info(str(wave[i])) for i in a]

    flux1 = flux[a]
    ferr1 = np.array([flux[i]*0.2 for i in a])
    #mJy to Jy
    flux = flux1 * 1e-3
    ferr = ferr1 * 1e-3

    print('generating {} spectral file'.format(objects[0]))
    #np.savetxt('{}_filter_flx2xsp_input_spec.dat'.format(objects[0]), np.column_stack([np.flip(wave_ul)[:,0], np.flip(wave_ul)[:,1], np.flip(flux),np.flip(ferr)]), fmt=["%5.3f","%5.3f","%2.18e","%2.18e"])
    for i in range(len(wave_ul)):
        np.savetxt('{0}_filter_flx2xsp_input_spec{1:02d}.dat'.format(objects[0], i), np.column_stack([np.flip(wave_ul)[:,0][i], np.flip(wave_ul)[:,1][i], np.flip(flux)[i],np.flip(ferr)[i]]), fmt=["%5.3f","%5.3f","%2.18e","%2.18e"])
        command = ['ftflx2xsp',
                    'infile={0}_filter_flx2xsp_input_spec{1:02d}.dat'.format(objects[0], i),
                    'phafile={0}_filter_spec{1:02d}.pha'.format(objects[0], i),
                    'rspfile={0}_filter_spec{1:02d}.rsp'.format(objects[0], i),
                    'xunit=angstrom',
                    'yunit=Jy',
                    'clobber=yes']
        subprocess.run(command)
    print('generated {} spectral file'.format(objects[0]))

"""
for i in range(1, n_objects):
    objects = data[i, :]
    flux = objects[3::2].astype(float)
    err = objects[4::2].astype(float)

    a = [i for i in range(nchain) if flux[i]!=-9999.0]
    wave_l = wave[a] - wave[a] * 0.01
    wave_u = wave[a] + wave[a] * 0.01
    flux = flux[a]
    err = err[a]
    np.savetxt('{}_flx2xsp_input.dat'.format(objects[0]), np.column_stack([np.flip(wave_u), np.flip(wave_l), np.flip(flux),np.flip(ferr)]), fmt=["%5.3f","%5.3f","%2.18e","%2.18e"])
"""

def main():
    start = time.time()
    args = sys.argv
    #with open("args[1]") as f:
    #    ncols = len(f.readline().split(','))

    #data=np.loadtxt(args[1],comments="#",usecols=range(3,ncols),unpack=True,delimiter=",")
    data=np.loadtxt(args[1], comments="#", unpack=True, delimiter=",", dtype=str)
    n_objects = data.shape[1] - 1
    # wave (micron), fem16: flux (mJy)
    if 'spec' in args[1]:
        wave = data[3::, 0].astype(str)
        Parallel(n_jobs=8,verbose=1)([delayed(make_input_dat_spec)(data[:, j], wave) for j in range(1, n_objects+1)])
    else:
        wave = data[3::2, 0].astype(str)
        Parallel(n_jobs=8,verbose=1)([delayed(make_input_dat)(data[:, j], wave) for j in range(1, n_objects+1)])

    print('elapsed time {}sec'.format(time.time() - start))
    print('{} files'.format(n_objects))


if __name__ == '__main__':
    main()
