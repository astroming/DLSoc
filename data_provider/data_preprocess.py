import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,RobustScaler,MinMaxScaler
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
warnings.filterwarnings('ignore')


def interpolate_missing(y, method='linear'):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        if method =='spline' or method == 'polynomial':
            y = y.interpolate(method=method, limit_direction='both', order=5)
        else:
            y = y.interpolate(method=method, limit_direction='both')
    return y
def rangelands_process(args):
    ## tss2 to ts format ################################################
    tss2_bands = [ 'nbart_coastal_aerosol', 'nbart_blue',
     'nbart_green', 'nbart_red', 'nbart_red_edge_1', 'nbart_red_edge_2',
     'nbart_red_edge_3', 'nbart_nir_1', 'nbart_swir_2', 'nbart_swir_3', 'nbart_nir_2']

    tss2_bands = [ele for ele in tss2_bands if ele in args.ts_bands]


    tss2 = pd.read_csv('./dataset/rangelands/ALL_S2_bands.csv')

    tss2['sentinel_date'] = pd.to_datetime(tss2['time']).dt.to_period('M')
    tss2 = tss2[tss2_bands+['core_number', 'sentinel_date']]
    tss2 = tss2.groupby(['core_number', 'sentinel_date']).mean()
    tss2['core_number'] = tss2.index.get_level_values(0)
    tss2['sentinel_date'] = tss2.index.get_level_values(1)
    tss2.reset_index(inplace=True,drop=True)

    stamp = pd.date_range(args.ts_start, args.ts_end, freq='MS')
    stamp = stamp.to_series().dt.to_period('M').to_frame('monthly')

    def myfuc(core_df, stamp=stamp, tss2_bands=tss2_bands, args=args):
        core_ts = stamp.merge(core_df, how='left', left_on='monthly', right_on='sentinel_date')
        core_ts = core_ts[tss2_bands]
        ## fill nan ##
        core_ts = core_ts.transform(lambda x: interpolate_missing(x, method=args.ts_interpolate_method))
        ## convert columns to an array to save as an element of dataframe##
        ts_df = pd.DataFrame(columns=tss2_bands)
        for i, col in enumerate(tss2_bands):
            ts_df.loc[0, col] = core_ts[col].values
        return ts_df

    ts = tss2.groupby('core_number').apply(myfuc)
    ts = ts[tss2_bands]
    ts['core_number'] = ts.index.get_level_values(0)
    ts.reset_index(inplace=True, drop=True)

    ## climate to ts format and then merge with tss2-based ts ###############################

    def func_climate_ts(row,stamp=stamp, args=args):
        row = row.iloc[:-1].to_frame('value')
        row['date'] = row.index.values
        row['date'] = pd.to_datetime(row.date, format="%Y%m").dt.to_period('M')
        row_stamp = stamp.merge(row, how='left', left_on='monthly', right_on='date')
        ## fill nan ##
        re = row_stamp['value'].transform(lambda x: interpolate_missing(x, method=args.ts_interpolate_method))
        return (re.values)

    for cli in ['climate_evap','climate_pw','climate_rain','climate_srad','climate_tavg',
                'climate_tmax','climate_tmin','climate_vpd']:

        df = pd.read_csv(f'./dataset/rangelands/site_{cli}.csv')
        df[cli] = df.apply(lambda x: func_climate_ts(x),axis=1)
        ts = df[['core_number',cli]].merge(ts, how='inner', left_on='core_number', right_on='core_number')

    ts = ts[args.ts_bands+['core_number']]

    ## static features from Cibo ###############

    all_cores = pd.read_csv(f'./dataset/rangelands/site_cibo_covariates.csv')

    ## target y
    df_soc3500 = pd.read_csv(f'./dataset/rangelands/site_soc.csv')

    all_cores = all_cores.merge(df_soc3500[['core_number',args.predtarget]], how='inner', left_on='core_number', right_on='core_number')

    ## static features from terrain #####
    static_terrain = ['srtm-1sec-demh-v1-COG|band_1', 'mrvbf_int|band_1', 'aspect_1s|band_1',
                      'focalrange300m_1s|band_1', 'mrrtf6g-a5_1s|band_1',
                      'plan_curvature_1s|band_1', 'PrescottIndex_01_1s_lzw|band_1',
                      'profile_curvature_1s|band_1', 'slopedeg_1s|band_1',
                      'slopepct1s|band_1', 'slope_relief|band_1', 'twi_1s(wetness)|band_1'
                      ]
    static_terrain = [ele for ele in static_terrain if ele in args.static_f]

    terrain = pd.read_csv(f'./dataset/rangelands/site_terrain.csv')

    terrain_df = terrain[static_terrain+['core_number']]
    # merge terrain_df and all_cores
    all_cores = all_cores.merge(terrain_df, how='inner', left_on='core_number', right_on='core_number')
    # fill nan with median of each static feature
    static_median = all_cores[args.static_f].median()
    all_cores[args.static_f] = all_cores[args.static_f].fillna(static_median)


    ## merge ts and static features
    all = all_cores.merge(ts,how='inner',left_on='core_number',right_on='core_number')
    all = all[args.ts_bands+args.static_f+['task',args.predtarget]]

    ## cores clustering ###########

    file = open(args.processed_path, 'wb')
    pickle.dump(all, file)
    file.close()
    return None