#%%
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.decomposition import PCA

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import kDBCV


def cal_DBSCAN(infile, ep, sp):
    def reset_gps(df):
        idx_mx = df.mag.argmax()
        a = df.grps[idx_mx]

        if a !=0:
            lst = sorted( df.grps.unique())
            n = len(lst)
            if -1 in lst: n-=1
                
            lst2 = [ (i+n-a)%n if i >=0 else -1 for i in lst   ]
            mp_grp = dict(zip(lst, lst2))
            df['grps2'] = df.apply(lambda x: mp_grp[x.grps], axis=1)
            df['grps'] = df['grps2']
            df = df.drop('grps2', axis=1)
        return df

    df = pd.read_csv(infile)
    df['otime'] = pd.to_datetime(df.otime)
    gdf = gpd.GeoDataFrame(data=df, geometry=gpd.points_from_xy(df['lon'], df['lat']))
    gdf = gdf.set_crs('epsg:4326') # WGS84
    gdf = gdf.to_crs('epsg:3826')  # trans to TWD97
    df['x'] = gdf.geometry.x       # TWD97 x
    df['y'] = gdf.geometry.y       # TWD97 y
    df['z'] = df.dep * 1000        # Z, km to m

    #--
    dfx = df.reset_index(drop=True)
    idx_lmg = dfx.mag >=5
    arr_weight = np.ones(dfx.shape[0]) 
    arr_weight[idx_lmg] = 3
    X = dfx[['x', 'y', 'z']]
    scale = MinMaxScaler()
    Y  = scale.fit_transform(X)

    clustering = DBSCAN(eps=ep, min_samples=sp).fit(Y, sample_weight=arr_weight) 
    dfx['grps'] = clustering.labels_
    dfx = reset_gps(dfx)
    n = max(clustering.labels_)+1 # noise, gp1, gp2 ....
    n_noise = dfx[dfx.grps == -1].shape[0]

    #-- some score
    idx_x = dfx.grps != -1
    labels = clustering.labels_
    s_ss = metrics.silhouette_score(Y, labels, metric='euclidean')
    s_dbcv, _ = kDBCV.DBCV_score(   Y[idx_x], labels[idx_x])
    s_dbi = metrics.davies_bouldin_score(   Y, labels)
    s_chi = metrics.calinski_harabasz_score(Y, labels)

    return n, s_ss, s_dbcv, s_dbi, s_chi, n_noise, dfx

def run_pca(df):
    def area2mw(area):
        mw = 4.07+0.98*np.log10(area)
        return mw
    
    def plane_4corn(plane_points, ix, iy):
        # TWD97
        x = plane_points[iy, ix, 0]
        y = plane_points[iy, ix, 1]
        z = plane_points[iy, ix, 2]
        return (x, y, z)

    def cal_dis3D(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)

        distance = np.linalg.norm(p1 - p2)
        distance = distance / 1000.
        return round(distance, 1) # m to km

    def define_geometry(plane_points):
        nn = plane_points.shape[0] -1
        pt1 =  plane_4corn(plane_points,  0,  0)
        pt2 =  plane_4corn(plane_points, nn,  0)
        pt3 =  plane_4corn(plane_points, nn, nn)
        pt4 =  plane_4corn(plane_points,  0, nn)

        length= cal_dis3D(pt2, pt1)
        width = cal_dis3D(pt3, pt2)
        if width > length:
            _width2 = width
            _length2 = length
            length = _width2
            width = _length2

        area = width*length # km*km
        mw = area2mw(area)

        return mw, width, length
    
    def cal_pca(data):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(data)
        pca = PCA(n_components=3)
        pca.fit(X_scaled)

        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        pc3 = pca.components_[2] # normal_vector

        if pc3[2] < 0:  pc3 = -pc3

        normal_vector = pc3 / np.linalg.norm(pc3)

        dip = np.degrees(np.arccos(abs(normal_vector[2])))  
        strike = np.degrees(np.arctan2(normal_vector[0], normal_vector[1]))  
        strike = (strike - 90)%360
        center_point = np.mean(data, axis=0)

        projected = (data - center_point) @ np.vstack((pc1, pc2)).T

        du = (projected[:, 0].max() - projected[:, 0].min()) * 0.
        dv = (projected[:, 1].max() - projected[:, 1].min()) * 0.
        u_range = np.linspace(projected[:,0].min()-du, projected[:,0].max()+du, 10)
        v_range = np.linspace(projected[:,1].min()-dv, projected[:,1].max()+dv, 10)
        u, v = np.meshgrid(u_range, v_range)

        plane_points = center_point + u[..., None]*pc1 + v[..., None]*pc2

        return plane_points, strike, dip 

    dic_info  = {'grp':[], 'strike':[], 'dip':[], 
                 'width':[], 'length':[], 'mw':[],  
                 } 

    keys = sorted(df.grps.unique())
    
    for igp in keys:
        if igp == -1: continue # noise
        idx = (df.grps == igp)
        data = df.loc[idx, ['x', 'y', 'dep']].to_numpy()
        data[:, -1] *= -1000.
        plane_points, strike, dip  = cal_pca(data)
        mw, width, length = define_geometry(plane_points)

        dic_info['grp'].append(igp)
        dic_info['mw'].append(mw.round(2))
        dic_info['width'].append(width.round(1))
        dic_info['length'].append(length.round(1))
        dic_info['strike'].append(strike.round(2))
        dic_info['dip'].append(dip.round(2))

    df_pca = pd.DataFrame(data=dic_info)
    return  df_pca

file = '20240403.csv'
infile = os.path.join('catalog', file)
n, s_ss, s_dbcv, s_dbi, s_chi, n_noise, df = cal_DBSCAN(infile, ep=0.07, sp=10)
df_pca = run_pca(df)

