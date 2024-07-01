import re

import pandas as pd

# ファイルを読み込む
nustar_df = pd.read_table('nustarbass.csv', header=0,delimiter=",",parse_dates=['time', 'end_time'],dtype=str)
suzaku_df = pd.read_table('suzakubass.csv', header=0,delimiter=",",parse_dates=['time', 'stop_time'],dtype=str)
xmm_df = pd.read_table('xmmbass.csv', header=0,delimiter=",",parse_dates=['time', 'end_time'],dtype=str)
chandra_df = pd.read_table('chandrabass.csv', header=0,delimiter=",",parse_dates=['time'],dtype=str)

nustar_df.columns = nustar_df.columns.str.strip()
suzaku_df.columns = suzaku_df.columns.str.strip()
xmm_df.columns = xmm_df.columns.str.strip()
chandra_df.columns = chandra_df.columns.str.strip()

nustar_df = nustar_df[nustar_df["status"]=="archived "]
xmm_df = xmm_df[xmm_df["status"]=="archived "]
chandra_df = chandra_df[chandra_df["status"]=="archived  "]

# 時刻列をdatetime型に変換する
nustar_df['time'] = pd.to_datetime(nustar_df['time'])
nustar_df['end_time'] = pd.to_datetime(nustar_df['end_time'])
suzaku_df['time'] = pd.to_datetime(suzaku_df['time'])
suzaku_df['stop_time'] = pd.to_datetime(suzaku_df['stop_time'])
xmm_df['time'] = pd.to_datetime(xmm_df['time'])
xmm_df['end_time'] = pd.to_datetime(xmm_df['end_time'])

# 観測時間でソート
nustar_df = nustar_df.sort_values(by='time')
suzaku_df = suzaku_df.sort_values(by='time')
xmm_df = xmm_df.sort_values(by='time')

# 重複観測IDを格納するリスト
overlapping_obsids = []

# インデックスの初期化
suzaku_index = 0

# NuSTARの各観測について、Suzakuの観測時間と重なるかを確認
for _, n_row in nustar_df.iterrows():
    n_start = n_row['time']
    n_stop = n_row['end_time']

    # Suzakuの観測を進める
    while suzaku_index < len(suzaku_df) and suzaku_df.iloc[suzaku_index]['stop_time'] < n_start:
        suzaku_index += 1

    # 現在のSuzaku観測が重なるかを確認
    for s_index in range(suzaku_index, len(suzaku_df)):
        s_row = suzaku_df.iloc[s_index]
        s_start = s_row['time']
        s_stop = s_row['stop_time']

        if s_start > n_stop:
            break

        if n_start <= s_stop and n_stop >= s_start and n_row['SwiftName'] == s_row['SwiftName']:
            overlapping_obsids.append((n_row['SwiftName'], n_row['obsid'], s_row['obsid']))

# 重複観測IDを表示
print(len(overlapping_obsids))
df = pd.DataFrame(overlapping_obsids, columns=["id", "nustar_obsid", "suzaku_obsid"],dtype=str)
df.to_csv('suzaku_nustar.csv', index=False)


overlapping_obsids = []
# インデックスの初期化
xmm_index = 0

# NuSTARの各観測について、xmmの観測時間と重なるかを確認
for _, n_row in nustar_df.iterrows():
    n_start = n_row['time']
    n_stop = n_row['end_time']

    # xmmの観測を進める
    while xmm_index < len(xmm_df) and xmm_df.iloc[xmm_index]['end_time'] < n_start:
        xmm_index += 1

    # 現在のxmm観測が重なるかを確認
    for s_index in range(xmm_index, len(xmm_df)):
        s_row = xmm_df.iloc[s_index]
        s_start = s_row['time']
        s_stop = s_row['end_time']

        if s_start > n_stop:
            break

        if n_start <= s_stop and n_stop >= s_start and n_row['SwiftName'] == s_row['SwiftName']:
            overlapping_obsids.append((n_row['SwiftName'], n_row['obsid'], s_row['obsid']))

# 重複観測IDを表示
print(len(overlapping_obsids))
df = pd.DataFrame(overlapping_obsids, columns=["id", "nustar_obsid", "xmm_obsid"],dtype=str)
df.to_csv('xmm_nustar.csv', index=False)
df.to_csv('xmm_nustar_nustar.csv', index=False, columns=['id', 'nustar_obsid'])
df.to_csv('xmm_nustar_xmm.csv', index=False, columns=['id', 'xmm_obsid'])