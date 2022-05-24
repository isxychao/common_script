import os
import h5py
import pandas as pd

base_path = 'c3_muse_stress_2022'

h5 = h5py.File("c3_muse_stress_2022.hdf5","w")

dirList1 = os.listdir(base_path)
dirList1.sort()
dirList1.remove('raw_data')
dirList1.remove('metadata')

for dir1 in dirList1:  # dirList1: (feature_segments, label_segments , ...)

    h5_g1 = h5.create_group(dir1)

    path1 = os.path.join(base_path, dir1)
    dirList2 = os.listdir(path1)
    dirList2.sort()

    for dir2 in dirList2: # dirList2: (bert-4, biosignals, ...)

        h5_g2 = h5_g1.create_group(dir2)

        path2 = os.path.join(path1, dir2)
        dirList3 = os.listdir(path2)
        dirList3.sort(key= lambda x:int(x[:-4])) # 对列表进行排序

        features = []
        for dir3 in dirList3: # dirList3: (1.csv, 2.csv, ...)

            path3 = os.path.join(path2, dir3)

            feature = pd.read_csv(path3)

            h5_g2.create_dataset(dir3, data=feature)
			
h5_g = h5['metadata']
feature = pd.read_csv('c3_muse_stress_2022/metadata/partition.csv')


# csv中不仅有数字也会有字符串，分别取出每列
# 将两列转成list
feature_c1, feature_c2 = feature['Id'].to_list(), feature['Partition'].to_list()
feature_c1 =list(map(str,feature_c1)) # 将list中的数字转化为字符串

# 将list转为numpy，并增加维度
feature_c1 = np.expand_dims(np.array(feature_c1, dtype='S'), axis=1)
feature_c2 = np.expand_dims(np.array(feature_c2, dtype='S'), axis=1)

# 将两者拼接
feature = np.hstack((feature_c1, feature_c2))

h5_g.create_dataset('partition.csv', data = feature)

h5.close()
