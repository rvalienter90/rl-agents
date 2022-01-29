import pickle
import  numpy as np
#
# a = {'hello': 'world'}
#
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data.pickle', 'rb') as handle:
    b = pickle.load(handle)

data_all=b
for i in range(1,6):
    data='data'+str(i) + '.pickle'

    with open(data, 'rb') as handle:
        b = pickle.load(handle)
    data_temp = np.append(data_all['vx'],b['vx'])
    data_all['vx'] = data_temp

    data_temp = np.append(data_all['vy'], b['vy'])
    data_all['vy'] = data_temp

    data_temp = np.append(data_all['xt'], b['xt'])
    data_all['xt'] = data_temp

    data_temp = np.append(data_all['yt'], b['yt'])
    data_all['yt'] = data_temp

with open('../libs/datainter.pickle', 'wb') as handle:
    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Ok")

#
# with open('../libs/data_highway.pickle', 'rb') as handle:
#     b = pickle.load(handle)

