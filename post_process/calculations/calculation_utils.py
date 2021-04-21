import os
import json
from numpy import genfromtxt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# there is a bug in log files, some of them have __ instead of _
EPISODE_LOG_NAME = "episode_logfile__average.csv"
EPISODE_LOG_NAME_2 = "episode_logfile_average.csv"
METADATA_FILE_NAME = "metadata"
RAW_LOGFILES_FOLDER = "raw_logfiles"


def read_metadata_file(path):
    metadata_filename = None

    json_files = get_json_files(path)
    for json_file in json_files:
        if json_file[:len(METADATA_FILE_NAME)] == METADATA_FILE_NAME:
            metadata_filename = json_file

    if metadata_filename is None:
        raise ValueError("The file does not exist: 'episode_logfile__average.csv'")

    metadata_path = os.path.join(path, metadata_filename)
    with open(metadata_path, 'r') as f:
        data = json.load(f)

    return data

def get_json_files(path):
    json_files = []
    for rootdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(file)
    return json_files

def get_csv_files(path):
    csv_files = []
    for rootdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(file)
    return csv_files

def get_subfolders(path):
    subfolders = os.listdir(path)

    return subfolders

def load_episode_average_log(path, folder_name):
    log_path = os.path.join(path, folder_name, RAW_LOGFILES_FOLDER)
    episode_average_log_name = None
    csv_files = get_csv_files(log_path)
    for csv_file in csv_files:
        if csv_file == EPISODE_LOG_NAME or csv_file == EPISODE_LOG_NAME_2:
            episode_average_log_name = csv_file

    if episode_average_log_name is None:
        raise ValueError("The file does not exist: 'episode_logfile__average.csv'")

    log_path = os.path.join(log_path, episode_average_log_name)

    df = pd.read_csv(log_path, sep=',')
    # log = genfromtxt(log_path, delimiter=',', encoding="utf8")
    print(len(df.values), folder_name)
    return df

def load_time_step_data(time_step_path, vehicles=[1], episode=1):
    time_step_data = {}
    for v in vehicles:
        vehicle_path = "vehicle_" + str(v)
        episode_path = "timestep_logfile_" + vehicle_path + "_episode_" + str(episode) + ".csv"
        log_path = os.path.join(time_step_path, vehicle_path, episode_path)
        df = pd.read_csv(log_path, sep=',')
        time_step_data[v] = df

    return time_step_data

def load_espisodes_data(simulation_path_base, file ="episode_logfile_average.csv" , subfolders = None):
    if subfolders is None:
        subfolders = get_subfolders(simulation_path_base)
    espisodes_data = {}
    for subfolder in subfolders:
        path = os.path.join(simulation_path_base, subfolder)
        log_path = os.path.join(path, "raw_logfiles", file)
        df = pd.read_csv(log_path, sep=',')
        espisodes_data[subfolder] = df
    return espisodes_data

def ewma(x, alpha):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n,n)) * (1-alpha)
    p = np.vstack([np.arange(i,i-n,-1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p,0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)

def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order='C', out=None):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out

def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon)/np.log(1-alpha)) + 1

def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def espisodes_data_array_from_dict(espisodes_data, stat="episode_reward" ,  span = 1000 , fix_same_len =True , ema=True , std=False , span_std = 100):
    """

    """
    espisodes_data_list = []
    espisodes_data_list_std = []
    lens = []

    for k, values in espisodes_data.items():
        datax = values[stat]
        # ema = values[stat].ewma(span=10).mean()
        # sma = values[stat].rolling(10, min_periods=1).mean()
        # sma = datax.rolling(10, min_periods=1).mean()
       # span >= 1
        if std:
            data_std = np.std(rolling_window(np.array(datax), span_std), 1)
            espisodes_data_list_std.append(np.array(data_std))
        if ema:
            alpha = 2 / (span + 1)
            datax = ewma_vectorized_safe(datax, alpha)


        espisodes_data_list.append(np.array(datax))
        lens.append(len(datax))
    if fix_same_len:
        lenmin = min(lens)
        espisodes_data_list_filtered = [data[:lenmin] for data in espisodes_data_list]
        if std:
            espisodes_data_list_std_filtered = [data[:lenmin-span_std] for data in espisodes_data_list_std]
    else:
        espisodes_data_list_filtered = espisodes_data_list
    if std:
        espisodes_data_list_std_filtered_zeros = np.zeros((np.shape(espisodes_data_list_filtered)))
        espisodes_data_list_std_filtered = np.array(espisodes_data_list_std_filtered)
        espisodes_data_list_std_filtered_zeros[:,span_std:] = espisodes_data_list_std_filtered
        return np.array(espisodes_data_list_filtered) , np.array(espisodes_data_list_std_filtered_zeros)
    else:
        return np.array(espisodes_data_list_filtered)

def espisodes_distance_traveled(espisodes_data, stat="episode_average_speed_all" ,  span = 1000 , fix_same_len =True , ema=True , std=False , span_std = 100):
    """

    """
    espisodes_data_list = []
    espisodes_data_list_std = []
    lens = []
    for k, values in espisodes_data.items():
        datav = values[stat]
        # ema = values[stat].ewma(span=10).mean()
        # sma = values[stat].rolling(10, min_periods=1).mean()
        # sma = datax.rolling(10, min_periods=1).mean()
        data_duration = values["episode_length"]
        distance_traveled = datav * data_duration
        if std:
            data_std = np.std(rolling_window(np.array(distance_traveled), span_std), 1)
            espisodes_data_list_std.append(np.array(data_std))
       # span >= 1
        if ema:
            alpha = 2 / (span + 1)
            distance_traveled = ewma_vectorized_safe(distance_traveled, alpha)
        espisodes_data_list.append(np.array(distance_traveled))
        lens.append(len(distance_traveled))
    if fix_same_len:
        lenmin = min(lens)
        espisodes_data_list_filtered = [data[:lenmin] for data in espisodes_data_list]
        if std:
            espisodes_data_list_std_filtered = [data[:lenmin-span_std] for data in espisodes_data_list_std]

    else:
        espisodes_data_list_filtered = espisodes_data_list
    if std:
        espisodes_data_list_std_filtered_zeros = np.zeros((np.shape(espisodes_data_list_filtered)))
        espisodes_data_list_std_filtered = np.array(espisodes_data_list_std_filtered)
        espisodes_data_list_std_filtered_zeros[:, span_std:] = espisodes_data_list_std_filtered
        return np.array(espisodes_data_list_filtered), np.array(espisodes_data_list_std_filtered_zeros)
    else:
        return np.array(espisodes_data_list_filtered)

def time_step_array_from_dict(time_step_logs, stats=["timestep_reward", "vehicle_speed"]):
    """
    Return time_step_data as a numpy array
    shape: stats x vehicle x time_step
    """
    vehicles_data = []
    time_step_data = []
    for stat in stats:
        vehicles_data = []
        for k, values in time_step_logs.items():
            vehicles_data.append(np.array(values[stat]))
        time_step_data.append(np.array(vehicles_data))

    return np.array(time_step_data)



def episodes_data_array_from_dict(time_step_logs, stats=["timestep_reward", "vehicle_speed"]):
    """
    Return time_step_data as a numpy array
    shape: stats x vehicle x time_step
    """
    vehicles_data = []
    time_step_data = []
    for stat in stats:
        vehicles_data = []
        for k, values in time_step_logs.items():
            vehicles_data.append(np.array(values[stat]))
        time_step_data.append(np.array(vehicles_data))

    return np.array(time_step_data)

def average_binary_array(data, n=1000):
    out = []
    for i in range(0, len(data), n):
        ini = i
        end = min(len(data)-1, i +n)
        dataint = data.astype(int)
        # dataint = [int(val) for val in data]
        out.append(np.sum(dataint[ini:end]))

    return np.array(out)/n