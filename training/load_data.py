import copy
import numpy as np
import os


# Used for loading sequential embedding
def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    # Go through all the tickers one by one
    for index, ticker in enumerate(tickers):
        # Load raw data of each ticker. There are six columns when loading from 2013-01-01 folder.
        # column[0] index or time
        # column[1] most likely 5 day average of normalized price
        # column[2]: most likely 10 day average of normalized price
        # column [3]: most likely 20 day average of normalized price
        # column [4]: most likely 30 day average of normalized price
        # column [5]: most likely the normalized price
        # The length is 1245 which represents total number of days from 2013-2017
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            # Print the length of the overall time series
            print('single EOD data shape:', single_EOD.shape)
            # tensor of time series data
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32) # (num_company,num_days,5)
            # Initially all the masks will be 1
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32) # (num_company,num_days)
            
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32) # (num_company,num_days)
            # Take the price of all stocks
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32) # (num_company,num_days)
        for row in range(single_EOD.shape[0]): # for each day range(0, num_days)
            # Calculate return or ground truth
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # This is most likely to deal with missing data. If any day is 
                # missing for a stock, mask for that day will be 0. Raw data of
                # that day is -1234 in the raw/individual file
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234)> 1e-8:
                # Return is calculated as ground truth
                ground_truth[index][row] =(single_EOD[row][-1] - single_EOD[row - steps][-1]) /single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:] # take all eod data except index
        base_price[index, :] = single_EOD[:, -1] # Take the normalized price of all stocks. Last column of single_EOD
    return eod_data, masks, ground_truth, base_price


def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file) # (num_company, num_company, relation_types)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)

# Used for loading relational data
def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file) # Contains relation data (only 0 or 1)
    print('relation encoding shape:', relation_encoding.shape) # (num_company, num_company, relation_type)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]] # (num_company, num_company)
    # Sum all types of relations. mask_flags will be 1 if no relation exists between
    # two companies
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2)) # (num_company, num_company)
    
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape)) # If there is no relation the mask will be a large negative number there -1e9
    return relation_encoding, mask # return individual and masked relations


def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # handle missing data
                if row < 3:
                    # eod_data[index, row] = 0.0
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            # print(index, row, i, eod_data[index][row])
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
                    # print(index, row, eod_data[index][row])
            else:
                eod_data[index][row] = single_EOD[row][-1]
        # print('test point')
    np.save(market_name + '_sfm_data', eod_data)