import numpy as np

def get_batch(x,batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end <x.shape[0] else x.shape[0]
    return x.iloc[start:end,:]


def transfer_data(sample, fields_dict, array_length):
    array = np.zeros([array_length])
    for field in fields_dict:
        if field == 'click':
            field_value = sample[field]
            ind = fields_dict[field][field_value]
            if ind == (array_length - 1):
                array[ind] = -1
            else:
                array[ind + 1] = 1
        else:
            field_value = sample[field]
            ind = fields_dict[field][field_value]
            array[ind] = 1
    return array


