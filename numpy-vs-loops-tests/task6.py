import numpy as np
from itertools import groupby


class RunLengthEncode(object):

    def vect_version(self, x):
        if np.ndim(x) == 0:
            return np.array([x]), np.array([1])
        where = np.flatnonzero
        starts = np.r_[0, where(~np.isclose(x[1:], x[:-1])) + 1]
        times_repeat = np.diff(np.r_[starts, x.shape[0]])
        values = x[starts]
        return np.array(values), np.array(times_repeat)

    def non_vect_version(self, x):
        if np.ndim(x) == 0:
            return np.array([x]), np.array([1])
        count = 1
        prev = None
        values = []
        times_repeat = []
        for elem in x:
            if elem != prev:
                if prev is not None:
                    values.append(prev)
                    times_repeat.append(count)
                count = 1
                prev = elem
            else:
                count += 1
        values.append(prev)
        times_repeat.append(count)
        return np.array(values), np.array(times_repeat)

    def version_groupby(self, x):
        if np.ndim(x) == 0:
            return np.array([x]), np.array([1])
        (values,
         times_repeat) = zip(*((elem, len(list(group)))
                               for elem, group in groupby(x)))
        return np.array(values), np.array(times_repeat)
