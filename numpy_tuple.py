import numpy as np

class ArrayTupleConverter:
    """ this class is used to convert back and forth nd arrays of specific shape to tuple
    and tuples to nd arrays of that shape"""

    def __init__(self, array_shape):
        self.shape = array_shape
        self.flat_length = self.calc_flat_length()

    def calc_flat_length(self):
        flat_length = 1
        for dim in self.shape:
            flat_length = flat_length * dim
        return flat_length

    def to_tuple(self, array: np.ndarray):
        assert self.shape == array.shape
        as_1d_tuple = tuple(array.reshape(self.flat_length))
        return as_1d_tuple

    def to_array(self, tup: tuple):
        assert len(tup)==self.flat_length
        flat_array = np.array(tup)
        as_nd_array = flat_array.reshape(self.shape)
        return as_nd_array


if __name__ == '__main__':
    """ semi test """
    con = ArrayTupleConverter((2,2))
    a = np.array([0, 0, 0, 1]).reshape((2, 2))
    b = np.array([0, 0, 1, 0]).reshape((2, 2))
    c = np.array([0, 1, 0, 0]).reshape((2, 2))
    d = np.array([1, 0, 0, 0]).reshape((2, 2))
    s = set()
    for i in (a,b,c,d):
        t = con.to_tuple(i)
        s.add(t)
    print(len(s))