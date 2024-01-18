import unittest
import numpy as np
from motionmapperpy.motionmapper import rolling_window_with_padding

class TestRollingWindowWithPadding(unittest.TestCase):

    def test_rolling_window_with_padding(self):
        data=np.arange(0, 20, 1)
        print(data.shape)

        windows=rolling_window_with_padding(arr=data, window_size=2)
        print(windows.shape)

        windows=rolling_window_with_padding(arr=data, window_size=2, skip=5)
        print(windows)

    
if __name__ == '__main__':
    unittest.main()