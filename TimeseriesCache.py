# https://gist.github.com/jamesp/1385745/4736616d10551a6abc9c7c58725ee5b2f8de9f4d
# Modified to return just the data elements from the list
from bisect import bisect_left, bisect_right


class TimeseriesCache(object):
    """Store a timeseries of data in memory for random access."""
    def __init__(self, ttl=None):
        self.timestamps = []
        self.data = []
        self.ttl = ttl
        
    def add(self, timestamp, data):
        """Add a set of time-value data to the cache"""
        index = bisect_right(self.timestamps, timestamp)
        self.timestamps.insert(index, timestamp)
        self.data.insert(index, data)
        if self.ttl:
            self.__sweep(timestamp)
    
    def __sweep(self, timestamp):
        """Remove ticks older than the ttl."""
        index = bisect_right(self.timestamps, timestamp - self.ttl)
        self.timestamps = self.timestamps[index:]
        self.data = self.data[index:]
    
    def get(self, timestamp, default=None):
        """Returns the data value stored for the time specified."""
        index = bisect_right(self.timestamps, timestamp)
        if index == 0:
            return default
        else:
            return self.data[index - 1]
    
    def __setitem__(self, timestamp, data):
        self.add(timestamp, data)
    
    def __getitem__(self, timestamp):
        """Returns the last available data for a given timestamp.
        If a slice is passed, returns a list of tuples: (timestamp, data)
        that fall within the slice range.
        Returns a data item or a list of data items."""
        if isinstance(timestamp, slice):
            i, j = None, None
            if timestamp.start:
                i = bisect_left(self.timestamps, timestamp.start)
            if timestamp.stop:
                j = bisect_right(self.timestamps, timestamp.stop)
            return self.data[i:j]
        else:
            i = bisect_right(self.timestamps, timestamp)
            if i == 0:
                return None
            else:
                return self.data[i - 1]
            
    def head(self):
        """Return the most recent data in the cache."""
        return self.data[-1]

    def size(self):
        """Returns the size of the data."""
        return len(self.data)


if __name__ == '__main__':
    c = TimeseriesCache()
    c.add(1, 'A')
    c.add(2, 'B')
    c.add(3, 'C')
    c.add(5, 'E')
    c.add(4, 'D')
    c.add(6, 'F')

    assert(c[0] is None)
    assert(c[1] == 'A')
    assert(c[3.5] == 'C')
    assert(c[78] == 'F')
    assert(c[3.4:6] == ['D', 'E', 'F'])