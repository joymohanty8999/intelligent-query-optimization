from pybloom_live import BloomFilter

class QueryCache:
    def __init__(self, capacity, error_rate):
        """
        Initializes a Bloom Filter for query caching.
        """
        self.bloom_filter = BloomFilter(capacity=capacity, error_rate=error_rate)

    def check_and_insert(self, query):
        """
        Checks if the query exists in the Bloom Filter. If not, inserts it.
        """
        if query in self.bloom_filter:
            return True  # Query is already in the filter
        else:
            self.bloom_filter.add(query)  # Insert the query into the filter
            return False  # Query was not in the filter