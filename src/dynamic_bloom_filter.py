from src.bloom_filter import QueryCache

class DynamicBloomFilter:
    """
    Implements a dynamic Bloom Filter that can resize as needed.
    """
    def __init__(self, initial_capacity, error_rate):
        self.filters = [QueryCache(initial_capacity, error_rate)]

    def check_and_insert(self, query):
        """
        Checks if the query exists in any of the Bloom Filters. If not, inserts it.
        If the latest Bloom Filter is full, a new Bloom Filter is added with double the capacity.
        """
        for bloom in self.filters:
            if bloom.check_and_insert(query):
                return True  # Query is already in one of the filters

        # If the last Bloom Filter is full, create a new one with double capacity
        if len(self.filters[-1].bloom_filter) >= self.filters[-1].bloom_filter.capacity:
            new_capacity = self.filters[-1].bloom_filter.capacity * 2
            self.filters.append(QueryCache(new_capacity, self.filters[-1].bloom_filter.error_rate))
        self.filters[-1].check_and_insert(query)  # Insert into the newest filter
        return False  # Query was not in the filters