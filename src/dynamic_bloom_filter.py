from src.bloom_filter import QueryCache

class DynamicBloomFilter:
    """
    Implements a dynamic Bloom Filter that can resize as needed.
    """
    def __init__(self, initial_capacity, error_rate):
        self.filters = [QueryCache(initial_capacity, error_rate)]

    def check_and_insert(self, query):
        for bloom in self.filters:
            if bloom.check(query):
                return True
        if self.filters[-1].is_full():
            self.filters.append(QueryCache(self.filters[-1].capacity * 2, self.filters[-1].error_rate))
        self.filters[-1].insert(query)
        return False