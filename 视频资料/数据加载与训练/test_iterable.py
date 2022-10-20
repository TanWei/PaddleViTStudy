class MyIterable:
    def __init__(self):
        self.data = [1,2,3,4,6]
    def __iter__(self):
      return MyIterator(self.data)
    def __getitem__(self, idx):
      return self.data[idx]
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.counter = 0
    def __iter__(self):
      return self
    def __next__(self):
        
      if self.counter >= len(self.data):
          raise StopIteration()
      x = self.data[self.counter]
      self.counter += 1
      return x
def main():
    my_iterable = MyIterable()
    for i in my_iterable:
        print(i)
    print(my_iterable[3])
    
def get_batch_sampler(size_param):
    print("starting...")
    batch_indexes = []
    for i in iter(range(12)):
        batch_indexes.append(i)
        if len(batch_indexes) == size_param:
            yield batch_indexes
            batch_indexes.clear()
    yield batch_indexes
if __name__ == "__main__":
    for i in range(13):
        print(i)
    # main()
    # print(get_batch_sampler(5))
    # print(get_batch_sampler(5))
    l1 = get_batch_sampler(5)
    for cc in l1:
        print(cc)
    n = 10
