# from numpy import number


# class Test:
#     def __init__(self, number):
#         self.i = 0
#         self.number = number
    
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.i < self.number:
#             i = self.i
#             self.i += 2
#             return i
        
#         else:
#             raise StopIteration()

# for i in Test(10):
#     print(i)


class MyIterable():
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
    def __iter__(self):
        return MyIterator(self.data)

    def __getitem__(self, idx):
        # 重载[]
        return self.data[idx]

class MyIterator():
    def __init__(self, data):
        self.data = data
        self.counter = 0

    def __iter__(self):
        # 返回一个迭代器
        return self

    def __next__(self):
        # 循环过程
        if self.counter >= len(self.data):
            raise StopIteration()
        data = self.data[self.counter]
        self.counter +=1
        return data


# my_iterable = MyIterable()

# for d in my_iterable:
#     print(d)

# for d in my_iterable:
#     print(d)

# print(my_iterable[0])

def h():
    while True:
        print('study yield')
        yield 5
        print('go on!')

c = h()
n1 = next(c)
n1 = next(c)