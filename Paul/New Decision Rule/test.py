import numpy as np

print('hl')
class A(object):
 def __init__(self):
   print("world")

class B(A):
    def __init__(self):
        print("hello")
        super().__init__()


a = A()
b=B()
a = np.arange(0, 20)
print(a)

print(a[a > 10])
