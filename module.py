import numpy as np

def sum(a,b):
    if not type(a) is int:
        raise TypeError("sum is only for integers")
    if not type(b) is int:
        raise TypeError("sum is only for integers")
    return a+b

def substract(a,b):
    if a<0:
        if b<0:
            return -(np.abs(a)+nb.abs(b))
    return a-b

if __name__=="__main__":
    print(sum(1,1))

