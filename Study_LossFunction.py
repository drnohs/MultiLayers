import numpy as np


def cross_entropy_error(y, t):
    h=1e-7
    return -np.sum(t*np.log(y + h))

def mean_squared_error(y,t):
    return np.average((y-t)**2)

print("#1 Perfectly Right Case")
t=np.array([0,0,1,0,0,0,0,0,0])
y1=np.array([0,0,1,0,0,0,0,0,0])

print("CEE :",cross_entropy_error(t,y1))
print("MSE :",mean_squared_error(t,y1))
print("")

print("#2. Completely Wrong Case")
y2=np.array([0.5,0.5,0,0,0,0,0,0,0])

print("CEE :",cross_entropy_error(t,y2))
print("MSE :",mean_squared_error(t,y2))
print("")

print("#3. Partially Wrong Case")
y3=np.array([0,0.5,0.5,0,0,0,0,0,0])

print("CEE :",cross_entropy_error(t,y3))
print("MSE :",mean_squared_error(t,y3))
print("")

print("#4. Another Partially Wrong Case")
y4=np.array([0.25,0.25,0.5,0,0,0,0,0,0])

print("CEE :",cross_entropy_error(t,y4))
print("MSE :",mean_squared_error(t,y4))
print("")