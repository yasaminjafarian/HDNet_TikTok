import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../training_progress/trainLog.txt')


plt.figure()
data2 = data[0:,1]
plt.plot(data[0:,0],data2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('training loss in iterations')
plt.show()

