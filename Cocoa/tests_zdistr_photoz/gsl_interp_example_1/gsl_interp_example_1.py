import numpy as np
import matplotlib.pyplot as plt

interp = np.genfromtxt("./interp.dat")

x_interp = interp[:,0] ## 1st col
y_interp = interp[:,1] ## 2nd col

x_data = [i + 0.5 * np.sin(i) for i in np.arange(0,10)] 
y_data = [i + np.cos(i*i) for i in np.arange(0,10)]

plt.figure()
plt.scatter(x_data,y_data,label="Data",color="k",marker="s")
plt.plot(x_interp,y_interp,label="Interpolation with GSL")
plt.xlabel("x",fontsize=15)
plt.ylabel("y",fontsize=15)
plt.xlim(0,10)
plt.ylim(0,10)
plt.legend(loc="lower right",fontsize=15)
plt.tight_layout()
plt.savefig("./test.pdf")