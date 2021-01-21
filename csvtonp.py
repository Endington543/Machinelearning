import matplotlib.image
from numpy import genfromtxt
my_data = genfromtxt('encoded1.csv', delimiter=',')
matplotlib.image.imsave('name.png', my_data, cmap="gray")