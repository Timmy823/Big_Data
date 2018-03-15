import math
import numpy as np
import matplotlib.pyplot as plt
def circle(x,y,r):
    xarr=[]
    yarr=[]
    for i in range(9):  #point
        jiao=float(i)/8*2*math.pi
        x0=x+r*math.cos(jiao)
        y0=y+r*math.sin(jiao)
        xarr.append(x0)
        yarr.append(y0)
    print(xarr)
    print(yarr)
    theta = np.linspace(0, 2*np.pi, 100)    #circle
    x1 = r*np.cos(theta)
    x2 = r*np.sin(theta)
    labels = ['000', '001', '011', '010', '110', '111', '101', '100']
    fig, ax = plt.subplots(1)
    ax.plot(x1, x2)
    ax.plot(xarr,yarr,"o", marker = 'o', color = 'r', markersize = 5)
    #for xy in zip(xarr, yarr):                                       # <--data 
    #    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    for label, x, y in zip(labels, xarr, yarr):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax.set_aspect(1)
    plt.xlabel('Re')
    plt.ylabel('Im')
    #plt.title('Circle')
    plt.show()

circle(0,0,1)
