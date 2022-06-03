#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:09:35 2020

@author: njwgis
"""

import numpy as np # import numpy package for math functions
import matplotlib.pyplot as plt # import matplotlib package for plotting

X = np.linspace(0,10,1000) # create array of 1000 evenly spaced numbers from 0 to 10
Ys = X**2/4 # compute values of supply curve
Yd = 10 - X # compute values of demand curve

idxintersect = np.argmin(np.abs(Ys-Yd)) # find index of intersection point
xintersect = X[idxintersect] # find x value of intersection point
yintersect = Yd[idxintersect] # find y value of intersection pont

Yp = X*0 + yintersect # compute values of equilibrium price curve (p=5)

fig = plt.figure(figsize=(4,4),dpi=300) # create figure object named fig with size 4"x4" and 300 dpi resolution
plt.plot(X,Ys,color='k') # add linep lot of supply curve
plt.plot(X,Yd,color='k',linestyle=':') # add linep lot of supply curve
plt.plot(X,Yp,color='k',linestyle=':',lw=3,zorder=11) # add line plot of price
plt.plot([0,2,2,6,6,10,10],[0,0,4,4,6,6,8],color='orange') # add step function
plt.xlabel("X") # label x axis
plt.ylabel("Y") # label y axis
plt.title("Example Figure") # create title
plt.gca().set_aspect('equal') # set aspect ratio
plt.xlim(0,10) # set x axis limits for plot
plt.ylim(0,10) # set y axis limits for plot
plt.fill_between(X,Ys,Yp,where=X<xintersect,color='b') # fill area between price and supply curves for x < xintersect
plt.fill_between(X,Yd,Yp,where=X<xintersect,color='r') # fill area between price and supply curves for x < xintersect
plt.axvline(xintersect,color='r',linestyle='--') # plot dashed vertical line at xintersect
plt.text(5.5,6,"Equilibrium",fontsize=18,color='purple') # add text to plot area
plt.arrow(5.5,6,-0.5,-0.5, head_width=.2, head_length=.2, color = 'k', length_includes_head=True) # add an arrow from equilibrium text to equilibrium point
plt.scatter([xintersect],[yintersect],color='g',zorder=12) # add scatter plot for equilibrium point in green
plt.gca().add_patch(plt.Polygon([[10,10],[8,10],[10,8]], color='m')) # add a magenta triangle in the upper right corner
plt.gca().add_patch(plt.Circle((8,4),1, color='c')) # add a cyan circle centered at 8, 4 with radius 1
plt.legend(['Supply', 'Demand', 'Equilibrium Price','Step Function','Equilibrium Quantity'],loc='lower right') # create legend in lower right corner
fig.savefig('example.png',dpi=fig.dpi) # save figure as png file