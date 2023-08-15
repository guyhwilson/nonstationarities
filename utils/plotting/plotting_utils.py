import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d



def figSize(x, y):
    '''Change pyplot figure size. Inputs are:

    x (float) - height
    y (float) - width

    swapped so it feels like doing matrix stuff'''

    matplotlib.rcParams['figure.figsize'] = [y, x]
    matplotlib.rcParams['pdf.fonttype'] = 42 
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'

def setTickLabels(ax, labels, which):
    if which == 'x':
        ax.set_xticklabels(np.concatenate([[''],  labels]))
    elif which == 'y':
        ax.set_yticklabels(np.concatenate([[''],  labels]))
    else:
        raise ValueError


def modifyBoxPlotAlpha(ax, alpha):
    '''Change opacity of box plots.'''

    if len(ax.artists) != 0:
        iterator = ax.artists
    else:
        iterator = ax.patches

    for patch in iterator:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

    
def modifyViolinAlpha(ax, alpa):
    ax.setp(ax.collections, alpha=.3)
    
    
def getGroupedColors(subgroups, reference_list):
  '''For different subgroups of <reference_list>, generate a list of colors for
     them each. '''
    
    
    
    
class Arrow3D(FancyArrowPatch):
  '''Plot arrows on end of 3D trajectory plots. Inputs are:

    xs (tuple) - start and stop of arrow (also defines orientation)
    ys (tuple) - see above.
    zs (tuple) - see above.


  * copied from stack exchange question reply 

  '''
  def __init__(self, xs, ys, zs, *args, **kwargs):
      FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
      self._verts3d = xs, ys, zs

  def draw(self, renderer):
      xs3d, ys3d, zs3d = self._verts3d
      xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
      self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
      FancyArrowPatch.draw(self, renderer)
      