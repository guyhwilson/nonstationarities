import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def createHeatmapAnimation(probs, stateLocs, cursorPos, targetPos=None, toggle_log = False,
                          cursor_color = 'white', marker_size = 20, target_color = 'k', cmap = 'hot_r', 
                          frames = 1000, start_frame = 0):
    
    '''Create a GIF of PRI-T inferences on top of cursor movements. Inputs are:
    
        probs (2D float)      - times x n_states of probabilities
        stateLocs (2D float)   - n_states x 2 of coordinates for each possible state
        cursorPos (2D float)  - time x 2 of cursor positions
        targetPos (2D float)  - time x 2 of ground-truth target positions '''
    
    assert probs.shape[1] == stateLocs.shape[0], "n_states different according to <probs> and <stateLocs>"
    n_states = int(np.sqrt(probs.shape[1]))


    def get_data(t):
        cursor = (cursorPos[t, :] - np.min(stateLocs, axis = 0)) / (np.max(stateLocs, axis = 0) - np.min(stateLocs, axis = 0))
        cursor *= 20
        cursor = np.maximum(cursor, 0)
        cursor = np.minimum(cursor, n_states)

        if targetPos is not None and not np.isnan(targetPos[t, :]).any():
            target = (targetPos[t, :] - np.min(stateLocs, axis = 0)) / (np.max(stateLocs, axis = 0) - np.min(stateLocs, axis = 0))
            target *= 20
            target = np.maximum(target, 0)
            target = np.minimum(target, n_states)
        else:
            target = None

        grid_probs = np.log(probs[t, :].reshape(n_states, n_states)) if toggle_log else probs[t, :].reshape(n_states, n_states)

        return cursor, target, grid_probs


    cursor, target, grid_probs = get_data(0)

    vmin = np.log(1 / n_states**2) if toggle_log else 0
    vmax = np.log(0.99) if toggle_log else 0

    fig = plt.figure()
    im  = plt.imshow(grid_probs, cmap = cmap, vmin = vmin, vmax = vmax, )#extent = [0, 20, 0, 20])
    if target is not None:
        targ = plt.scatter(target[0], target[1], c = target_color, s = marker_size * 50, alpha = 0.7)
    else:
        targ = plt.scatter([], [])
    curs   = plt.scatter(cursor[0], cursor[1], c = cursor_color, s = marker_size, edgecolors = 'k')

    plt.xlim(0, n_states)
    plt.ylim(0, n_states)

    ax      = plt.gca()
    divider = make_axes_locatable(ax)
    cax  = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('PRI-T log prob')

    ax.set_yticks([])
    ax.set_xticks([])


    def animateinit(): #tells our animator what artists will need re-drawing every time
        return [im, targ, curs]

    def animate(t):
        idx    = t + start_frame

        cursor, target, grid_probs = get_data(idx)

        im.set_array(grid_probs)

        if target is not None and not np.isnan(target).any():
            targ.set_offsets(np.c_[target[0], target[1]])
        curs.set_offsets(np.c_[cursor[0], cursor[1]])


        return [im, targ, curs]


    anim = matplotlib.animation.FuncAnimation(fig, animate, init_func = animateinit, 
                                              frames=frames, interval = 20, blit = True, cache_frame_data=False)
    return anim