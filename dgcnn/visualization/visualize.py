# python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# matplotlib settings
# matplotlib.use('Agg') # TkAgg


def get_tableau_palette():
    """
    Creates a pallete of colors for pretty plotting.
    """
    palette = np.array([[78, 121, 167],  # blue
                        [255, 87, 89],  # red
                        [89, 169, 79],  # green
                        [242, 142, 43],  # orange
                        [237, 201, 72],  # yellow
                        [176, 122, 161],  # purple
                        [255, 157, 167],  # pink
                        [118, 183, 178],  # cyan
                        [156, 117, 95],  # brown
                        [186, 176, 172]  # gray
                        ], dtype=np.uint8)
    return palette


def set_axes_equal(ax, limits=None):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    :param ax: a matplotlib axis, e.g., as output from plt.gca().
    :param limits: The list limits for each dimesnion in the current axis
    """
    if limits is None:
        # get current axes limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        # normalize the axes
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    else:
        x_limits, y_limits, z_limits = limits
        ax.set_xlim3d([x_limits[0], x_limits[1]])
        ax.set_ylim3d([y_limits[0], y_limits[1]])
        ax.set_zlim3d([z_limits[0], z_limits[1]])


def plot2d_img(imgs, title_name=None, dpi=200, cmap=None, save_fig=False,
               show_fig=False, save_path=None, sub_name='0'):
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    # define number of rows and colmns
    nrows = 1
    ncols = 2
    # define subfigure heights
    heights = [50 for _ in range(nrows)]
    widths = [60 for _ in range(ncols)]
    cmaps = [['viridis', 'binary'],
             ['plasma', 'coolwarm'],
             ['Greens', 'copper']]
    # define figure size
    fig_width = 10  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    # create fig and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_width, fig_height),
                             gridspec_kw={'height_ratios': heights},
                             dpi=dpi)
    # iterate over rows
    for i in range(nrows):
        # iterate over columns
        for j in range(ncols):
            # plot image
            axes[j].imshow(imgs[j])
            # turn axes off
            axes[j].axis('off')
    # prettify image by adjusting margins
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0.01, wspace=0.01)
    # show the figure
    if show_fig:
        plt.show()
    # save figure
    if save_fig:
        # create directories
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save image
        filename = os.path.join(save_path, f"{sub_name}.png")
        fig.savefig(filename, pad_inches=0)


def plot3d_pts(pts, pts_name, s=2, dpi=150, title_name=None, sub_name='default',
               color_channel=None, colorbar=False, bcm=None, puttext=None,
               view_angle=None, save_fig=False, save_path=None, axis_off=False,
               show_fig=True):
    """Plot list of 3D points using matplotlib.

    :param pts: List of list of points to plot.
    :param pts_name: List of names of parts.
    :param s: Size of the scatter plot
    :param dpi: Resolution of the figure
    :param title_name: Title of the figure plot.
    :param sub_name: Title of the subfigure's plt
    :param color_channel: Color encoding values for each point
    :param colorbar: Whether to show color bar or not.
    :param bcm: (Not sure)
    :param puttext: (Not sure)
    :param view_angle: [elev, azim] angles for viewing 3D plot.
    :param save_fig: Wehther to save figure or not.
    :param save_path: Directory in which to save the figure at.
    :param axis_off: Whether to turn of axes or not.
    :param show_fig: Whether to display the figure in UI or not.
    """
    # check input
    assert len(pts) == len(pts_name)
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    if view_angle is None:
        view_angle = (36, -49)
    # create figure
    fig = plt.figure(dpi=dpi)
    # All possible colors for plotting points.
    top_cmap = plt.cm.get_cmap('Oranges_r', 128)
    bottom_cmap = plt.cm.get_cmap('Blues', 128)
    colors = np.vstack((top_cmap(np.linspace(0, 1, 10)),
                        bottom_cmap(np.linspace(0, 1, 10))))
    # colors = ListedColormap(newcolors, name='OrangeBlue')
    # colors  = cmap(np.linspace(0., 1., 5))
    # colors = ['Blues', 'Blues',  'Blues', 'Blues', 'Blues']

    # All possible style of plotting the points.
    all_poss = ['o', 'o', 'o', '.', 'o', '*',
                '.', 'o', 'v', '^', '>', '<',
                's', 'p', '*', 'h', 'H', 'D',
                'd', '1', '', '']
    # number of list of points
    num = len(pts)
    # iterate list of points
    for m in range(num):
        # create subplot (nrows, ncols, current_idx)
        ax = plt.subplot(1, num, m + 1, projection='3d')
        # put the viewing angle for the image
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        # plot the points in the part
        if len(pts[m]) > 1:
            for n in range(len(pts[m])):
                # check if there's any point in this part
                if pts[m][n].shape[0] == 0:
                    continue
                # if no color sepecified just plot?
                if color_channel is None:
                    ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                               marker=all_poss[n], s=s, cmap=colors[n],
                               label=pts_name[m][n])                 
                else:
                    if colorbar:
                        rgb_encoded = color_channel[m][n]
                    else:
                        # find the min and max in the list to normalize
                        color_min = np.amin(color_channel[m][n], axis=0, keepdims=True)
                        color_max = np.amax(color_channel[m][n], axis=0, keepdims=True)
                        rgb_encoded = (color_channel[m][n] - color_min) / (color_max - color_min)
                    # plot the points
                    if len(pts[m]) == 3 and n == 2:
                        p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                                       marker=all_poss[4], s=s, c=rgb_encoded,
                                       label=pts_name[m][n])
                    else:
                        p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                                       marker=all_poss[n], s=s, c=rgb_encoded,
                                       label=pts_name[m][n])
                    # show the colorbar
                    if colorbar:
                        fig.colorbar(p)
        else:
            for n in range(len(pts[m])):
                # if no color sepecified just plot?
                if color_channel is None:
                    _ = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                                   marker=all_poss[n], s=s, cmap=colors[n])
                else:
                    if colorbar:
                        rgb_encoded = color_channel[m][n]
                    else:
                        # find the min and max in the list to normalize
                        color_min = np.amin(color_channel[m][n], axis=0, keepdims=True)
                        color_max = np.amax(color_channel[m][n], axis=0, keepdims=True)
                        rgb_encoded = (color_channel[m][n] - color_min) / (color_max - color_min)
                    # plot the points
                    if len(pts[m]) == 3 and n == 2:
                        p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                                       marker=all_poss[4], s=s, c=rgb_encoded)
                    else:
                        p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2],
                                       marker=all_poss[n], s=s, c=rgb_encoded)
                    # show color bar for current
                    if colorbar:
                        fig.colorbar(p, ax=ax)
        # set axes label
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # plotting of axes
        if axis_off:
            plt.axis('off')
        # title of the plot
        if title_name is not None:
            #if len(pts_name[m]) == 1:
            #    plt.title(title_name[m] + ' ' + pts_name[m][0] + '    ')
            #else:
            #    plt.legend(loc=0)
            #    plt.title(title_name[m] + '    ')
            plt.title(title_name)
        # TODO: What?
        if bcm is not None:
            for j in range(len(bcm)):
                ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]],
                          [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]],
                          [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]],
                          'blue')

                ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]],
                          [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]],
                          [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]],
                          'gray')

                for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                    ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]],
                              [bcm[j][pair[0]][1], bcm[j][pair[1]][1]],
                              [bcm[j][pair[0]][2], bcm[j][pair[1]][2]],
                              'red')
        # TODO: What?
        if puttext is not None:
            ax.text2D(0.55, 0.80, puttext, transform=ax.transAxes, color='blue', fontsize=6)
        # set axes limit as equal for pretty
        # limits = [[-1, 1], [-1, 1], [-1, 1]]
        set_axes_equal(ax, limits=None)

    # show figure if requested
    if show_fig:
        plt.show()
    # save figure if requested
    if save_fig:
        # check existence of directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the figure
        filename = os.path.join(save_path, f'{sub_name}.png')
        fig.savefig(filename, pad_inches=0)
    # close current figure
    plt.close()


def plot_imgs(imgs, imgs_name, title_name='default',
              sub_name='default', save_path=None, save_fig=False,
              axis_off=False, show_fig=True, dpi=150):
    """Plot list of images using Matplotlib

    :param imgs: List of images to plot.
    :param imgs_name: List of image names.
    :param title_name: The title of the figure.
    :param sub_name: The titiles to the subplots.
    :param save_path: The directory to save images at.
    :param save_fig: Whether to save image or not.
    :param axis_off: Whether to display axes or not.
    :param show_fig: Whether to show plot or not.
    :param dpi: The reslution of the figure.
    :return:
    """
    # check input
    assert len(imgs) == len(imgs_name)
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    # create the figure
    fig = plt.figure(dpi=dpi)
    # get number of images
    num = len(imgs)
    # iterate over images
    for m in range(num):
        # get current axes
        ax1 = plt.subplot(1, num, m + 1)
        # plt image
        plt.imshow(imgs[m].astype(np.uint8))
        # show title
        if title_name is not None:
            plt.title(title_name[0] + ' ' + imgs_name[m])
        else:
            plt.title(imgs_name[m])
        # plotting of axes
        if axis_off:
            plt.axis('off')
    # display image
    if show_fig:
        plt.show()
    # save image
    if save_fig:
        # crteate directories if doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save image
        filename = os.path.join(save_path, f'{sub_name}_{title_name[0]}.png')
        fig.savefig(filename, pad_inches=0)
    # close the figure.
    plt.close()


def plot_arrows(points, offset=None, joint=None, whole_pts=None, title_name='default', idx=None, dpi=200, s=5,
                thres_r=0.1, show_fig=True, sparse=True, index=0, save_fig=False, save_path=None, apply_axes_lim=True):
    """
    points: [N, 3]
    offset: [N, 3] or list of [N, 3]
    joint : [P0, ll], a list, array

    """
    # default arguments to the function
    if save_path is None:
        save_path = './results/test/'
    # create the figure
    fig = plt.figure(dpi=dpi)
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'b', 'g', 'k', 'm']
    all_poss = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    num = len(points)
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=32, azim=-54)
    p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=all_poss[0], s=s)
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2], marker=all_poss[1], s=s)

    if offset is not None:
        if not isinstance(offset, list):
            offset = [offset]

        for j, offset_sub in enumerate(offset):
            if sparse:
                if idx is None:
                    ax.quiver(points[::10, 0], points[::10, 1], points[::10, 2],
                              offset_sub[::10, 0], offset_sub[::10, 1], offset_sub[::10, 2],
                              color=c_set[j])
                else:
                    points = points[idx, :]
                    ax.quiver(points[::2, 0], points[::2, 1], points[::2, 2],
                              offset_sub[::2, 0], offset_sub[::2, 1], offset_sub[::2, 2],
                              color=c_set[j])
            else:
                if idx is None:
                    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                              offset_sub[:, 0], offset_sub[:, 1], offset_sub[:, 2],
                              color=c_set[j])
                else:
                    ax.quiver(points[idx[:], 0], points[idx[:], 1], points[idx[:], 2],
                              offset_sub[idx[:], 0], offset_sub[idx[:], 1], offset_sub[idx[:], 2],
                              color=c_set[j])
    if joint is not None:
        for j, sub_j in enumerate(joint):
            length = 0.5
            sub_j[0] = sub_j[0].reshape(-1)
            sub_j[1] = sub_j[1].reshape(-1)
            ax.plot3D([sub_j[0][0] - length * sub_j[1][0], sub_j[0][0] + length * sub_j[1][0]],
                      [sub_j[0][1] - length * sub_j[1][1], sub_j[0][1] + length * sub_j[1][1]],
                      [sub_j[0][2] - length * sub_j[1][2], sub_j[0][2] + length * sub_j[1][2]],
                      c=c_set[j], linewidth=2)
    # set_axes_equal(ax)

    plt.title(title_name)
    if apply_axes_lim:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # show the figure
    if show_fig:
        plt.show()
    # save the figure
    if save_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig('{}/{}_{}.png'.format(save_path, index, title_name[0]), pad_inches=0)
    # close the figure
    plt.close()


def plot_lines(orient_vect):
    """
    orient_vect: list of [3] or None
    """
    fig = plt.figure(dpi=150)
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'b', 'g', 'k', 'm']
    all_poss = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=32, azim=-54)
    for sub_j in orient_vect:
        if sub_j is not None:
            length = 0.5
            ax.plot3D([0, sub_j[0]],
                      [0, sub_j[1]],
                      [0, sub_j[2]], 'blue', linewidth=5)
    plt.show()
    plt.close()


def plot_arrows_list(points_list, offset_list, whole_pts=None, title_name='default', dpi=200, s=5, lw=1, length=0.5,
                     view_angle=None, sparse=True, axis_off=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    fig = plt.figure(dpi=dpi)
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0., 1., 5))
    c_set = ['r', 'g', 'b', 'k', 'm']
    all_poss = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    ax = plt.subplot(1, 1, 1, projection='3d')
    if view_angle is None:
        ax.view_init(elev=36, azim=-49)
    else:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    if whole_pts is not None:
        p = ax.scatter(whole_pts[:, 0], whole_pts[:, 1], whole_pts[:, 2],
                       marker=all_poss[0], s=s)
    for i in range(len(points_list)):
        points = points_list[i]
        p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       marker=all_poss[1], s=10, cmap=colors[i + 1])
        offset = offset_list[i]
        if sparse:
            ls = 5
            ax.quiver(points[::ls, 0], points[::ls, 1], points[::ls, 2],
                      offset[::ls, 0], offset[::ls, 1], offset[::ls, 2],
                      color=c_set[i], linewidth=lw)
        else:
            ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                      offset[:, 0], offset[:, 1], offset[:, 2],
                      color='r', linewidth=lw)
    set_axes_equal(ax)
    plt.title(title_name)
    if axis_off:
        plt.axis('off')
        plt.grid('off')
    plt.show()
    plt.close()


def plot_joints_bb_list(points_list, offset_list=None, joint_list=None, whole_pts=None, bcm=None, view_angle=None,
                        title_name='default', sub_name='0', dpi=200, s=15, lw=1, length=0.5, sparse=True,
                        save_path=None, show_fig=True, save_fig=False):
    """
    points: list of [N, 3]
    offset: nested list of [N, 3]
    joint : [P0, ll], 2-order nested list, array

    """
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    if view_angle is None:
        view_angle = (36, -49)
    # create the figure
    fig = plt.figure(dpi=dpi)
    # all possible color options
    top = plt.cm.get_cmap('Oranges_r', 128)
    bottom = plt.cm.get_cmap('Blues', 128)
    colors = np.vstack((top(np.linspace(0, 1, 10)),
                        bottom(np.linspace(0, 1, 10))))
    # all possible color sets for tips
    c_set = ['g', 'b', 'm', 'y', 'r', 'c']
    # all possible markers
    all_poss = ['.', 'o', '.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    # get current axes
    ax = plt.subplot(1, 1, 1, projection='3d')
    # viewing angle of 3d plot
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    # construct parts name
    pts_name = ['part {}'.format(j) for j in range(10)]
    # show all the point sif not none
    if whole_pts is not None:
        for m, points in enumerate(whole_pts):
            p = ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=all_poss[1], s=s, cmap=colors[m],
                           label=pts_name[m])
    # compute mean of the points
    center_pt = np.mean(whole_pts[0], axis=0)
    # iterate over list of points
    for i in range(len(points_list)):
        points = points_list[i]
        # p = ax.scatter(points[:, 0], points[:, 1], points[:, 2],  marker=all_poss[i], s=s,  c='c')
        if offset_list is not None:
            offset = offset_list[i]  # with m previously
            if sparse:
                ax.quiver(points[::50, 0], points[::50, 1], points[::50, 2],
                          offset[::50, 0], offset[::50, 1], offset[::50, 2],
                          color=c_set[i])
            else:
                ax.quiver(points[:, 0], points[:, 1], points[:, 2],
                          offset[:, 0], offset[:, 1], offset[:, 2],
                          color='r')
        # we have two layers
        palette = get_tableau_palette()
        if joint_list is not None:
            if joint_list[i] is not []:
                joint = joint_list[i]  # [[1, 3], [1, 3]]
            for j, sub_j in enumerate(joint_list):
                length = 0.5
                sub_j[0] = sub_j[0].reshape(1, 3)
                sub_j[1] = sub_j[1].reshape(-1)
                ax.plot3D([sub_j[0][0, 0] - length * sub_j[1][0], sub_j[0][0, 0] + length * sub_j[1][0]],
                          [sub_j[0][0, 1] - length * sub_j[1][1], sub_j[0][0, 1] + length * sub_j[1][1]],
                          [sub_j[0][0, 2] - length * sub_j[1][2], sub_j[0][0, 2] + length * sub_j[1][2]],
                          c=c_set[j], linewidth=2)
    # set_axes_equal(ax)
    # ax.dist = 8
    print('Viewing distance is ', ax.dist)
    # TODO: What?
    if bcm is not None:
        for j in range(len(bcm)):
            color_s = 'gray'
            lw_s = 1.5
            # if j == 1:
            #     color_s = 'red'
            #     lw_s = 2plot_joints_bb_list
            ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]],
                      [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]],
                      [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]],
                      color=color_s, linewidth=lw_s)

            ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]],
                      [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]],
                      [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]],
                      color=color_s, linewidth=lw_s)

            for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]],
                          [bcm[j][pair[0]][1], bcm[j][pair[1]][1]],
                          [bcm[j][pair[0]][2], bcm[j][pair[1]][2]],
                          color=color_s, linewidth=lw_s)
    # prettyfying the plot
    plt.title(title_name, fontsize=10)
    plt.axis('off')
    plt.grid('off')
    # plt.legend('off')
    # set axes limits
    limits = [[0, 1], [0, 1], [0, 1]]
    set_axes_equal(ax, limits)
    # show figure
    if show_fig:
        plt.show()
    # save figure
    if save_fig:
        # create directories if they don't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save figure
        filename = os.path.join(save_path, f"{sub_name}_{title_name}.png")
        fig.savefig(filename, pad_inches=0)
        print(f'Saving fig into: {filename}')
    # close the figure
    plt.close()


def plot_arrows_list_threshold(points_list, offset_list, joint_list, title_name='default', dpi=200, s=5, lw=5,
                               length=0.5, threshold=0.2):
    """
    points: [N, 3]
    offset: [N, 3]
    joint : [P0, ll], a list, array

    """
    # create the figure
    fig = plt.figure(dpi=dpi)
    # all possible color options
    cmap = plt.cm.jet
    colors = cmap(np.linspace(0., 1., 5))
    # all possible color sets
    c_set = ['r', 'g', 'b', 'k', 'm']
    # all possible markers
    all_poss = ['.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '', '']
    # get current axes
    ax = plt.subplot(1, 1, 1, projection='3d')
    # plot list of points
    for i in range(len(points_list)):
        points = points_list[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=all_poss[i], s=s, c='c')
        if joint_list[i] is not []:
            for m in range(len(joint_list[i])):
                offset = offset_list[i][m]
                joint = joint_list[i][m]
                offset_norm = np.linalg.norm(offset, axis=1)
                idx = np.where(offset_norm < threshold)[0]
                ax.quiver(points[idx, 0], points[idx, 1], points[idx, 2],
                          offset[idx, 0], offset[idx, 1], offset[idx, 2],
                          color=c_set[i])
                ax.plot3D([joint[0][0, 0] - length * joint[1][0], joint[0][0, 0] + length * joint[1][0]],
                          [joint[0][0, 1] - length * joint[1][1], joint[0][0, 1] + length * joint[1][1]],
                          [joint[0][0, 2] - length * joint[1][2], joint[0][0, 2] + length * joint[1][2]],
                          linewidth=lw, c='blue')
    # show title
    plt.title(title_name)
    # show plot
    plt.show()
    # close plot
    plt.close()


def hist_show(values, labels, tick_label, axes_label, title_name,
              total_width=0.5, dpi=300, save_fig=False, sub_name='seen'):
    x = list(range(len(values[0])))
    n = len(labels)
    width = total_width / n
    colors = ['r', 'b', 'g', 'k', 'y']
    fig = plt.figure(figsize=(20, 5), dpi=dpi)
    ax = plt.subplot(111)

    for i, num_list in enumerate(values):
        if i == int(n / 2):
            plt.xticks(x, tick_label, rotation='vertical', fontsize=5)
        plt.bar(x, num_list, width=width, label=labels[i], fc=colors[i])
        if len(x) < 10:
            for j in range(len(x)):
                if num_list[j] < 0.30:
                    ax.text(x[j], num_list[j], '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
                else:
                    ax.text(x[j], 0.28, '{0:0.04f}'.format(num_list[j]), color='black', fontsize=2)
        for j in range(len(x)):
            x[j] = x[j] + width
    if title_name.split()[0] == 'rotation':
        ax.set_ylim(0, 30)
    elif title_name.split()[0] == 'translation':
        ax.set_ylim(0, 0.10)
    elif title_name.split()[0] == 'ADD':
        ax.set_ylim(0, 0.10)
    plt.title(title_name)
    plt.xlabel(axes_label[0], fontsize=8, labelpad=0)
    plt.ylabel(axes_label[1], fontsize=8, labelpad=5)
    plt.legend()
    plt.show()
    if save_fig:
        if not os.path.exists('./results/test/'):
            os.makedirs('./results/test/')
        fig.savefig('./results/test/{}_{}.png'.format(title_name, sub_name), pad_inches=0)
    plt.close()


def draw(img, imgpts, axes=None, color=None):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([1, 3, 7, 5], [3, 7, 5, 1]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 2)

    # draw pillars in blue color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip([0, 2, 6, 4], [1, 3, 7, 5]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)

    # finally, draw top layer in color
    for i, j in zip([0, 2, 6, 4], [2, 6, 4, 0]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 2)

    # draw axes
    if axes is not None:
        img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3)  ## y last

    return img


def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)

    bbox_margin = 10
    text_margin = 10

    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2 * text_margin),
                       min(bbox[2] + bbox_margin, 475 - retval[1] - 2 * text_margin))
    text_box_pos_br = (
        text_box_pos_tl[0] + retval[0] + 2 * text_margin, text_box_pos_tl[1] + retval[1] + 2 * text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)

    if draw_box:
        cv2.rectangle(draw_image,
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255, 0, 0), -1)

    cv2.rectangle(draw_image,
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0, 0, 0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255, 255, 255), thickness)

    return draw_image


def plot_distribution(d, labelx='Value', labely='Frequency',
                      title_name='Mine', dpi=200, xlimit=None,
                      put_text=False):
    fig = plt.figure(dpi=dpi)
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title_name)
    if put_text:
        plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if xlimit is not None:
        plt.xlim(xmin=xlimit[0], xmax=xlimit[1])
    plt.show()


def viz_err_distri(val_gt, val_pred, title_name):
    if val_gt.shape[1] > 1:
        err = np.linalg.norm(val_gt - val_pred, axis=1)
    else:
        err = np.squeeze(val_gt) - np.squeeze(val_pred)
    plot_distribution(err, labelx='L2 error', labely='Frequency',
                      title_name=title_name, dpi=160)


def plot3d_pts_in_camera_plane(pts, proj_matrix, title_name=None, pts_name=None,
                               show_fig=False, save_fig=False, axis_off=True, im_size=(512, 512),
                               s=1, save_path=None, filename=None, dpi=200):
    """
    Projects a list of points into a 2D plane.

    Modified (Tianxu): support pts with inverted y-axis (y-aixs pointing downwards and x-axis pointing to the right)

    :param pts: List of list of points to plot.
    :param pts_name: List of names of parts.
    :param proj_matrix: A 4 x 4 projection matrix for the camera.
    :param dpi: Resolution of the figure
    :param title_name: Title of the figure plot.
    :param im_size: Size of the image in camera plane.
    :param axis_off: Whether to turn of axes or not.
    :param show_fig: Whether to display the figure in UI or not.
    :param save_fig: Wehther to save figure or not.
    :param save_path: Directory in which to save the figure at.
    :param filename: Name of the saved file.
    """
    # check input
    if pts_name is not None:
        assert len(pts) == len(pts_name)
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    if filename is None:
        filename = 'points'
    # create figure
    fig = plt.figure(dpi=dpi)
    # iterate over list of points
    for part_index, pts_on_part in enumerate(pts):
        # Skip the parts without any point
        if pts_on_part.shape[0] != 0:
            # convert points to homogenous
            points_homo = np.pad(pts_on_part, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
            # project points to 2d
            points_cam = np.dot(points_homo, proj_matrix.T)
            points_cam = points_cam / (points_cam[..., -1:])
            points_cam[..., :2] = 0.5 * (points_cam[..., :2] + 1)
            # convert to pixels
            im_pixels = points_cam[..., :2]
            im_pixels[:, 0] = im_size[0] * im_pixels[:, 0]
            im_pixels[:, 1] = im_size[1] * im_pixels[:, 1]
            im_pixels = np.asarray(im_pixels, dtype=np.int)
            # plot points into image
            # we invert the y-axis since scatter is used and not imshow
            if pts_name is not None:
                plt.scatter(im_pixels[:, 0], im_pixels[:, 1], label=pts_name[part_index], s=s)
            else:
                plt.scatter(im_pixels[:, 0], im_pixels[:, 1], s=s)
        else:
            empty_array = np.array([])
            if pts_name is not None:
                plt.scatter(empty_array, empty_array, label=pts_name[part_index], s=s)
            else:
                plt.scatter(empty_array, empty_array, s=s)

    # plotting of axes
    if axis_off:
        plt.axis('off')
    plt.axis('equal')
    plt.xlim(0, im_size[0])
    plt.ylim(0, im_size[1])
    # title of the plot
    if title_name is not None:
        if pts_name is not None:
            if len(pts_name) == 1:
                plt.title(title_name + ' ' + pts_name[0])
            else:
                plt.legend(loc=0)
                plt.title(title_name)
        else:
            plt.title(title_name)
    # show figure if requested
    if show_fig:
        plt.show()
    # save figure if requested
    if save_fig:
        # check existence of directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save figure
        filename = os.path.join(save_path, f'{filename}.png')
        fig.savefig(filename, pad_inches=0)
    # close current figure
    plt.close()


def plot3d_joints_in_camera_plane(pts, joints, proj_matrix, joint_angles=[None],
                                  title_name='default', pts_name=None,
                                  show_fig=False, save_fig=False, axis_off=True,
                                  im_size=(512, 512), s=1, dpi=200, arrow_scale=1.0,
                                  save_path=None, filename=None):
    # check input
    if pts_name is not None:
        assert len(pts) == len(pts_name)
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    if filename is None:
        filename = 'joints'
    # create figure
    fig = plt.figure(dpi=dpi)
    # iterate over list of points
    for part_index, pts_on_part in enumerate(pts):
        # convert points to homogenous
        points_homo = np.pad(pts_on_part, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        # project points to 2d
        points_cam = np.dot(points_homo, proj_matrix.T)
        points_cam = points_cam / (points_cam[..., -1:])
        points_cam[..., :2] = 0.5 * (points_cam[..., :2] + 1)
        # convert to pixels
        im_pixels = points_cam[..., :2]
        im_pixels[:, 0] = im_size[0] * im_pixels[:, 0]
        im_pixels[:, 1] = im_size[1] * im_pixels[:, 1]
        im_pixels = np.asarray(im_pixels, dtype=np.int)
        # plot points into image
        # we invert the y-axis since scatter is used and not imshow
        if pts_name is not None:
            plt.scatter(im_pixels[:, 0], im_pixels[:, 1], label=pts_name[part_index], s=s)
        else:
            plt.scatter(im_pixels[:, 0], im_pixels[:, 1], s=s)

    # iterate over joints
    for joint in joints:
        # convert to homogenous coordinates
        # joint: [3x1 joint pt, 3x1 joint axis]
        augmented_joint_pos = np.pad(joint[0], ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        augmented_joint_axis = np.pad(joint[1], ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        # project points to 2d
        proj_joint_pos = np.dot(augmented_joint_pos, proj_matrix.T)
        proj_joint_axis = np.dot(augmented_joint_axis, proj_matrix.T)
        # dehomogenize the points
        proj_joint_pos = proj_joint_pos / (proj_joint_pos[..., -1:])
        proj_joint_axis = proj_joint_axis / (proj_joint_axis[..., -1:])
        # convert to im frame
        proj_joint_pos[..., :2] = 0.5 * (proj_joint_pos[..., :2] + 1)
        # convert joint position to pixels
        im_joint_pos = proj_joint_pos[..., :2]
        im_joint_pos[:, 0] = im_size[0] * im_joint_pos[:, 0]
        im_joint_pos[:, 1] = im_size[1] * im_joint_pos[:, 1]
        im_joint_pos = np.asarray(im_joint_pos, dtype=np.int)
        # convert joint axis to pixels
        im_joint_axis = proj_joint_axis[..., :2]
        im_joint_axis[:, 0] = im_size[0] * im_joint_axis[:, 0]
        im_joint_axis[:, 1] = im_size[1] * im_joint_axis[:, 1]
        im_joint_axis = np.asarray(im_joint_axis, dtype=np.int)
        # draw arrow
        plt.quiver(im_joint_pos[:, 0], im_joint_pos[:, 1],
                   -im_joint_axis[:, 0], -im_joint_axis[:, 1],
                   color='r', scale_units='xy', scale=arrow_scale)

    # plotting of axes
    if axis_off:
        plt.axis('off')
    plt.xlim((0, im_size[0]))
    plt.ylim((0, im_size[1]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # title of the plot
    if title_name is not None:
        if pts_name is not None:
            if len(pts_name) == 1:
                plt.title(title_name + ' ' + pts_name[0])
            else:
                plt.legend(loc=0)
                fig.suptitle(title_name)
        else:
            fig.suptitle(title_name)
    #plt.legend([F'Joint {i+1} angle: {joint_angle:.2f} degree' for i, joint_angle in enumerate(joint_angles)])
    # show figure if requested
    if show_fig:
        plt.show()
    # save figure if requested
    if save_fig:
        # check existence of directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save figure
        filename = os.path.join(save_path, f'{filename}.png')
        fig.savefig(filename, pad_inches=0)
    # close current figure
    plt.close()


def plot3d_joints_with_rgbd(rgb, depth, pts, joints, proj_matrix, joint_angles=[None],
                            title_name='default', pts_name=None,
                            show_fig=False, save_fig=False, axis_off=True,
                            im_size=(512, 512), s=1, dpi=200, arrow_scale=1.0,
                            save_path=None, filename=None):
    # check input
    if pts_name is not None:
        assert len(pts) == len(pts_name)
    # default variables to the functions
    if save_path is None:
        save_path = './results/test/'
    if filename is None:
        filename = 'joints'
    # create figure
    fig = plt.figure(dpi=dpi)
    # plot rgb
    _ = plt.subplot(131)
    plt.imshow(rgb)
    plt.title('RGB')
    plt.axis('off')
    # plt depth
    _ = plt.subplot(132)
    plt.imshow(depth)
    plt.title('Depth')
    plt.axis('off')
    # plt joints
    _ = plt.subplot(133)
    # iterate over list of points
    for part_index, pts_on_part in enumerate(pts):
        # convert points to homogenous
        points_homo = np.pad(pts_on_part, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        # project points to 2d
        points_cam = np.dot(points_homo, proj_matrix.T)
        points_cam = points_cam / (points_cam[..., -1:])
        points_cam[..., :2] = 0.5 * (points_cam[..., :2] + 1)
        # convert to pixels
        im_pixels = points_cam[..., :2]
        im_pixels[:, 0] = im_size[0] * im_pixels[:, 0]
        im_pixels[:, 1] = im_size[1] * im_pixels[:, 1]
        im_pixels = np.asarray(im_pixels, dtype=np.int)
        # plot points into image
        # we invert the y-axis since scatter is used and not imshow
        if pts_name is not None:
            plt.scatter(im_pixels[:, 0], -im_pixels[:, 1], label=pts_name[part_index], s=s)
        else:
            plt.scatter(im_pixels[:, 0], -im_pixels[:, 1], s=s)

    # iterate over joints
    for joint in joints:
        # convert to homogenous coordinates
        # joint: [3x1 joint pt, 3x1 joint axis]
        augmented_joint_pos = np.pad([joint[0]], ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        augmented_joint_axis = np.pad([joint[1]], ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
        # project points to 2d
        proj_joint_pos = np.dot(augmented_joint_pos, proj_matrix.T)
        proj_joint_axis = np.dot(augmented_joint_axis, proj_matrix.T)
        # dehomogenize the points
        proj_joint_pos = proj_joint_pos / (proj_joint_pos[..., -1:])
        proj_joint_axis = proj_joint_axis / (proj_joint_axis[..., -1:])
        # convert to im frame
        proj_joint_pos[..., :2] = 0.5 * (proj_joint_pos[..., :2] + 1)
        # convert joint position to pixels
        im_joint_pos = proj_joint_pos[..., :2]
        im_joint_pos[:, 0] = im_size[0] * im_joint_pos[:, 0]
        im_joint_pos[:, 1] = im_size[1] * im_joint_pos[:, 1]
        im_joint_pos = np.asarray(im_joint_pos, dtype=np.int)
        # convert joint axis to pixels
        im_joint_axis = proj_joint_axis[..., :2]
        im_joint_axis[:, 0] = im_size[0] * im_joint_axis[:, 0]
        im_joint_axis[:, 1] = im_size[1] * im_joint_axis[:, 1]
        im_joint_axis = np.asarray(im_joint_axis, dtype=np.int)
        # draw arrow
        plt.quiver(im_joint_pos[:, 0], -im_joint_pos[:, 1],
                   im_joint_axis[:, 0], -im_joint_axis[:, 1],
                   color='r', scale_units='xy', scale=arrow_scale)
    # title of the plot
    if title_name is not None:
        if pts_name is not None:
            if len(pts_name) == 1:
                plt.title(title_name + ' ' + pts_name[0])
            else:
                plt.legend(loc=0)
                plt.title(title_name)
        else:
            plt.title(title_name)
    # plotting of axes
    if axis_off:
        plt.axis('off')
    plt.xlim((0, im_size[0]))
    plt.ylim((-im_size[1], 0))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    plt.legend([F'Joint {i+1} angle: {joint_angle:.2f} degree' for i, joint_angle in enumerate(joint_angles)], prop={'size': 5})
    # show figure if requested
    if show_fig:
        plt.show()
    # save figure if requested
    if save_fig:
        # check existence of directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save figure
        filename = os.path.join(save_path, f'{filename}.png')
        fig.savefig(filename, pad_inches=0)
    # close current figure
    plt.close()


if __name__ == '__main__':
    d = np.random.laplace(loc=15, scale=3, size=500)
    plot_distribution(d)

# EOF
