import matplotlib.patches as patches
from matplotlib import colorbar as cbar

def plot_anno_pred(anno_img, img, pred, save_path=None):
    min_det_score = 0.5
    boxes, scores = pred[:, :4].astype(np.int), pred[:, 4]
    valid_idx = scores >= min_det_score
    boxes = boxes[valid_idx]; scores = scores[valid_idx]
    fig = plt.figure(figsize=(30, 15))
    fig.add_subplot(121)
    plt.imshow(anno_img[:,:,::-1])
    fig.add_subplot(122)
    ax = plt.gca()
    plt.imshow(img[:,:,::-1])
    # cmap=plt.cm.Wistia
    cmap = plt.cm.spring
    normal = plt.Normalize(0.5, max(scores))
    colors = cmap(scores)
    for box, c in zip(boxes, colors):
        rect=patches.Rectangle(box[:2],*box[2:],linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(rect)
#     cax, _ = cbar.make_axes(ax)
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
    pad_fraction = 0.5; aspect = 20
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cb2 = cbar.ColorbarBase(cax, cmap=cmap, norm=normal)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
