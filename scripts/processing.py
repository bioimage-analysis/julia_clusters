from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
import numpy as np
from sklearn.cluster import KMeans
from skimage import feature
from skimage.morphology import dilation, disk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage import filters

def background_correct(img):

    gauss = gaussian(img, sigma=5)

    #size = 200
    #background = uniform_filter(gauss, size)
    #img_cor = img - background

    return(gauss)

def _binarization(image):
    data = image.ravel().reshape(1,-1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(data.T)
    binary = kmeans.labels_.reshape(image.shape)
    #cluster can be "reverse" background = 1 and foreground = 0
    if np.count_nonzero(binary) > np.count_nonzero(1 - binary):
        binary = 1-binary
    # Adding some erosion to be more conservative
    #binary = morphology.opening(binary, morphology.ball(2))
    return(binary)

def clusters(img, sigma=9, min_distance =27, eps=0.15, min_samples = 6,plot = True):

    gauss = gaussian(img, sigma=sigma)

    thresh = filters.threshold_otsu(gauss)
    local_maxi = feature.peak_local_max(gauss, min_distance = min_distance,
                                        threshold_abs = thresh + (10*thresh)/100,
                                        exclude_border=False)
    X = local_maxi
    X = StandardScaler().fit_transform(X)

    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    if plot:
        label_plot = np.copy(labels)
        label_plot[labels == 0] = max(labels)+1
        label_plot[labels == -1] = 0

        fig, (ax) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8),
                               sharex=True, sharey=True)
        ax[0].imshow(gauss, alpha=0.6)
        ax[0].scatter(local_maxi[:,1], local_maxi[:,0], s=4)

        ax[1].imshow(gauss, alpha=0.3)
        ax[1].scatter(local_maxi[:,1], local_maxi[:,0], c=label_plot, cmap = "nipy_spectral")

        for axes in ax:
            axes.axis('off')

    return (local_maxi, labels, gauss)

def segmentation(img, local_maxi, labels, meta, directory, plot=True, save=False):

    only_clusters = np.zeros(img.shape, dtype=np.int)
    for pos, new in zip(local_maxi, labels):
        if new > 0:
            only_clusters[pos[0], pos[1]] = new
        elif new == 0:
            only_clusters[pos[0], pos[1]] = max(labels) + 1
    only_clusters = dilation(only_clusters, disk(10))

    binary = _binarization(img)

    dist_water = ndi.distance_transform_edt(binary)
    segmentation_ws = watershed(-img, only_clusters, mask = binary)

    ganglion_prop = regionprops(segmentation_ws)

    if plot == True:

        image_label_overlay = label2rgb(segmentation_ws, image=img.astype('uint16'),
                                        bg_label=0)

        fig,ax = plt.subplots(1,1, figsize=(16,16))
        ax.imshow(image_label_overlay, interpolation='nearest')
        ax.axis('off')
        ax.scatter(local_maxi[:, 1], local_maxi[:, 0], c='red',s = 10)


        for prop in ganglion_prop:
            ax.annotate("ganglion_{}".format(prop.label),
                        (prop.centroid[1]-150, prop.centroid[0]-250), color='green',
                        fontsize=15, weight = "bold")

        if save:
            try:
                filename = meta['Name']+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
    elif plot == False:

        image_label_overlay = label2rgb(segmentation_ws, image=img.astype('uint16'),
                                        bg_label=0)
        plt.ioff()

        fig,ax = plt.subplots(1,1, figsize=(16,16))
        ax.imshow(image_label_overlay, interpolation='nearest')
        ax.axis('off')
        ax.scatter(local_maxi[:, 1], local_maxi[:, 0], c='red',s = 10)


        for prop in ganglion_prop:
            ax.annotate("ganglion_{}".format(prop.label),
                        (prop.centroid[1]-150, prop.centroid[0]-250), color='green',
                        fontsize=15, weight = "bold")

        if save:
            try:
                filename = meta['Name']+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
        plt.close(fig)

    return(ganglion_prop)
