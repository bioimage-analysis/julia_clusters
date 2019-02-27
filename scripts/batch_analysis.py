import javabridge
import bioformats
import scripts.io as io
import scripts.processing as processing
import scripts.analysis as analysis
import os

def batch_analysis(path, **kwargs):

    """Go through evry image files in the directory (path).
    Parameters
    ----------
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat
    """


    imageformat= kwargs.get('imageformat', '.nd2')
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]

    for file in imfilelist:

        javabridge.start_vm(class_path=bioformats.JARS)
        _, series = io._metadata(file)
        for serie in range(series):

            _, directory, mip, meta = io.load_bioformats(file, serie = serie)
            local_maxi, labels, gauss = processing.clusters(mip, plot=False)
            del mip
            ganglion_prop = processing.segmentation(gauss, local_maxi, labels, meta,
                                                    directory, plot = False,save = True)
            del gauss
            analysis.create_dataframe(ganglion_prop, local_maxi, meta, directory, save=True)
