import bioformats
import numpy as np
import os
import time


def _metadata(path):
    xml = bioformats.get_omexml_metadata(path)
    md = bioformats.omexml.OMEXML(xml)

    meta={'AcquisitionDate': md.image().AcquisitionDate}
    meta['Name']=md.image().Name.replace(' ', '_')
    meta['SizeC']=md.image().Pixels.SizeC
    meta['SizeT']=md.image().Pixels.SizeT
    meta['SizeX']=md.image().Pixels.SizeX
    meta['SizeY']=md.image().Pixels.SizeY
    meta['SizeZ']=md.image().Pixels.SizeZ
    meta['PhysicalSizeX'] = md.image().Pixels.PhysicalSizeX
    meta['PhysicalSizeY'] = md.image().Pixels.PhysicalSizeY
    meta['PhysicalSizeZ'] = md.image().Pixels.PhysicalSizeZ

    return(meta)



def load_bioformats(path):
    meta = _metadata(path)
    image = np.empty((meta['SizeT'], meta['SizeZ'], meta['SizeX'], meta['SizeY'], meta['SizeC']))

    with bioformats.ImageReader(path) as rdr:
        for t in range(0, meta['SizeT']):
            for z in range(0, meta['SizeZ']):
                for c in  range(0, meta['SizeC']):
                    image[t,z,:,:,c]=rdr.read(c=c, z=z, t=t, series=None,
                                                 index=None, rescale=False, wants_max_intensity=False,
                                                 channel_names=None)
    img = np.squeeze(image)
    if img.ndim == 3:
        mip = np.amax(img, 0)
    elif img.ndim == 4:
        mip = np.amax(img[:,:,:,1], 0)
    else:
        print("need to correct for ndim > 3")

    directory = _new_directory(path, meta)

    return(img, directory, mip, meta)

def _new_directory(path, meta):

    directory = os.path.dirname(path)+"/"+"result"+'_'+meta["Name"]+'_'+ time.strftime('%m'+'_'+'%d'+'_'+'%Y')
    if os.path.exists(directory):
        expand = 0
        while True:
            expand += 1
            new_directory = directory+"_"+str(expand)
            if os.path.exists(new_directory):
                continue
            else:
                directory = new_directory
                os.makedirs(directory)
                break
    else:
        os.makedirs(directory)
    return(directory)
