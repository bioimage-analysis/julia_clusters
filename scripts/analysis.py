import pandas as pd
import numpy as np

def create_dataframe(ganglion_prop, local_maxi, meta, directory, save=False):
    result = []
    for prop in ganglion_prop:
        coord = prop.coords
        #find neurons in ganglion
        a = np.in1d(local_maxi[:,0], coord[:,0])
        b = np.in1d(local_maxi[:,1], coord[:,1])
        c = np.where(a == True)[0]
        d = np.where(b == True)[0]
        e = np.in1d(c,d)
        number_of_neurons = len(np.where(e == True)[0])
        label = prop.label
        result.append((label, number_of_neurons))
    result = np.asarray(result)
    surface_gang = [prop.area for prop in ganglion_prop]
    major_gang = [prop.major_axis_length for prop in ganglion_prop]
    minor_gang = [prop.minor_axis_length for prop in ganglion_prop]
    orientation_gang = [prop.orientation for prop in ganglion_prop]

    df = pd.DataFrame(result, columns = ['ganglion', 'Nbr of neurons'])
    df["surface ganglion"] = surface_gang
    df["major axis length"] = major_gang
    df["minorm axis length"] = minor_gang
    df["orientation"] = orientation_gang
    df.loc[:, 'major axis length']*=(meta['PhysicalSizeX'])
    df.loc[:, 'minorm axis length']*=(meta['PhysicalSizeX'])
    df = df.rename(columns={"minorm axis length": "minor axis length in µm"})
    df = df.rename(columns={"major axis length": "major axis length in µm"})
    df = df.rename(columns={"surface ganglion": "surface ganglion in µm2"})
    df = df.replace({"ganglion" : 0},"background")

    df.loc[-1] =  np.nan
    df.index = df.index +1
    df = df.sort_index()
    df['ganglion'][0] = "background"
    df.loc[0,"Nbr of neurons"] = len(local_maxi)

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))

    return(df)
