
import os
from utils.data_directory_manager import DataDirectoryManager
from landsatUtil.landsat.downloader import Downloader
from utils.image_correction import LandsatTOACorrecter
from utils.raster_tools import *
from utils.rasterize import rasterize_label, save_raster
from models.antarctic_rock_outcrop_os import OutcropLabeler
from os import listdir
from os.path import isfile, join
from _thread import *
import threading 
import matplotlib.pyplot as plt
import rasterio.plot as rplt
from progress.bar import FillingSquaresBar
 


import numpy as np



"""
To use script, change base_dir to the directory where you intend to store your images
"""
print_lock = threading.Lock() 
un_compressed_data = []


def untar_helper(threadName, scene_IDs, chunkNum, dm, totalThreads):    
    lenList = len(scene_IDs)
    chunkSize = int(lenList/totalThreads);
    start_chunk = chunkNum*chunkSize
    
    if(chunkNum + 1 == totalThreads):
        end_chunk = lenList
    else:
        end_chunk = (chunkNum+1)*chunkSize
            
    scene_chunk = scene_IDs[start_chunk : end_chunk]
    for s in scene_chunk:
        if s in un_compressed_data:
            print(str(s) + " is already un-tarred")
            continue
        print("Thread: " + threadName + " Untarring: " + s)
        try:
            dm.untar_scenes([s])
            print_lock.acquire()
            un_compressed_data.append(s)
            print_lock.release()
        except:
            continue
    return

if __name__ == "__main__":
    
    base_dir = "/home/dsa/ant"
    dm = DataDirectoryManager(base_dir)
    
    dm.download_supplement()
    dm.extract_supplement_files()
    
    dataPath= dm.download_dir
    rawPath = dm.raw_image_dir
    stackedPath = os.path.join(dm.project_dir, "stacked_chunks")
    
    scene_IDs = []
    scene_IDs = [i["ID"] for i in dm.load_scene_ids()[:4]]
    expected_tar_files = [f + ".tar.bz" for f in scene_IDs]
    
    # tarred file path have .tar.bz extension so need to use splitext twice
    tar_files = [os.path.splitext(os.path.splitext(f)[0])[0] 
                 for f in os.listdir(dm.download_dir)]
    undownloaded_scenes = [i for i in scene_IDs if i not in tar_files]
    if undownloaded_scenes:
        downloader = Downloader(download_dir=dm.download_dir)
        dl_bands = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
        downloader.download(undownloaded_scenes)
    
    dataFiles = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    fName = [i.split(".")[0].replace("'", "") for i in dataFiles] 

    for s in scene_IDs:
        if s not in fName:
            scene_IDs.remove(s)
    print("Downloaded SceneIDs: ")
    print(scene_IDs)
    
    un_compressed_data = [i for i in os.listdir(dm.raw_image_dir)
                          if os.path.isdir(os.path.join(dm.raw_image_dir, i))]

    #Load Already Compressed Files
    
    print("Already Untarred Files: ")
    print(un_compressed_data)
    try:
        t1 = threading.Thread( target = untar_helper, args = ("Thread-1", scene_IDs, 0, dm, 4, ) )
        t2 = threading.Thread( target = untar_helper, args = ("Thread-2", scene_IDs, 1, dm, 4, ) )
        t3 = threading.Thread( target = untar_helper, args = ("Thread-3", scene_IDs, 2, dm, 4, ) )
        t4 = threading.Thread( target = untar_helper, args = ("Thread-4", scene_IDs, 3, dm, 4, ) )
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
    except:
       print("Error: unable to start thread")
    #Store Compressed File 
#     with open(rawPath+ '/raw_file.txt', 'w') as filehandle:
#         for listitem in un_compressed_data:
#             filehandle.write('%s\n' % listitem)
            
    # After Uncompressing the Images #
    rawFiles = [f for f in listdir(rawPath)]
    rawFName = [i.split(".")[0].replace("'", "") for i in rawFiles]
    
         
    # Create Chunks from Raw Images #
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
    rasters = []
    
    if not os.path.exists(stackedPath):
        os.mkdir(stackedPath)
        
    for r in un_compressed_data:
        if os.path.exists(os.path.join(stackedPath, r)):
            print("Scene {} already processed".format(r))
            continue
        # Generating the label of raw_file r
        print("Processing image {}".format(r))
        
        band_2_path = os.path.join(dm.raw_image_dir, r, r + "_B2.TIF")
        label_path = os.path.join(dm.label_dir, r + "_label.TIF")
        if not os.path.exists(label_path):
            try:
                label, meta = rasterize_label(band_2_path, dm.outcrop_shape_path)
                
                meta['dtype'] = label.dtype
                meta['count'] = 1
                with rio.open(label_path, 'w', **meta) as dst:
                    dst.write(label)
                    
            except ValueError as e:
                with rio.open(band_2_path) as rockless:
                    meta = rockless.meta.copy()
                    zero_shape = (meta["width"], meta["height"])
                label = np.zeros(zero_shape, np.int8)
                print("label for image {} has no overlapping shapes".format(band_2_path))
                
                meta['dtype'] = label.dtype
                meta['count'] = 1
                with rio.open(label_path, 'w', **meta) as dst:
                    dst.write(label, 1)
        
        with rio.open(label_path) as label_file:
            label = label_file.read(1)
        # label = plt.imread(label_path)
        print(label.shape)
        
        # Label Created
        
        
        cnt = 0;
        rasters.clear()
        for b in bands:
            imgName = os.path.join(dm.raw_image_dir, r + "/" + r + "_"+ b + ".TIF")
            raster = plt.imread(imgName)
#             print(cnt, raster.shape)
            cnt = cnt+1
            rasters.append(raster)

        stacked_rasters = np.stack(rasters, axis=0).transpose(1,2,0)
        print("Shape of Stacked Images", stacked_rasters.shape)
        stacked_chunks = [] 
        imgSize = min(stacked_rasters.shape[0], stacked_rasters.shape[1])
        chunkSize = 512
        numChunks  = int(imgSize / chunkSize)
        cnt = 0
        dirName = stackedPath+"/" + r
        print("Creating Directory: " , dirName)
        if not os.path.exists(dirName):
           
            os.mkdir(dirName)
        with FillingSquaresBar('Processing', max=numChunks*numChunks) as bar:
            for i in range(0,numChunks):
                for j in range(0, numChunks):
#                     print("Chunk ({},{})".format(i, j))
                    hstIndex = i*chunkSize
                    hendIndex = (i+1)*chunkSize
                    vstIndex = j*chunkSize
                    vendIndex = (j+1)*chunkSize
                    chunk = stacked_rasters[hstIndex:hendIndex, vstIndex:vendIndex, :]  
                    label_chunk = label[hstIndex:hendIndex, vstIndex:vendIndex] 
                    fil = np.where(chunk>0)
                    nZP = np.sum(fil)
                    fileName = os.path.join(dirName, "chunk_{}_{}.npy".format(i,j))
                    labelFileName = os.path.join(dirName, "chunk_{}_{}_label.npy".format(i,j))
                    if(nZP > 0):
                        np.save(fileName, chunk, allow_pickle = True)
                        np.save(labelFileName, label_chunk, allow_pickle = True)
        #                 print(cnt, chunk.shape)
                    cnt = cnt +1
                    bar.next()
