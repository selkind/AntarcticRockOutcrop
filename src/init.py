
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
from progress.bar import Bar
 


import numpy as np
dataPath= "data/downloads"
rawPath = "data/raw"
stackedPath = "data/stacked_chunks"


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
    base_dir = os.getcwd()
    dm = DataDirectoryManager(os.path.join(base_dir,"data"))
    '''
    dm.download_supplement() # where we get the zip file
    '''
    dm.extract_supplement_files() # where we get the sceneID .txt and outcrop .shp
    
    scene_IDs = []
    scene_IDs = [i["ID"] for i in dm.load_scene_ids()[:31]]
   
    dataFiles = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
    fName = [i.split(".")[0].replace("'", "") for i in dataFiles] 
    for s in scene_IDs:
        if s not in fName:
            scene_IDs.remove(s)

    print("Downloaded SceneIDs: ")
    print(scene_IDs)
    #Load Already Compressed Files
    with open(rawPath+ '/raw_file.txt', 'r') as filehandle:
        for line in filehandle:
            item = line[:-1]
            un_compressed_data.append(item)
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
    with open(rawPath+ '/raw_file.txt', 'w') as filehandle:
        for listitem in un_compressed_data:
            filehandle.write('%s\n' % listitem)
            
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
            imgName = "data/raw/"+r + "/" + r + "_"+ b + ".TIF"
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
        with Bar('Processing', max=numChunks*numChunks) as bar:
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
            
         

        
#     for s in scenes:
#         scene_IDs.append(s)
#     print(scene_IDs)
#     scene_IDs_raw = os.path.join(dm.raw_image_dir, scene_IDs[0])
#     test_scene_corrected = os.path.join(dm.corrected_image_dir, scene_IDs[0])
    
    
#     print(dm.download_dir)
    
#     scene_downloader = Downloader(download_dir=dm.download_dir)
#     scene_downloader.download(scene_IDs)
#     
    
#     correcter = LandsatTOACorrecter(scene_IDs_raw)
#     correcter.correct_toa_brightness_temp(dm.corrected_image_dir)
#     correcter.correct_toa_reflectance(dm.corrected_image_dir)
    
#     labeler = OutcropLabeler(test_scene_corrected, coast_shape_file)
#     labeler.write_mask_file(dir_manager.label_dir)
# =======
#     data_dir = os.path.join(base_dir,"data")
#     manual_dir = "/home/dsa/DSA/images_manual"
    
#     dm = DataDirectoryManager(manual_dir)
# >>>>>>> f8f3292aab92ba89203161b3f1bdf14f78d3a094

#     dm.download_supplement()
#     dm.extract_scene_id_file()
    
#     dm.download_coast_shapefile()
#     dm.extract_coast_shapefile()
