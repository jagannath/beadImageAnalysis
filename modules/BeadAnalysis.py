
# coding: utf-8

# In[12]:

        





#!/home/abardo/anaconda2/bin/python
class BeadAnalysis(self,exppath,outpath,saveimgs,groupby,saveto,h1,h2):

    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image
    from tqdm import tqdm
    import os
    import sys
    import cv2
    import csv
    from nd2reader import ND2Reader as nd2
    from statistics import mean 
    from statistics import stdev
    from statistics import median
    from skimage import io
    
    def loaded(self): print("BeadAnalysis")
  

    def wintopy(self,filepath):
        filepath = filepath.replace("\\", "/")
        if str(filepath[len(filepath)-2:]) ==  "//":
            filepath = filepath[:-1]
        return filepath
    def plot(self,plotimage):
        plt.imshow(plotimage,cmap='gray')
        plt.show()
        return
    def psave(self,plotimage,path):
        """plt.imshow(plotimage,cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.savefig(path)"""
        io.imsave(path,plotimage)
        return

    def plotmany(self,imagearray):
        fig=plt.figure(figsize=(16, 16))
        rows= 1
        columns = len(imagearray)
        for i in range(1, columns*rows +1):
            img = imagearray[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img,cmap='gray',vmin=0,vmax=35500)
        plt.show()

    """#### default inputs
    exppath= r"/stor/work/Marcotte/project/Bardo/OtherData/Beads/Test/"
    outpath=""
    saveimgs=True
    #groupby="field"
    groupby="filename"
    saveto="bead_analysis_out/"
    ###"""

    #values for 2048X2048
    thickness = 30
    half=(thickness/2)
    widthmax= 170
    widthmin= 100
    ###set paths
    if outpath == "":
        outpath= exppath
    outpath=outpath+saveto
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    exppath=wintopy(exppath)

    ### get all input filename
    ifilel = [f for f in os.listdir(exppath) if os.path.isfile(os.path.join(exppath, f))]
    ifile2=[]
    ###loop for all nd2 files
    beaddata=[]
    writedatac3=[]
    writedatac4=[]

    impath=outpath+'images/'
    if not os.path.exists(impath):
        os.makedirs(impath)
    for xifl in ifilel:
        if xifl[-3:]== 'nd2':
            ifile2.append(xifl)

    for xifl in ifile2[:1]:
        fullpath=exppath+xifl
        fname=xifl.replace(".nd2","")
        header=['Filename', 'Frame ID', 'Mask ID', 'oRadius(pix)','Mask X', 'Mask Y','oCircle Area pix^2', 'Ring Area pix ^2', 'iCircle Area pix ^2']
        #header2=['Filename', 'Frame ID','Full Image Circle Mask Area pix^2', 'Full Image Ring Mask Area pix ^2']
        with nd2(fullpath) as images:
            fnum=len(images.metadata['fields_of_view'])
            cnum=len(images.metadata['channels'])
            flst=(images.metadata['fields_of_view'])
            clst=(images.metadata['channels'])
            #allmeta=(images._parser._raw_metadata.image_metadata_sequence)
            #print(images._parser._raw_metadata.image_calibration)
        for rc in range(cnum):
            header.append('oCircle I: '+str(clst[rc]))
            header.append('Ring I: '+str(clst[rc]))
            header.append('iCircle I: '+str(clst[rc]))
            header.append('oCircle mI: '+str(clst[rc]))
            header.append('Ring Mask mI: '+str(clst[rc]))
            header.append('iCircle mI: '+str(clst[rc]))
            header.append('oCircle MI: '+str(clst[rc]))
            header.append('Ring MI: '+str(clst[rc]))
            header.append('iCircle MI: '+str(clst[rc]))
            header.append('RingMI/CircMI: '+str(clst[rc]))
    with open((outpath+"bybead"+'.csv'), mode='w') as writefile:
        writefile_writer = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writefile_writer.writerow(header)

    allcirc=[]

    y=8
    outnum=[]
    for x in range(cnum-1):
        y=y+4
        for z in range (4):
            outnum.append(y+z+1) 

    for iif, xifl in enumerate(ifile2):
        fullpath=exppath+xifl
        fname=xifl.replace(".nd2","")
        for rf in tqdm(range(fnum),desc="File"+str(iif)+"/"+str(len(ifile2))):
            imagei=[]
            writedatac=[]
            with nd2(fullpath) as images:
                for rc in range(cnum):
                    imagei.append(images.get_frame_2D(c=rc, v=rf))
            #plotmany(imagei)
            orig=imagei[0]

            ###find beads
            #blur = cv2.medianBlur(orig.astype(np.float32),5)
            
            blur =orig.astype(np.float32)
            blur2= (blur/256).astype('uint8')
            circMask = np.zeros(orig.astype(np.float32).shape,dtype=np.uint16)
            ringMask = np.zeros(orig.astype(np.float32).shape,dtype=np.uint16)
            circimgt= (orig/256).astype('uint8')
            circimg=circimgt
            #circles = (cv2.HoughCircles(blur2,cv2.HOUGH_GRADIENT,dp=1,minDist=200,param1=20,param2=70,minRadius=100,maxRadius=1000))                   
            #circles = (cv2.HoughCircles(blur2,cv2.HOUGH_GRADIENT,dp=1,minDist=200,param1=60,param2=100,minRadius=100,maxRadius=1000))                   
            #circles = (cv2.HoughCircles(blur2,cv2.HOUGH_GRADIENT,dp=1,minDist=250,param1=40,param2=90,minRadius=100,maxRadius=700))                   
            #circles = (cv2.HoughCircles(blur2,cv2.HOUGH_GRADIENT,dp=1,minDist=250,param1=40,param2=80,minRadius=100,maxRadius=700))                   
            circles = (cv2.HoughCircles(blur2,cv2.HOUGH_GRADIENT,dp=1,minDist=250,param1=h1,param2=h2,minRadius=100,maxRadius=700))                   

            
            ###get masks
            imgs=[]
            rings=[]
            circs=[]
            icircs=[]
            combinedframedata=[]
            fcircintensity=[]
            fringintensity=[]
            if circles is not None:
                for ic, xc in enumerate(circles[0]):
                    rad = int(xc[2])
                    if (rad < widthmax) and (rad > widthmin):
                        cv2.circle(circMask,(xc[0],xc[1]),rad+half,1,-1)
                        #cv2.circle(icircMask,(xc[0],xc[1]),rad-half,1,-1)
                        #cv2.circle(ringMask,(xc[0],xc[1]),rad,1, thickness)
                        cv2.circle(circimg,(xc[0],xc[1]),rad,(255, 255, 50),thickness)
                        tcircMask = np.zeros(orig.astype(np.float32).shape,dtype=np.uint16)
                        ticircMask = np.zeros(orig.astype(np.float32).shape,dtype=np.uint16)
                        tringMask = np.zeros(orig.astype(np.float32).shape,dtype=np.uint16)
                        tcircimg = cv2.cvtColor(circimgt,cv2.COLOR_GRAY2BGR)
                        cv2.circle(tcircMask,(xc[0],xc[1]),rad+half,1,-1)
                        cv2.circle(ticircMask,(xc[0],xc[1]),rad-half,1,-1)
                        cv2.circle(tringMask,(xc[0],xc[1]),rad,1, thickness)
                        cv2.circle(tcircimg,(xc[0],xc[1]),rad,(255, 255, 50),thickness)
                        circs.append(tcircMask)
                        rings.append(tringMask)
                        icircs.append(ticircMask)
                        imgs.append(tcircimg)
                        ### get intesity
                        circintensity=[]
                        ringintensity=[]
                        icircintensity=[]
                        icircarea=np.pi*float(rad-half)*float(rad-half)
                        circarea=np.pi*float(rad+half)*float(rad+half)
                        ringarea=2*np.pi*float(rad)*thickness    
                        writeline=[fname,rf,ic,rad+half,(xc[0]),(xc[1]),circarea,ringarea,icircarea]
                        for rc in range(cnum):
                            circtemp=(np.multiply(imagei[rc],tcircMask))
                            ringtemp=(np.multiply(imagei[rc],tringMask))
                            icirctemp=(np.multiply(imagei[rc],ticircMask))
                            #sum
                            writeline.append(np.sum(circtemp))
                            writeline.append(np.sum(ringtemp))
                            writeline.append(np.sum(icirctemp))
                            ##ID
                            writeline.append(np.nanmean(np.where(circtemp!=0,circtemp,np.nan)))
                            writeline.append(np.nanmean(np.where(ringtemp!=0,ringtemp,np.nan)))
                            writeline.append(np.nanmean(np.where(icirctemp!=0,icirctemp,np.nan)))
                            ##med
                            medtc=(np.nanmedian(np.where(circtemp!=0,circtemp,np.nan)))
                            medtr=(np.nanmedian(np.where(ringtemp!=0,ringtemp,np.nan)))
                            medtic=(np.nanmedian(np.where(icirctemp!=0,icirctemp,np.nan)))
                            writeline.append(medtc)
                            writeline.append(medtr)
                            writeline.append(medtic)
                            #ratio
                            if (medtic) > 0.0:
                                writeline.append(medtr/medtic)
                            else:
                                writeline.append(0.0)
                        writedatac.append(writeline)
                        beaddata.append(writeline)
            else:
                writeline=[fname,rf,0,0,0,0,0,0,0]
                for rc in range(cnum):
                    writeline.append(0)
                    writeline.append(0)
                    writeline.append(0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                beaddata.append(writeline)
                writedatac.append(writeline)
                writeline=[fname,rf,0,0,0,0,0,0,0]
                for rc in range(cnum):
                    writeline.append(0)
                    writeline.append(0)
                    writeline.append(0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                    writeline.append(0.0)
                beaddata.append(writeline)
                writedatac.append(writeline)
               
            #plot each circle
            if saveimgs:
                psave(blur2,(impath+fname+'_fld'+str(rf)+'.png'))
                psave(circimg,(impath+fname+'_fld'+str(rf)+'_ring.png'))
            with open((outpath+'bybead'+'.csv'), mode='a') as writefile:
                writefile_writer = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for xw in writedatac:
                    writefile_writer.writerow(xw)

# In[13]:

"""exppath= r"/stor/work/Marcotte/project/Sandra/BeadExperiment/2019_06_06_NaHCO3_DMF_SF554/test/"
outpath= r"/stor/work/Marcotte/project/Sandra/BeadExperiment/2019_06_06_NaHCO3_DMF_SF554/test/"
groupby='field'
saveimgs=True
saveto="bead_analysis_out/"
h1=45
h2=45
BeadAnalysis(exppath,outpath,saveimgs,groupby,saveto,h1,h2)"""







