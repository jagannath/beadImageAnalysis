#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import cv2
import csv
from statistics import mean, median, stdev
import pandas as pd; print("Pandas version: {0}".format(pd.__version__)); 

# Avoid user warnings from nikon (?)
import warnings; warnings.simplefilter('ignore',UserWarning)
import nd2reader as nd2; print("nd2 reader version: {0}".format(nd2.__version__))

from tqdm import tqdm_notebook,trange, tqdm 
warnings.simplefilter('ignore',DeprecationWarning) # This should be fine if tqdm not updated; it is the tqdm_notebook thats crying

# Mymodules
import mymodules

#-------------------------

class ImageStack(object):
    """
    Summary
    ----------
    Containing functions to (a) process Nikon images - nd2 file format (b) Extracts relevant file images and (c) metadata 
    for further image processing. 
    NOTE: The file form of the nd2 file must be only channels and no frames/time/zstacks etc; 
    TODO: A different nd2 stack will mess up the processing and analysis and will need reprocessiong; 
    A scaleable method will be to have a function that converts it to a channel_stack images
    Need to catch or reprocess exceptions if not an Nikon ND2 file format
    
    Attributes 
    ----------
    * bead_nd2_fpath: (str); Its the absolute path to the nd2 file
    * mono_channel: (int) : It is meant to be the channel used for extracting the image of the beads; Typically a 
    non-fluorescent bead channel
    * metadata: (list) : It is the nikon nd2 raw metadata
    * nd2image: (nd2 image class): This is their proprietary class; The relevant extraction here is the image information 
    that is an array (not yet numpy)
    
    Methods
    -------
    * check_nd2_file_form: This is currently critical to ensure that there is only one stack of channels 
    and not a time or xy series. So I have thrown some conditional statements in there
    Returns: True if the form is correct
    * metadata: Quick extraction of metadata; Returns metadata and the nd2image
    * channels: Specific call out to the channels
    * frames: Extracts out the frame information 
    * fname : File name (not the path)
    * mono_image: This is typically the grayscale image (NOTE - not uint8); Typically DIC or BrightField
    """

    def __init__(self,bead_nd2_fpath,mono_channel=0):
        self.bead_nd2_fpath = bead_nd2_fpath
        self.mono_channel = mono_channel
        self.metadata, self.nd2image = self.get_metadata()
        self.check_nd2_file_form()
      
    def check_nd2_file_form(self):
        cond1 = (len(self.metadata['fields_of_view']) == 1) #Only one allowed field of view ; Only one xy position
        cond2 = (len(self.metadata['frames'] == 1) # Again - no time series
        assert (cond1 and cond2),"Metadata {0}".format(self.metadata)
        assert ((len(self.metadata['channels']))>1), "Not enough channels in this image stack"
        return True
    
    def get_metadata(self):
        nd2image = nd2.ND2Reader(self.bead_nd2_fpath)
        metadata = nd2image.metadata
        return metadata, nd2image
    
    def get_mono_image(self): #Seems pretty regularly called upon
        return self.nd2image.get_frame(self.mono_channel)
    
    def channels(self):
        return self.metadata['channels']
    
    def frames(self):
        return self.metadata['frames']
   
    def fname(self):
        return(os.path.split(self.bead_nd2_fpath)[1])


class Beads(ImageStack):
    """
    Summary
    ----------
    The Beads class detects multiple Beads in an image. It tracks the intensity of the beads across the different channels. 
    It does not extract intensity and other relevant information for individual bead. The class inherits ImageStack mainly 
    to be able to retrieve the mono images and the metadata required for 
    Since this is a class that combines multiple information, it is responsible for exports data and images. Additionally 
    making the dataframe in pandas that will be parsed and plotted later
    TODO: Need to catch exceptions or only inherit an Imagestack that is a stack of channels and nothing else!
    
    Attributes 
    -----------
    * bead_nd2_fpath: This is the path to the nd2 image containing multiple beads. 
    An important part of this identifier is that it is passed and can be passed to the ImageStack Class; 
    * col_headers: This is the column of bead information for the pandas dataframe
    * mono_channel: This is again the mono_channel (the DIC or brightfield). 
    Made this an attribute as I can work around the ImageStack class, Not so elegant
    
    Function 
    ---------
    * make_column_header: Function to make a pre-determined column header for the dataframe. Returns the column header as list
    * make_dataframe: Function that creates a dataframe from a list of row information provided. Returns dataframe
    * preprocess_image: Function that preprocesses and cleans up the image. Its an internal function called by find_beads
    TODO: Add other processing algorithms and validate. Can be important if Hough's algorithm starts to fail. 
    * filter_beads: This is mainly a catch function to filter out beads (or circles drawn) that is smaller than expected. 
    It maybe an over aggressive move, given that Hough's algorithm already has a size filter
    * find_beads: This is the function using Hough's algorithm. The large parameter passed from user finds its way here. 
    Returns a list of tuple - circles 
    * export_data_frame: The function exports the dataframe as a csv file (kept as default);
    * export_masked_image: Creates a ring around the circle and exports the raw and the png file; 
    TODO: The export folder variable is being passed around in kwargs. Feeling risky
    """

    def __init__(self,bead_nd2_fpath,mono_channel=0):
        super().__init__(bead_nd2_fpath) #I don't like this, but there are too many methods that can be passed down
        self.col_headers = self.make_column_header()
        self.mono_channel = mono_channel
    
    def make_column_header(self):
        channels = self.channels()
        channel_dependent_headers = list()
        col_headers = ["Filename", "Frame ID", "Mask ID", "Mask-X0","Mask-Y0","oRadius (pix)",
                       "Circular Area (pix^2)", "Ring Area (pix^2)", "inner Circle Area (pix^2)"]
        
        for channel_name in channels:
            _header_list = [channel_name + '---' + _head_ 
                            for _head_ in ('oCircle Intensity', 'oCircle Mean Intensity', 'oCircle Median Intensity',
                                           'iCircle Intensity', 'iCircle Mean Intensity', 'iCircle Median Intensity',
                                           'ring Intensity', 'ring Mean Intensity', 'ring Median Intensity')]
            channel_dependent_headers.extend(_header_list)
        col_headers.extend(channel_dependent_headers)
        return col_headers
    
    def make_dataframe(self,row_values):
        allbeads_df = pd.DataFrame(row_values,columns=self.col_headers)
        return allbeads_df
    
    def preprocess_image(self,img,preprocessing_params=['median',5]):
        mod_img1 = img.astype(np.float32)
        mod_img2 = (mod_img1/256).astype('uint8') #Need an 8-bit, single-channel, grayscale input
        # Blurring
        if preprocessing_params[0] == 'median': cv2.medianBlur(mod_img2,preprocessing_params[1])
        if preprocessing_params[0] == 'gaussian': cv2.GaussianBlur(mod_img2,preprocessing_params[1])
        return mod_img2

    def filter_beads(self,circles, min_width=100,max_width=170):
        all_circles = list()
        for circle in circles:
            r = int(circle[2])
            if r in range(min_width,max_width):
                all_circles.append(circle)
        return all_circles

    def find_beads(self,**kwargs):
        """
        hough_params = dict(dp=1.5,minDist=250,param1=45,param2=45,minRadius=100,maxRadius=700)
        circles = t.find_beads(mono_image,**hough_params)
        """
        hough_params = kwargs['hough_params']
        _preprocessing_params = kwargs['preprocessing_params'] #I am barely modifying this
        _processed_img = self.preprocess_image(self.get_mono_image(),_preprocessing_params)
        circles = (cv2.HoughCircles(_processed_img,cv2.HOUGH_GRADIENT,
                                    dp = hough_params['dp'],
                                    minDist = hough_params['minDist'],
                                    param1 = hough_params['param1'],
                                    param2 = hough_params['param2'],
                                    minRadius = hough_params['minRadius'],
                                    maxRadius = hough_params['maxRadius']))
        circles = self.filter_beads(circles[0]) #Filter the beads across the expected width (and not too small)
        # A number of circles with each circle in form = (x,y,radius)
        return circles 
    
    def export_data_frame(self,data_frame, export_folder):
        #display(data_frame)
        _input_fname = os.path.split(self.bead_nd2_fpath)[1].split('.')[0]
        export_fpath = os.path.join(export_folder, _input_fname+'.csv')
        data_frame.to_csv(export_fpath, header=True, index=False)
        return export_fpath
    
    def export_masked_image(self,circles,export_folder):
        #Overlaying a ring around the detected mask
        ## Need to convert the mono image to a gray scale and convert to uint8
        _mono_image = self.get_mono_image()
        _mono_data = _mono_image.astype(np.float64) / np.max(_mono_image) #Normalizing from 0-1
        _mono_data = 255 * _mono_data # Now scaled to 255
        _img = _mono_data.astype(np.uint8)
        raw_mono_image = cv2.cvtColor(_img,cv2.COLOR_GRAY2BGR)
        
        ring_image = cv2.cvtColor(raw_mono_image,cv2.COLOR_BGR2RGB)
        
        for circle in circles:
            x0,y0,r = map(int,circle)
            cv2.circle(img=ring_image, center=(x0,y0), radius=r, color=(204,85,0),thickness=30)

        _image_fname = os.path.split(self.bead_nd2_fpath)[1].split('.')[0]
        export_fpath = os.path.join(export_folder,_image_fname)
        cv2.imwrite(export_fpath+'_ring'+'.png',ring_image)
        cv2.imwrite(export_fpath+'_raw'+'.png',_img)
        return export_fpath
            
class Bead(ImageStack):
    """
    Summary
    ----------
    This the class that extracts individual bead information, such as the intensity statistics, area, etc. This inherits the 
    imageStack Class as well. The class works in concert with Beads and could be inherited. Didnt find a clean way. 
    
    Attributes
    -----------
    * bead_nd2_fpath: Nikon microscope nd2 file that is contains the beads
    * circle: This is information of the particular bead - Center coordinates and radius = (x0,y0,r); Is calculated from 
    the Beads class and thus passed
    * mono_image: Grayscale image; TODO: Test what happens if a different frame is chosen across different classes.
    * mask_id: This is just an random numbering index to associate with the bead. --> int
    * image_shape: Gathering the shape of the images as tuple (h,w). Used for making empty images and masks
    * ring_thickness: An integer that is quite fixed. TODO: Unsure if this needs to be passed through kwargs.
    Used only in a couple of functions
    
    Methods
    ----------
    * get_areas: Calculates the area of the bead - outer, inner and ring area; Outputs a list of float
    * draw_mask: Creates an empty image mask using opencv (cv2) library; Requires the center, radius, thickness
    * get_values_under_mask: Internal function uses the masked image for each circle to perform a dot product 
    of the mask image with the image from the different channels. This generates the values for the intensity across the 
    different possibilities 
    * extract_intensity: A main function called that creates mask, calculates and extracts information for each bead. 
    Outputs an intensity vector for integrating into the dataframe
    * compile_row: This function integrates the general bead information along with the intensity values
    """

   def __init__(self,bead_nd2_fpath, circle, mask_id, mono_channel=0):
        super().__init__(bead_nd2_fpath)
        self.circle = list(map(int,circle)) #circle = x,y,r
        self.mono_image = self.nd2image.get_frame(mono_channel)
        self.image_shape = self.mono_image.shape
        self.ring_thickness = int(30)
        self.mask_id = mask_id

    def get_areas(self):
        x0,y0,r0 = self.circle
        delta = self.ring_thickness/2
        outer_circle_area = np.pi * float(r0+delta) * float(r0+delta)
        inner_circle_area = np.pi * float(r0-delta) * float(r0-delta)
        ring_area = 2 * np.pi * float(r0)*self.ring_thickness
        return [outer_circle_area, ring_area, inner_circle_area]
        
    def draw_mask(self,**kwargs):
        empty_img_mask = np.zeros(self.image_shape,dtype=np.uint16)
        cv2.circle(empty_img_mask,kwargs['center'],kwargs['radius'],kwargs['color'],kwargs['thickness'])
        return empty_img_mask
     
    def get_values_under_mask(self,mask_array,idx):
        raw_image = self.nd2image.get_frame(idx)
        _image_masked_ = np.multiply(raw_image,mask_array)
        sum_bead = np.sum(_image_masked_)
        mean_bead = np.nanmean(np.where(_image_masked_!=0, _image_masked_, np.nan))
        median_bead = np.nanmedian(np.where(_image_masked_!=0, _image_masked_, np.nan))
        return sum_bead,mean_bead,median_bead
        
    def extract_intensity(self):
        # The function is simply to do the intensity calculations
        # Initialize
        row_values = list()
        
        x0,y0,r0 = self.circle
        delta = self.ring_thickness/2
        outer_circle_params = dict(center=(x0,y0),radius=int(r0+delta),color=1,thickness=-1)
        inner_circle_params = dict(center=(x0,y0),radius=int(r0-delta),color=1,thickness=-1)
        ring_params = dict(center=(x0,y0),radius=int(r0),color=1,thickness=self.ring_thickness)
        
        # Create bead mask
        outer_circ_mask = self.draw_mask(**outer_circle_params)
        inner_circ_mask = self.draw_mask(**inner_circle_params)
        ring_mask = self.draw_mask(**ring_params)
        
        # Calculate intensity for each channel
        # Looping through the different masks modes
        for circ_mask in (outer_circ_mask,inner_circ_mask, ring_mask):
            # Looping to extract -- Sum, Mean, Median, Ratio
            for  idx,channel_name in enumerate(self.channels()):
                sum_bead, mean_bead, median_bead = self.get_values_under_mask(circ_mask,idx)
                row_values.extend([sum_bead,mean_bead,median_bead])
        return row_values

    def compile_row(self,row_values):
        complete_row_list = list()
        # Basic bead information
        basic_info_row_values = [self.fname(),self.frames()[0],self.mask_id,self.circle[0],self.circle[1],self.circle[2]] 
        complete_row_list.extend(basic_info_row_values)
        area_row_values = np.array(self.get_areas(),dtype=np.float64)
        complete_row_list.extend(area_row_values)
        
        # Intensity across channels
        intensity_row_values = np.array(row_values,dtype=np.float64)
        complete_row_list.extend(intensity_row_values)
        
        return list(complete_row_list)
    
 #---------------------FUNCTIONS------------------------------------------------#
    
def analyze_beads_image(nd2_fpath, **kwargs): 
    '''
    Beads in a single field of view detected and analysed (image stack of channels)
    The goal also is to ensure the class calls are restricted to only this function
    '''
    complete_row_list = list()
    # Processing a single nd2 file to extract bead information
    beadimage = Beads(nd2_fpath)   ## ATTENTION -- Beads class (all beads)
    circles = beadimage.find_beads(**kwargs) #Don't know why I am just rerouting all of kwargs here directly
    export_folder_path = kwargs['export_folder']
    save_status = kwargs['save_status']
    nbr_circles = len(circles)
    
    tqdm_descriptor = "Analyzing beads {0}: ".format(os.path.split(nd2_fpath)[1])
    tqdm_color_orange = '#cc5500'
    
    for mask_id in tqdm_notebook(range(nbr_circles),
                                 desc=tqdm_descriptor,colour=tqdm_color_orange,position=1, leave=False):
        circle = circles[mask_id]
        bead = Bead(nd2_fpath,circle,mask_id) ## ATTENTION -- Bead (single bead)
        bead_values = bead.extract_intensity() 
        row_list = bead.compile_row(bead_values)
        complete_row_list.append(row_list)
    
    beads_data_frame = beadimage.make_dataframe(complete_row_list) ## ATTENTION - Recalling the Beads class
    _nbr_rows = beads_data_frame.shape[0]
    if _nbr_rows >1: status = True
    else: status = False
            
    if save_status in ['data','both']:
        data_export_folder = os.path.join(export_folder_path,'data_frame_export')
        data_export_fpath = mymodules.make_folder(data_export_folder)
        exported_csv_fpath = beadimage.export_data_frame(beads_data_frame,data_export_fpath)
    if save_status in ['mask','both']:
        mask_export_folder = os.path.join(export_folder_path,'mask_image_export')
        mask_export_fpath = mymodules.make_folder(mask_export_folder)
        exported_mask_fpath = beadimage.export_masked_image(circles,mask_export_fpath)
    else:
        pass
    
    ## ASSERTIONS -- MAY NEED TO BE SILENCED
    assert (_nbr_rows == len(circles)), "Circles detected != Dataframe rows"
    assert (nbr_circles > 2), "Can you do better in number of circles detected {0} ?".format(nbr_circles)
    
    return True, beads_data_frame

#--------------------------------------------------------------------------------#
                 
if __name__ == '__main__':
    print("This is not the module ")
