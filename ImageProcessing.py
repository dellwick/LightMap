#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:42:18 2017

@author: Zachary Freer
"""

import magic
import cv2
import numpy as np
from math import floor


# Differentiates from animated and nonanimated image formats and crops them 
# into a circle with an alpha layer in the background.
class ImageProcessing():
    def __init__(self, filename):
        self.filename = filename
        self.size = ()
        
    # Sorts between animated and nonanimated images
    def file_sorting(self):
        animated_formats = ['gif', 'apng', 'mng', 'svg', 'svgz', 'webp']
        
        info = magic.from_file(self.filename, mime=True)
        info = info.split('/')
        
        if info[0] != 'image':
            raise IOError("Not an image")
        
        elif info[1] not in animated_formats:
            self.nonanimated_image()
            
        else:
            self.animated_image()
    
    # Handles all non animated file formats        
    def nonanimated_image(self):
        image = cv2.imread(self.filename, -1)
        #image = self._add_alpha(image)
        
        
        
        self.generate_circle(image)
        
        
        
        '''
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        
        roi_corners = np.array([[(10,400),(300,400),(10,300)]], dtype=np.int32)
        
        channel_count = image.shape[2]
        ignore_mask_color = (255, )*channel_count
        
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        
        masked_image = cv2.bitwise_and(image, mask)
        
        cv2.imwrite('output.png', masked_image)
        '''
        
    
        
    def generate_circle(self, image):
        self.crop(image)
        
    
    # Crops image into square with the dimensions 2*radius-X-2*radius    
    def crop(self, image):
        self.get_size(image)
        index = self.find_min(self.size[0], self.size[1])
        
        radius = floor(self.size[index] / 2)
        center = (floor(self.size[0] / 2), floor(self.size[1] / 2))
       
        circle = image[(center[0] - radius):(center[0] + radius),
                       (center[1] - radius):(center[1] + radius)]
        
        self.add_alpha(image, circle, radius, center)
        
    # Adds alpha layer behind circle image
    def add_alpha(self, image, circle, radius, center):
        b_channel, g_channel, r_channel = cv2.split(image)
        
        a_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50
        
        
        
    
    # Returns the index of the shorter dimension of the input image
    def find_min(self, x, y):
        if x < y:
            return 0
        else:
            return 1        
            
    # Gets the dimensions of the input image
    def get_size(self, image):
        self.size = image.shape[:2]
        
        
        
        
        
    # Handles animated image formats 
    def animated_image(self):
        return