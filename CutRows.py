import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import os
from glob import glob

def open_image(path):
    #open in grayscale mode 'L'
    image = imageio.imread(path, pilmode='L') 
    return image

def line_contrast(page_image):
    line_contr =[]
    #determine range per line
    for line in page_image: 
        line_contr.append(max(line) - min(line))
    return line_contr

def find_rows(line_contr):
    detected_rows = []
    row_start = 0
    row_end = 0
    detect_state = 0 #0 if previous line was not part of a row
    cur_row = 0
    for contrast in line_contr:
        if contrast < 50 and detect_state == 0:
            row_start = cur_row
        elif contrast >= 50 and detect_state == 0:
            row_start = cur_row
            detect_state = 1
        elif contrast < 50 and detect_state == 1: #if end of row, evaluate AOI height
            row_end = cur_row
            rowheight = row_start - row_end
            if abs(rowheight) >= 150:
                detected_rows.append((row_start, row_end))
            detect_state = 0
        elif contrast >= 50 and detect_state == 1:
            pass
        else:
            print("unknown situation, help!, detection state: " + str(detect_state))
        cur_row += 1
    return detected_rows

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_rows(sliced_rows, composer):
    path = 'Dataset/%s/' %composer
    checkpath(path)
    for row in sliced_rows:
        file_number = len(glob(path + '*'))
        misc.imsave(str(path) + '/' + str(file_number) + '.jpg', row)

def slice_rows(page_image, detected_rows, composer):
    sliced_rows = []
    max_height= 350
    max_width = 2000
    for x,y in detected_rows:
        im_sliced = np.copy(page_image[x:y])
        new_im = np.empty((max_height, max_width))
        new_im.fill(255)
        if im_sliced.shape[0] <= max_height:
            new_im[0:im_sliced.shape[0], 0:im_sliced.shape[1]] = im_sliced
            sliced_rows.append(new_im)
        elif max_height < im_sliced.shape[0] < 1.25 * max_height:
            im_sliced = im_sliced[0:max_height, 0:im_sliced.shape[1]]
            new_im[0:im_sliced.shape[0], 0:im_sliced.shape[1]] = im_sliced
            sliced_rows.append(new_im)
        else:
            print("Skipping block of height: %s px" %im_sliced.shape[0])
            checkpath('Dataset/%s/Errors/' %composer)
            file_number = len(glob('Dataset/%s/Errors/*' %composer))
            #save to error dir for manual inspection
            misc.imsave('Dataset/%s/Errors/%s_%s.jpg' %(composer, file_number, composer), im_sliced)
    return sliced_rows

if __name__ == '__main__':
    #open image and determine pixel ranges
    image = open_image('twinkle.jpg')
    #get contrast ranges
    line_contr = line_contrast(image)
    #find the rows
    detected_rows = find_rows(line_contr)
    #plot the pixel contrast
    plt.plot(line_contr)
    plt.title('Row-wise pixel value range')
    plt.xlabel('Horizontal row #')
    plt.ylabel('Pixel value range on row')
    plt.show()
    for row in detected_rows:
        #draws black line at beginnning and end of each row
        image[row[0]] = 0
        image[row[1]] = 0

    imageio.imwrite('output.jpg', image)