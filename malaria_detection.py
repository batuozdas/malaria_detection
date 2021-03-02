import cv2,os,glob,time,random
from matplotlib import pyplot as plt ; import matplotlib ; matplotlib.use('TkAgg')
import numpy as np

def dataset(path='resimler',label = ['Parasitized','Uninfected']):
    # We indicate the path of the images.
    infected_cell_names = os.listdir('{}/{}/'.format(path,label[0]))# This line returns infected cell images' names of our dataset.
    uninfected_cell_names = os.listdir('{}/{}/'.format(path,label[1]))# This line returns uninfected cell images' names of our dataset.
    infected_cell_images = [] # Creating an empty list to add infected cell images.
    uninfected_cell_images = [] # Creating an empty list to add uninfected cell images.
    for img in infected_cell_names: # Reading every infected cell images.
        image_bgr = cv2.imread('{}/{}/{}'.format(path,label[0],img), 1)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        infected_cell_images.append(np.array(image)) # Adding readed images to infected_cell_images list.
    for img in uninfected_cell_names: # Reading every uninfected cell images.
        image_bgr = cv2.imread('{}/{}/{}'.format(path,label[1],img), 1)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        uninfected_cell_images.append(np.array(image))# Adding readed images to uninfected_cell_images list.

    infected_cell_images = np.array(infected_cell_images,dtype='object')
    uninfected_cell_images = np.array(uninfected_cell_images,dtype='object')
    label_infected = np.ones(len(infected_cell_images)) # Labeling infected cell images as 1.
    label_uninfected = np.zeros(len(uninfected_cell_images)) # Labeling uninfected cell images as 0.
    # Creating labels and images values for machine learning;
    labels = np.append(label_infected,label_uninfected)
    images = np.append(infected_cell_images,uninfected_cell_images)
    return images,labels

def image_processing(img):
    # Firstly image's brightness will be increased for better visual. For that, rgb image will be converted to hsv image. And brightness will
    # be applied to v channel of hsv image.
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2] # Splitting channels.
    value = 30 # Brightness value.
    limit = 255 - value #Limit value. If we don't define limit value then pixel values of images might be bigger than 255.
    v[v > limit] = 255 # If pixel values of v channel are bigger than limit value, then these values will be equal to 255.
    v[v <= limit] += value # # If pixel values of v channel are not bigger than limit value, then these values will be increased by brihtness value.
    hsv_img_new = cv2.merge((h, s, v)) # After applying brightness operation to v channel, channels will be merged.

    img_brightness = cv2.cvtColor(hsv_img_new, cv2.COLOR_HSV2RGB) # Now we have brightness-increased image.

    # We convert RGB image to gray image for image processing.
    img_gray = cv2.cvtColor(img_brightness, cv2.COLOR_RGB2GRAY)
    row = img_gray.shape[0]
    col = img_gray.shape[1]
    # There are black areas in the dataset images. These black areas will be converted to white areas.
    img_white = np.zeros((row, col)) # Creating a zero matrix to fill later.
    img_white.fill(255) # Filling img_white matrix with 255 value. Now we have pure white image.
    for i in range(row):
        for j in range(col):
            if img_gray[i, j] != value: # Actually, instead of img_gray[i, j] != value, we should use img_gray[i, j] != 0 because black areas are equal to '0'.
                # But before that we increased brightness, so black areas pixel values also increased from 0 to value (30).
                img_white[i, j] = img_gray[i, j]
            else:
                continue
    # Now black areas are converted to white areas. The reason of this operation: These black areas may cause a problem for thresholding and contour detection operation.
    # When we detect malaria, it will also detect borders of the black areas. That is why black areas are converted to white areas.

    # Now there will be sharp transition between gray pixel values and white pixel values. To avoid this we use;
    img_white = cv2.convertScaleAbs(img_white, alpha=1.5, beta=25)

    # To see malaria more clear, CLAHE operation is used.
    clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8, 8))#CLAHE metodunu uygulamak için clahe oluşturuyoruz.
    img_hist = clahe.apply(img_white)#Oluşturduğumuz clahe'yi img_white görüntüsüne uyguluyoruz.

    # To blur the noises of image, low pass filter is used. For that Fourier transform is used.
    img_fft = np.fft.fft2(img_hist, s=[row, col])
    img_fft_shift = np.fft.fftshift(img_fft)
    # Center of img_fft_shift values will be low frequency values. So if all areas are masked except center areas,
    # Only low frequencies will pass.
    c_row, c_col = int(row / 2), int(col / 2) # Center of image.
    center = [c_row, c_col]
    x, y = np.ogrid[:row, :col]# x and y coordinates are created.
    r = 55 # radius
    mask = np.zeros((row, col)) # For masking operation, 0 matrix is created.
    mask_range = (x - center[0]) ** 2 + (y - center[1]) ** 2 < r * r # We define a mask range. ((x-a)^2 + (y-b)^2 < r^2)
    mask[mask_range] = 1 # The area of the circle with center (c_row, c_col) and radius 55 will be 1, other places will be 0.

    masked_img = mask * img_fft_shift # Image is masked.
    # Inverse Fourier Transform;
    img_ifft_shift = np.fft.ifftshift(masked_img)
    img_ifft = np.fft.ifft2(img_ifft_shift, s=[row, col])
    img_blur = np.abs(img_ifft) # img_ifft values are complex values. So we get amplitudes of these values.
    # Now we have blurred image.

    # Thresholding operation.
    val_thresh, img_thresh = cv2.threshold(img_blur, 210, 255, cv2.THRESH_BINARY)
    # After thresholding operation there are still some 1-pixel noises in some images. To eliminate these noises, opening (erode+dilate) and
    # closing (dilate+erode) operations were applied to images.

    # Opening operation
    mask = cv2.erode(img_thresh, kernel=np.ones((3, 3)), iterations=1)
    mask2 = cv2.dilate(mask, kernel=np.ones((3, 3)), iterations=1)

    # Closing operation
    mask3 = cv2.dilate(mask2, kernel=np.ones((3, 3)), iterations=1)
    img_thresh_clean = cv2.erode(mask3, kernel=np.ones((3, 3)), iterations=1)

    # Now we have white background and black object (malaria) image. For contour detection, background of image must be
    # black and object (malaria) in the image must be white.

    img_black_background = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if img_thresh_clean[i, j] == 0: # If pixel value of img_thresh_clean = 0 (black),
                img_black_background[i, j] = 255 # Make that value white.
            elif img_thresh_clean[i, j] == 255: # If pixel value of img_thresh_clean = 255 (white),
                img_black_background[i, j] = 0 # Make that value black.

    # Now we have black background image.
    img_black_background = cv2.dilate(img_black_background, kernel=np.ones((7, 7))) # After opening and closing operation,
    # area of malaria in some of images has decreased. For increase the area again, dilate operation is used.
    img_black_background = np.array(img_black_background, dtype=np.uint8)
    # Contour detection;
    contours, hierarchy = cv2.findContours(img_black_background, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    return image,contours # Return malaria detected (contour detected) image, and contours of that image.

def machine_learning(x_train,y_train):
    tp = 0 # True positive
    tn = 0 # True negative
    total = 0 # Total images
    fp = 0 # False positive
    fn = 0 # False negative
    nc = 0 # No classified
    img_preprocess = [] # Empty series for adding the contour detected images coming from image_processing function.
    img_contours = [] # Empty series for adding contours of contour detected images coming from image_processing function.
    label = list(map(lambda x : 'a',range(0,len(x_train)))) # To label the images, label list consisting of 'a' values is created.

    for i in range(len(x_train)):
        img_proc,img_cont = image_processing(x_train[i]) # x_train list has images of dataset. We send every images to image_processing
        # function. And we take 2 values. One of them is contour detected image and other one is contours of that image.
        img_preprocess.append(img_proc) # We add contour detected images to img_preprocess empty list.
        img_contours.append(img_cont) # We add contours of contour detected images to img_preprocess empty list

    # x_train list has images of dataset and y_train list has labels (consisting of 1(infected) and 0(uninfected) values) of dataset.
    for i in range(len(x_train)):
        if (len(img_contours[i]) >= 1) & (y_train[i] == 1) : # If numbers of detected contour of image is greater than 1 and that image
            # is infected;
            tp += 1
            label[i] = 'Infected'

        elif (len(img_contours[i]) == 0) & (y_train[i] == 0) : # If numbers of detected contour of image is equal to 0 and that image
            # is uninfected;
            tn += 1
            label[i] = 'Uninfected'

        elif (len(img_contours[i]) >= 1) & (y_train[i] == 0) : # If numbers of detected contour of image is greater than 1 and that image
            # is uninfected
            fp += 1
            label[i] = 'False Positive'

        elif (len(img_contours[i]) == 0) & (y_train[i] == 1) : # If numbers of detected contour of image is equal to 0 and that image
            # is infected
            fn += 1
            label[i] = 'False Negative'
        else:
            nc += 1
        total += 1
    return tp,tn,total,fp,fn,nc,label,img_preprocess

start_time = time.time()
x_train,y_train = dataset('resimler/blood_cells/cell_images/test',['Parasitized','Uninfected'])
tp,tn,total,fp,fn,nc,label,img = machine_learning(x_train,y_train)
end_time = time.time()

# Results;
print('Operation took {} seconds.'.format(end_time-start_time))
accuracy = ((tp + tn) / total) * 100 # Accuracy
print('Total Images : {}'.format(total))
print('Count of True Positive : {}'.format(tp))
print('Count of True Negative : {}'.format(tn))
print('Count of False Positive : {}'.format(fp))
print('Count of False Negative : {}'.format(fn))
print('No classified : {}'.format(nc))
print('Accuracy : % {}'.format(accuracy))

# Random plotting;
r = 3
c = 4
fig,axes = plt.subplots(r,c)
for i in range(0,r):
    for j in range(0,c):
        rand = random.randint(0,len(x_train))
        axes[i,j].imshow(img[rand])
        axes[i,j].set_title(label[rand])
plt.show()