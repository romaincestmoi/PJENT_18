import numpy as np 

import matplotlib.pyplot as plt 

from skimage import data, filters 

import os 

import cv2 

import numpy as np 

 

 


data_path = 'C:/Users/hilal/OneDrive/Bureau/PROCESSED/VIDEO_360/DATASET_SENSEA/images' 

image_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg') or f.endswith('.png')]) 


if not image_files: 

    print("Aucune image trouvée dans le dossier.") 

else: 

    print(f"{len(image_files)} images chargées.") 

 
 

img = cv2.imread(image_files[0]) 

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

plt.title('Aperçu de la première image') 

plt.show() 

 

 

for i in range(len(image_files) - 1): 

    img1 = cv2.imread(image_files[i]) 

    img2 = cv2.imread(image_files[i + 1]) 

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0) 

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 

    hsv = np.zeros_like(img1) 

    hsv[..., 1] = 255 

    hsv[..., 0] = angle * 180 / np.pi / 2 

    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 

 

    plt.imshow(flow_rgb) 

    plt.title(f'Flux optique entre image {i+1} et image {i+2}') 

    plt.show() 