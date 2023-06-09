import pickle
import numpy as np
import time
from sklearn import svm
from skimage.feature import hog
from skimage import io , transform
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
image_path = 'Data/truck/000000515266_truck_270.76_99.18_358.77_164.23.jpg'
image = io.imread(image_path, as_gray=True)
print(image)
image = transform.resize(image , (128, 128))
hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
start = time.time()
x = np.array([hog_features])  # Wrap hog_features in a list to create a 2D array
end = time.time()
print(end - start)
predictions = model.predict(x)


print(predictions)