import numpy as np
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from scipy.io import arff
import matplotlib.pyplot as plt

# Load ARFF Dataset
print("Load Dataset ARFF")
data, meta = arff.loadarff('mnist_784.arff')
dataset = np.array(data.tolist(), dtype=np.float32)

# Data Processing
print("Processing Data")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
num_samples, num_features = X.shape
image_size = int(np.sqrt(num_features))
X_images = X.reshape((num_samples, image_size, image_size))

# HOG Feature Extraction Function
def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# HOG Feature Extraction
print("Extract_HOG_Features")
train_features = np.array([extract_hog_features(image) for image in X_images])

# Data Splitting
print("Data Sharing")
X_train, X_val, y_train, y_val = train_test_split(train_features, y, test_size=0.2, random_state=42)

# SVM Model Training
print("SVM Model Training:")
clf = svm.SVC()
clf.fit(X_train, y_train)

# Prediction on Validation Set
val_predictions = clf.predict(X_val)

# Accuracy Calculation
accuracy = accuracy_score(y_val, val_predictions)
print(f'Set Validasi accuracy: {accuracy}\n')

# Confusion Matrix
conf_matrix = confusion_matrix(y_val, val_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Precision Score for each class
precision_per_class = precision_score(y_val, val_predictions, average=None)
print("Precision Score for each class:")
for i, precision in enumerate(precision_per_class):
    print(f"Class {i}: {precision}")

# Display sample images with predicted labels
plt.figure(figsize=(10, 8))
random_indices = np.random.choice(len(X_val), 9, replace=False)

for i, index in enumerate(random_indices, 1):
    plt.subplot(3, 3, i)
    sample_image = X_images[index]
    sample_hog_feature = extract_hog_features(sample_image)
    predicted_label = clf.predict([sample_hog_feature])[0]

    plt.imshow(sample_image, cmap='gray')
    plt.title(f'Predicted Label: {predicted_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()

