import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt


training_real_folder = r'C:\training_real' #My C drive path
training_fake_folder = r'C:\training_fake' #My C drive path

def load_images_from_folder(folder, label): #To check and read the images from each folder
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):  # Check the file name
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  #resizing numbers to resize the photos in each folders
            if img is not None:
                images.append(img.flatten())
                labels.append(label)
    return images, labels

#2 Functions to load each of the images
training_real, real_labels = load_images_from_folder(training_real_folder, 1)
training_fake, fake_labels = load_images_from_folder(training_fake_folder, 0)

#Combing both images together
X = np.array(training_real + training_fake)
y = np.array(real_labels + fake_labels)

#To train and test each image
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#The required algorithms
models = {
    "1.Naive Bayes": GaussianNB(),
    "2. K-Nearest Neighbor": KNeighborsClassifier(),
    "3. Decision Tree": DecisionTreeClassifier(),
    "4. Logistic Regression": LogisticRegression(max_iter=3000)
}
#Plotting the accuracy scores and the losses
accuracy_scores = []
losses = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    if hasattr(model, 'predict_proba'): #predict proba is the columns and arrays
        loss = log_loss(y_test, model.predict_proba(X_test))
        losses.append(loss)
    else:
        losses.append(None)

#Plots for each result
fig, ax1 = plt.subplots()

#Blue is for the bars on the grpah and titles of the bars
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy', color=color)
ax1.bar(models.keys(), accuracy_scores, color=color, alpha=0.6, label="Accuracy")
ax1.tick_params(axis='y', labelcolor=color)

#Green is for the line on the graph
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Log Loss', color=color)
ax2.plot(models.keys(), losses, color=color, alpha=0.6, marker='o', label="Log Loss")
ax2.tick_params(axis='y', labelcolor=color)

#The title of the figure
fig.tight_layout()
plt.title( ' The Model Accuracy and Loss Chart')
plt.show()
