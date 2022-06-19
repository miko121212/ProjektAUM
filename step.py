#LIBRARIES
from PIL import Image
import joblib
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import shape
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.utils import compute_class_weight
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


#SRC DIR
src1 = r"C:\Users\Sumez\Documents\Food\bananas"
src2 = r"C:\Users\Sumez\Documents\Food\apples"
cat = ['banana','apple']
tsrc1 = r"C:\Users\Sumez\Documents\Food\train_bananas\\"
tsrc2 = r"C:\Users\Sumez\Documents\Food\train_apples\\"
print(dir)
y = []
X = []

#LABELS & RESIZE
def resize_img(src1,src2):
    r = 1
    for i in os.listdir(src1):
        current = os.path.join(src1, i)
        print(current)
        if os.path.isfile(current):
            print(current)
            im = Image.open(current)
            a, b = os.path.splitext(current)
            resized = im.resize((300,300))
            resized = resized.convert('L')
            resized.save(tsrc1 + 'Banana_' + str(r) + b)
            img_array = asarray(resized)
            X.append(img_array)
            y.append(0)
            r = r + 1
    r = 1
    for i in os.listdir(src2):
        current = os.path.join(src2, i)
        if os.path.isfile(current):
            print(current)
            im = Image.open(current)
            a, b = os.path.splitext(current)
            resized = im.resize((300,300))
            resized = resized.convert('L')
            resized.save(tsrc2 + 'Apple_' + str(r) + b)
            img_array = asarray(resized)
            X.append(img_array)
            y.append(1)
            r = r + 1
resize_img(src1,src2)

#Reshape features
X = np.reshape(X,(-1,300, 300, 1))

# Display the first image in training data
plt.figure(figsize=[5,5])
curr_img = np.reshape(X[0], (300,300))
plt.imshow(curr_img, cmap='gray')

#Transform to float and normalize
X = np.array(X, dtype=np.float32)
X /= 255

#Labels to numpy array
y = np.array(y)

#SHUFFLE
def unison_shuffled_copies(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]
X_train,y_train = unison_shuffled_copies(X,y)

#80/20 split
X_train, X_test = train_test_split(X, test_size=0.2, shuffle = False )
y_train, y_test = train_test_split(y, test_size=0.2, shuffle = False )
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(2, activation='softmax'))
model.summary()
tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.0,
    nesterov=False,
    name='SGD',
)
model.compile(optimizer='SGD', loss="categorical_crossentropy", metrics=['accuracy'])

exit = "e"
strain = "t"
stest = "r"
shelp = "h"
inp = None

while exit != inp:
    print('Aby rozpoczac uczenie wcisnij "T"', 'Aby rozpoczac test wcisnij "R"', 'Aby uzyskać pomoc wcisnij "H"',
            'Aby zakończyć działanie programu wciśniej "E"', sep='\n')
    inp = input()
    if inp == strain:
        #Train + plot
        model.fit(X_train, y_train, epochs=12)
        model.summary()
    elif inp == stest:
        #Test
        test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1)
        print(test_acc)
    elif inp == shelp:
        print('Program natury problemu klasyfikacji obrazów z wykorzystaniem algorytmu CNN','T - Rozpoczyna uczenie maszyny rozróżniania banana od jabłka na bazie danych treningowych',
              'R - Rozpoczyna właściwy test algorytmu UWAGA jeśli nie uruchomiłeś wcześniej trybu treningowego maszyna będzie klasyfikować "na ślepo"',
              'E - Terminuje program oraz wyświetla grafy', sep='\n')

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    ):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

X, y = load_digits(return_X_y=True)

title = "Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(
    estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
)

plt.show()