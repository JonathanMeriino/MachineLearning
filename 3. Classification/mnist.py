import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
# Descargar MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

x, y = mnist.data, mnist.target

def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# Mostrar el primer dígito
some_digit = x[0]
plot_digit(some_digit)
plt.show()

x_train , x_test , y_train, y_test = x[:60000], x[60000:], y[:60000],y[60000:]

# Entrenamiento del clasificador binario "5-detector"

y_train_5 = (y_train=='5') #Verdadero para "5". Falso para los demas digitos 
y_test_5 = (y_test=='5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train_5)

#Medida de deseempeño 

# Medida de la precisioin usando validacion cruzada
scores = cross_val_score(sgd_clf, x_train, y_train_5, cv=3, scoring="accuracy")

print(scores)

#Clasificador DUMMY (Ficticio)

dummy_clf = DummyClassifier()
dummy_clf.fit(x_train,y_train_5)
print(any(dummy_clf.predict(x_train))) # Falso por que no detecta "5"

scoredummy = cross_val_score(dummy_clf, x_train, y_train_5, cv=3,scoring="accuracy")
print(scoredummy)
