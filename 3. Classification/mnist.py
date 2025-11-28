import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Descargar el dataset MNIST
mnist = fetch_openml('mnist_784', version=1)

x, y = mnist.data, mnist.target

# Función para graficar un dígito
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

# Tomamos el primer dígito
some_digit = x[0]
plot_digit(some_digit)

# Mostrar la imagen
plt.show()



