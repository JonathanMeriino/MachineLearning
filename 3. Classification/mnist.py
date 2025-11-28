import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

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


