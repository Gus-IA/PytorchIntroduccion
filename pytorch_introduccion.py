import torch
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score

# ---- Numpy ----

# matrix de ceros, 5 filas y 3 columnas

x = torch.zeros(5, 3)
print(x)


# tensor con valores aleatorios

a = torch.randn(5, 3, 2)
print(a)


# tensor a partir de lista
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(b)


# tensor a partir de array

c = np.array([[1, 2, 3], [4, 5, 6]])
d = torch.from_numpy(c)
print(d)


# operaciones 

e = torch.rand(3, 3)
f = torch.rand(3, 3)

print(e,f)
print(e+f)


# idexado
# primera fila

print(c[0])


# primera fila, primera columna
print(c[0, 0])

# primera columna
print(c[0, :])


# troceado
print(c[:-1, 1:])


# añadir una dimensión extra
x.view(5, 3, 1).shape
print(c)


# usamos el -1 para asignar todos los valores restantes a una dimensión
x.view(-1).shape


x.numpy()
print(x)


# ---- Autograd ----

g = torch.tensor(1., requires_grad=True)
h = torch.tensor(1., requires_grad=True)
i = g + h

j = torch.tensor(3., requires_grad=True)
k = i * j


k.backward()

j.grad
print(j)


# ---- GPU ----

print(torch.cuda.is_available())

l = torch.randn(10000, 10000).cuda()
m = torch.randn(10000, 10000).cuda()

n = l*m


device = torch.device("cuda")

o = torch.randn((10000, 10000), device="cuda")
o = o.cuda()
o = o.to("cuda")
o = o.to(device)


# Cargar MNIST
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]

# Convertir etiquetas a string por seguridad en plt.title
Y = Y.astype(str)

# Parámetros de la cuadrícula
r, p = 3, 5
fig = plt.figure(figsize=(2*p, 2*r))

for _r in range(r):
    for _p in range(p):
        plt.subplot(r, p, _r*p + _p + 1)
        ix = random.randint(0, len(X)-1)
        img = X.iloc[ix].values  # fetch_openml devuelve un DataFrame por defecto
        plt.imshow(img.reshape(28,28), cmap='gray')
        plt.axis("off")
        plt.title(Y[ix])

plt.tight_layout()
plt.show()


# normalizamos los datos

X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., Y[:60000].astype(int), Y[60000:].astype(int)



# función de pérdida y derivada

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

def cross_entropy(output, target):
    logits = output[torch.arange(len(output)), target]
    loss = - logits + torch.log(torch.sum(torch.exp(output), axis=-1))
    loss = loss.mean()
    return loss



D_in, H, D_out = 784, 100, 10

# pesos del MLP (copiamos en gpu)
w1 = torch.tensor(np.random.normal(loc=0.0, 
          scale = np.sqrt(2/(D_in+H)), 
          size = (D_in, H)), requires_grad=True, device="cuda", dtype=torch.float)
b1 = torch.zeros(H, requires_grad=True, device="cuda", dtype=torch.float)

w2 = torch.tensor(np.random.normal(loc=0.0, 
          scale = np.sqrt(2/(D_out+H)), 
          size = (H, D_out)), requires_grad=True, device="cuda", dtype=torch.float)
b2 = torch.zeros(D_out, requires_grad=True, device="cuda", dtype=torch.float)

# convertimos datos a tensores y copiamos en gpu
X_t = torch.from_numpy(X_train.to_numpy().astype(np.float32)).cuda()
Y_t = torch.from_numpy(y_train.to_numpy().astype(np.int64)).cuda()


epochs = 100
lr = 0.8
log_each = 10
l = []
for e in range(1, epochs+1): 
    
    # forward
    h = X_t.mm(w1) + b1
    h_relu = h.clamp(min=0) # relu
    y_pred = h_relu.mm(w2) + b2

    # loss
    loss = cross_entropy(y_pred, Y_t)
    l.append(loss.item())

    # Backprop (calculamos todos los gradientes automáticamente)
    loss.backward()

    with torch.no_grad():
        # update pesos
        w1 -= lr * w1.grad
        b1 -= lr * b1.grad
        w2 -= lr * w2.grad  
        b2 -= lr * b2.grad
        
        # ponemos a cero los gradientes para la siguiente iteración
        # (sino acumularíamos gradientes)
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
    
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

# función de evaluación
def evaluate(x):
    h = x.mm(w1) + b1
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2) + b2
    y_probas = softmax(y_pred)
    return torch.argmax(y_probas, axis=1)

# predicciones
y_pred = evaluate(torch.from_numpy(X_test.values).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())

# mostramos el resultado
r, c = 3, 5
fig = plt.figure(figsize=(2*c, 2*r))

for _r in range(r):
    for _c in range(c):
        plt.subplot(r, c, _r * c + _c + 1)

        # obtenemos una posición aleatoria (no un índice del DataFrame)
        ix = random.randint(0, len(X_test) - 1)

        # usamos iloc para obtener la imagen y el label correctamente
        img = X_test.iloc[ix].values  # convertir a ndarray
        label = y_test.iloc[ix] if hasattr(y_test, "iloc") else y_test[ix]  # funciona para Series y arrays

        # convertimos a tensor correctamente con batch size = 1
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()  # shape: (1, features)

        # inferencia
        y_pred = evaluate(input_tensor)[0].item()  # obtenemos el número predicho

        # mostrar imagen
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"{label}/{y_pred}", color="green" if label == y_pred else "red")

plt.tight_layout()
plt.show()