# PyTorch MNIST MLP Demo 🧠⚡

Este repositorio contiene una implementación básica de una red neuronal multicapa (MLP) desde cero utilizando PyTorch, entrenada para clasificar dígitos del conjunto de datos MNIST.

## 🔍 Descripción

El proyecto muestra paso a paso cómo usar:

- Tensores en PyTorch (`torch.tensor`, `.view()`, `.cuda()`)
- Operaciones básicas de autograd
- Entrenamiento de una red neuronal simple en GPU
- Visualización de resultados con `matplotlib`
- Evaluación de resultados con `accuracy_score` de `sklearn`

La red neuronal implementada tiene la siguiente arquitectura:

- Entrada: 784 (28x28 píxeles)
- Capa oculta: 100 neuronas (ReLU)
- Capa de salida: 10 neuronas (Softmax)

## 📦 Requisitos

Instala los paquetes necesarios con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
