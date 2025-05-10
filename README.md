# 🧠 MNIST-GAN com PyTorch

Este projeto implementa uma Rede Generativa Adversarial (GAN) simples utilizando o framework PyTorch para gerar imagens similares às do conjunto de dados MNIST (dígitos escritos à mão).

## ✨ Visão Geral

A arquitetura consiste em dois modelos treinados simultaneamente:

* **Gerador (Generator)**: Gera imagens falsas a partir de vetores de ruído aleatório.
* **Discriminador (Discriminator)**: Avalia se uma imagem é real (do conjunto MNIST) ou falsa (criada pelo gerador).

Esses modelos são treinados de forma adversarial até que o gerador produza imagens suficientemente realistas para enganar o discriminador.

## 📁 Estrutura do Projeto

```
.
├── gan_mnist.py       # Código principal com a definição e treinamento dos modelos
├── data/              # Pasta onde o PyTorch baixará o dataset MNIST
└── README.md          # Este arquivo
```

## 🧪 Requisitos

* Python 3.7+
* PyTorch
* torchvision
* matplotlib

Instale os pacotes necessários com:

```bash
pip install torch torchvision matplotlib
```

## 🚀 Como Executar

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/mnist-gan.git
cd mnist-gan
```

2. Execute o script:

```bash
python gan_mnist.py
```

Durante o treinamento, imagens geradas serão exibidas a cada 10 épocas.

## 🧮 Hiperparâmetros

* Latent dimension: `100`
* Tamanho do batch: `64`
* Épocas: `50`
* Learning rate: `0.0002`
* Função de perda: `BCELoss`
* Otimizador: `Adam`

## 🧊 Exemplo de Saída

A cada 10 épocas, o script gera uma grade de imagens como esta:

![Exemplo de saída](https://pytorch.org/tutorials/_images/mnist_fake_samples.png)

## 📚 Referência

A implementação foi inspirada na obra:

> **Goodfellow et al. (2014)** - Generative Adversarial Nets. *NeurIPS 2014*
> **Montgomery, D. C., & Runger, G. C. (2018)** - *Applied Statistics and Probability for Engineers*. Wiley.

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Sinta-se livre para usar e modificar.


