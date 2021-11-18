# Otimizando os Parametros.

# Treinar o modelo é um processo iterativo, em cada iteração(epochs) o modelo
# advinha um output, calcula o erro de seu palpite(loss), coleta as derivadas 
# do erro respeitando os parametros e otimiza esses parametros usando um
# gradiente descendente.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Os hyperparametros são parametros ajustaveis que te deixam controlar o processo
# de otimização do modelo. Diferentes valores dos hyperparametros podem impactar
# no treinamento do modelo e taxa de convergência.

# Definimos os seguintes hyperparametros no treinamento

# -> Numero de Epochs (O numero de vezes para iterar no dataset)
# -> Bach Size (Numero de amostra de dados propagados atraves da rede
# antes de atualizar o hiperparametros)
# -> Learning Rate (Quantas atualizações em cima dos parametros do modelo em
# cada batch/epoch. Valores pequenos tendem a desacelerar a velocidade do 
# aprendizado, enquanto grandes valores resultam em comportamento imprevisivel
# durante o treinamento)

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Cada Epoch consiste em duas partes principais

# -> The Train loop (Itera sobre o dataset em treinamento e tenta convergir os 
# melhores parametros)
# -> The validation/Test Loop(Itera sobre dataset de test para checar se a 
# performance do modelo esta melhorando)


# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Loss Function mede o nivel de dissimilaridade dos resultados obtidos com 
# o resultado desejado, nessa função queremos sempre minimizar durante o 
# treinamento.

# Para calcular essa função, fazemos uma previsao usando os impust dos dados 
# segmentados, e comparamos contra os dados reais.

# Funções de loss functions

# nn.MSELoss (Mean Square Error) for regression tasks
# nn.NLLLoss (Negative Log Likelihood) for classification. 
# nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()


# Otimização

# O processo de otimização consiste em ajustar os parametros do modelo para
# reduzir o erro do modelo em cada etapa do treinamento.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#LOOP DE TREINAMENTO


# Definimos o train_loop sobre a otimização, e o test_loop para avaliar
# a performance do modelo contra os dados de teste.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Inicializamos a Loss Function e a Otimização, passamos para o train_loop
    # e test_loop.


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")