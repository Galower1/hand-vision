import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import SignsDataset, get_labels
from models import ANN

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")


dataset = SignsDataset()

BATCH_SIZE = 16
SHUFFLE = False
EPOCHS = 300

train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


model = ANN(input_nodes=30 * 42 * 3, features=15).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)


sample, label = dataset[7]
sample = sample.unsqueeze(0).to(device)
word = dataset.labels[label]

pred: list = model(sample).tolist()[0]
result = dataset.labels[pred.index(max(pred))]

print("EXPECTED", word, "GOT", result)
# torch.save(model.state_dict(), "model.pth")