import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed


#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=784).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="D:\\projects\\summerProject2020\\project3", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="D:\\projects\\summerProject2020\\project3", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2
)

if __name__ == "__main__":
    # epochs = 20
    # for epoch in range(epochs):
    #     loss = 0
    #     for batch_features, _ in train_loader:
    #         # reshape mini-batch data to [N, 784] matrix
    #         # load it to the active device
    #         batch_features = batch_features.view(-1, 784).to(device)
    #
    #         # reset the gradients back to zero
    #         # PyTorch accumulates gradients on subsequent backward passes
    #         optimizer.zero_grad()
    #
    #         # compute reconstructions
    #         outputs = model(batch_features)
    #
    #         # compute training reconstruction loss
    #         train_loss = criterion(outputs, batch_features)
    #
    #         # compute accumulated gradients
    #         train_loss.backward()
    #
    #         # perform parameter update based on current gradients
    #         optimizer.step()
    #
    #         # add the mini-batch training loss to epoch loss
    #         loss += train_loss.item()
    #
    #     # compute the epoch training loss
    #     loss = loss / len(train_loader)
    #
    #     # display the epoch training loss
    #     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    # torch.save(model, "autoencoder")
    model = torch.load("D:\\projects\\summerProject2020\\project3\\autoencoder")
    iter_data = iter(test_dataset)
    features = next(iter_data)
    feature = features[0]

    # plot original image
    origin = feature.detach().numpy()
    origin = np.squeeze(origin)
    img1 = Image.fromarray(np.uint8(origin * 255), 'L')
    img1.show()

    # regeneration with auto-encoder
    feature1 = feature.view(-1, 784)
    feature1 = feature1.to(device)
    model.to(device)
    result = model(feature1)

    # plot generated image
    result = result.view(1, 28, 28)
    result = torch.squeeze(result)
    result = result.cpu()
    result = result.detach().numpy()
    img = Image.fromarray(np.uint8(result * 255), 'L')
    img.show()

    print(model)
