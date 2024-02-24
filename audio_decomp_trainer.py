import audio_decomp_module as adm
from torch.utils.data import DataLoader


TRAIN_MAX_LENGTH = 27719408
TEST_MAX_LENGTH = 18979538

# Create dataset
# dataset = AudioDataset(root_dir='./dataset')

# # Create data loader
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Split dataset into training set and validation set
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset = adm.AudioDataset(root_dir='./data/train')
test_dataset = adm.AudioDataset(root_dir='./data/test')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Create model
num_sources = 2 * 4  # replace with the number of sources in your task
model = adm.SourceSeparationModel(num_sources)

device = adm.torch.device("cuda" if adm.torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = adm.nn.MSELoss()
optimizer = adm.torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
print_every = 2000  # print every 2000 mini-batches
best_val_loss = float('inf')  # initialize best validation loss

for epoch in range(num_epochs):  # loop over the dataset multiple times
    model.train()  # set the model to training mode
    running_loss = 0.0
    for i, batch in enumerate(train_loader, 0):
        inputs = batch['mixture']
        instruments = batch['instruments']
        labels = batch['labels']

        print('inputs', inputs.shape, type(inputs))
        print('instruments', len(instruments), type(instruments))
        print('instrument 0', instruments[0].shape, type(instruments[0]))

        # Concatenate instrument tensors along the channel dimension (assuming it's dimension 1)
        targets = adm.torch.cat(instruments, dim=1)

        inputs = inputs.to(device)
        targets = targets.to(device)  # Ensure targets are also moved

        # forward + backward + optimize
        outputs = model(inputs)
        print('outputs', outputs.shape, type(outputs))
        print('targets', targets.shape, type(targets))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Zero the gradient buffers
        optimizer.zero_grad()   

        # Print statistics for every iteration
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
# validation
model.eval()  # set the model to evaluation mode
with adm.torch.no_grad():
    val_loss = 0.0
    for i, batch in enumerate(test_loader, 0):
        inputs = batch['mixture']
        instruments = batch['instruments']
        labels = batch['labels']

        # Stack instrument tensors along a new dimension to create the targets
        targets = adm.torch.stack(instruments)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    val_loss /= len(test_loader)
    print(f'Validation Loss: {val_loss:.4f}')

    # save the model if validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
        adm.torch.save(model, 'new_model.pth')
        best_val_loss = val_loss

print('Finished Training')
