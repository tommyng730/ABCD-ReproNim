
# Generate some data and return the pandas dataframe
num_obs = 1000
num_spiral = 4
noise_amt = 1

X, y, df = fx.generateSpiral(num_obs, num_spiral, noise = noise_amt)

# Plot the data and use color to view classes
fig, ax = plt.subplots(figsize = (7, 7))
sns.scatterplot(x = 'x', y = 'y', hue = 'class', palette = 'coolwarm', data = df, s = 75)


# Create a random numpy array
z = np.random.rand(50, 50)

# Recast this to a tensor
z = torch.tensor(z, dtype = torch.float)

print(z.requires_grad)



class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float)
        self.y = torch.tensor(y, dtype = torch.float)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i, :], self.y[i]


# Generate some spiral data
X, y, df = fx.generateSpiral(1000, 4, noise = 1.0)

# Split into train and test
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33, random_state=42)

# We first create datasets of our custom dataset class for train/test
tr_data = MyDataset(X_tr, y_tr)
te_data = MyDataset(X_te, y_te)

# Now create the instances of the train/test loaders
tr_loader = DataLoader(tr_data, batch_size = 50, shuffle = True)
te_loader = DataLoader(te_data, batch_size = 50, shuffle = True)


# The parameters we can tune
act = nn.ReLU() # Set activation function
lr = .01 # Set the learning rate
n_epochs = 250 # How long to train for

n = Net(act) # Create an instance of our networks
criterion = nn.BCELoss() # Our loss function

net, perf = train_model(tr_loader, te_loader, n, criterion, lr = lr, n_epochs = n_epochs)
fx.modPlot(X_tr, y_tr, X_te, y_te, net, perf, cmap = 'coolwarm')


# Get the observation
img, dig = mnist_train.__getitem__(0)

print(f'Shape of image: {img.shape} | MNIST Number: {dig} | Requires Grad: {img.requires_grad}')

# Plot
fig, ax = plt.subplots(3, 3, figsize = (8, 8))

# Random vals
idx = np.random.randint(1000, size = 9)

curr_idx = 0

# Plot images in a grid
for ii in [0, 1, 2]:
    for jj in [0, 1, 2]:
        img, dig = mnist_train.__getitem__(idx[curr_idx]) # Retrieve observation
        ax[ii, jj].imshow(img.squeeze(), cmap = 'bone') # Plot it
        curr_idx += 1

The added weight decay allows us to uncover variability in pixel intensity
[The CNN allows us to uncover low level features and relationships among these
The CNN has substantially fewer trainable parameters and may be able to generalize better
Flattening our 2D images obscures import context embedded in that native 2D space


class mnistCNN(nn.Module):

    def __init__(self):

        nn.Module.__init__(self)

        self.model = nn.Sequential(
                        # Convolution layer 1
                        nn.Conv2d(1, 4, kernel_size = 4),
                        nn.MaxPool2d(2, 2),
                        nn.ReLU(),

                        # Convolution layer 2
                        nn.Conv2d(4, 8, kernel_size = 4),
                        nn.MaxPool2d(2, 2),
                        nn.ReLU(),

                        # Linear layer
                        nn.Flatten(),
                        nn.Linear(128, 10),
                        nn.LogSoftmax()
                                    )


class mnistDense(nn.Module):

    def __init__(self):

        nn.Module.__init__(self)

        self.model = nn.Sequential(

                            # Layer 1
                            nn.Flatten(),
                            nn.Linear(784, 10),
                            nn.ReLU(),

                            # Layer 2
                            nn.Linear(10, 10),
                            nn.ReLU(),

                            # Layer 3
                            nn.Linear(10, 10),
                            nn.LogSoftmax()
                                    )
