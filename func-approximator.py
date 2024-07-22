import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.optim as optim

# Define the actual function
def actual_function(x):
    return x ** 3 - 3 * x

# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Generate data
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = actual_function(x)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Initialize the model, loss function and optimizer
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
actual_line, = ax.plot(x, y, 'r-', label='Actual Function')
predicted_line, = ax.plot(x, y, 'b-', label='NN Prediction')
ax.legend()

# Function to update the plot
def update(frame):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).numpy()

    predicted_line.set_ydata(y_pred)
    return predicted_line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Save the animation as a GIF
ani.save('quad function.gif', writer='pillow')

# Display the animation
plt.title("Neural Network Learning a Mathematical Function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()