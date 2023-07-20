import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plot

class MetricsModel(nn.Module):
    def __init__(self):
        super(MetricsModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

metrics_model = MetricsModel()
optimizer_metrics = optim.Adam(metrics_model.parameters())
criterion_metrics = nn.BCELoss()

start_board = expand(np_board())
boards_list = generate_next(start_board)
branching_position = 0
num_generated_games = 1000

while len(boards_list) < num_generated_games:
    temp = len(boards_list) - 1
    for i in range(branching_position, len(boards_list)):
        if possible_moves(reverse(expand(boards_list[i]))) > 0:
            boards_list = np.vstack((boards_list, generate_next(reverse(expand(boards_list[i])))))
    branching_position = temp

metrics = np.zeros((0, 10))
winning = np.zeros((0, 1))

for board in boards_list[:num_generated_games]:
    temp = get_my_metrics(board)
    metrics = np.vstack((metrics, temp[1:]))
    winning = np.vstack((winning, temp[0]))

metrics = torch.from_numpy(metrics).float()
winning = torch.from_numpy(winning).float()

num_epochs_metrics = 32
batch_size_metrics = 64

history = {'loss': [], 'acc': []}

for epoch in range(num_epochs_metrics):
    permutation = torch.randperm(metrics.size(0))
    for i in range(0, metrics.size(0), batch_size_metrics):
        indices = permutation[i:i+batch_size_metrics]
        batch_metrics, batch_winning = metrics[indices], winning[indices]

        optimizer_metrics.zero_grad()
        outputs = metrics_model(batch_metrics)
        loss = criterion_metrics(outputs, batch_winning)
        loss.backward()
        optimizer_metrics.step()

        # Calculate accuracy
        predicted_labels = (outputs > 0.5).float()
        accuracy = (predicted_labels == batch_winning).float().mean()

        # Save loss and accuracy for plotting
        history['loss'].append(loss.item())
        history['acc'].append(accuracy.item())

print('Metrics Model trained successfully.')




# for plotting
# History for accuracy
# plot.plot(history.history['acc'])
# plot.plot(history.history['val_acc'])
# plot.title('model accuracy')
# plot.ylabel('accuracy')
# plot.xlabel('epoch')
# plot.legend(['train', 'validation'], loc='upper left')
# plot.show()

# # History for loss
# plot.plot(history.history['loss'])
# plot.plot(history.history['val_loss'])
# plot.title('model loss')
# plot.ylabel('loss')
# plot.xlabel('epoch')
# plot.legend(['train', 'validation'], loc='upper left')
# plot.show()