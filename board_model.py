import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class BoardModel(nn.Module):
    def __init__(self):
        super(BoardModel, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

board_model = BoardModel()
optimizer_board = optim.Adam(board_model.parameters())
criterion_board = nn.BCELoss()

metrics = np.zeros((0, 10))
winning = np.zeros((0, 1))
data = boards_list

for board in data:
    temp = get_my_metrics(board)
    metrics = np.vstack((metrics, temp[1:]))
    winning = np.zeros((0, 1))

metrics_tensor = torch.from_numpy(metrics).float()
winning_tensor = torch.from_numpy(winning).float()

probabilistic_tensor = metrics_model(metrics_tensor)
probabilistic_tensor = torch.sign(probabilistic_tensor)

confidence_tensor = 1 / (1 + torch.abs(winning_tensor - probabilistic_tensor[:, 0]))

data_tensor = torch.from_numpy(data).float()
probabilistic_numpy = probabilistic_tensor.detach().numpy()

# Check if data_tensor and confidence_tensor are not empty
if data_tensor.size(0) > 0 and confidence_tensor.size(0) > 0:
    num_epochs_board = 32
    batch_size_board = 64

    for epoch in range(num_epochs_board):
        permutation = torch.randperm(data_tensor.size(0))
        for i in range(0, data_tensor.size(0), batch_size_board):
            indices = permutation[i:i+batch_size_board]

            # Check if indices is not empty
            if indices.size(0) > 0:
                batch_data = data_tensor[indices]
                batch_probabilistic = probabilistic_tensor[indices]
                batch_confidence = confidence_tensor[indices]

                optimizer_board.zero_grad()
                outputs = board_model(batch_data)
                loss = criterion_board(outputs, batch_probabilistic)
                loss.backward()
                optimizer_board.step()

    torch.save(board_model.state_dict(), 'board_model.pth')