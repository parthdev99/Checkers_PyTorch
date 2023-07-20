import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ReinforcementModel(nn.Module):
    def __init__(self):
        super(ReinforcementModel, self).__init__()
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

reinforced_model = ReinforcementModel()
optimizer_reinforced = optim.Adadelta(reinforced_model.parameters())
criterion_reinforced = nn.MSELoss()

data = np.zeros((1, 32))
labels = np.zeros(1)
win = lose = draw = 0
winrates = []
learning_rate = 0.5
discount_factor = 0.95

num_generations = 300
num_games = 200
num_epochs_reinforced = 16
batch_size_reinforced = 256

for gen in range(num_generations):
    for game in range(num_games):
        temp_data = np.zeros((1, 32))
        board = expand(np_board())
        player = np.sign(np.random.random() - 0.5)
        turn = 0

        while True:
            moved = False
            boards = np.zeros((0, 32))

            if player == 1:
                boards = generate_next(board)
            else:
                boards = generate_next(reverse(board))

            scores = reinforced_model(torch.from_numpy(boards).float())
            max_index = np.argmax(scores.detach().numpy())
            best = boards[max_index]

            if player == 1:
                board = expand(best)
                temp_data = np.vstack((temp_data, compress(board)))
            else:
                board = reverse(expand(best))

            player = -player

            # Punish losing games, reward winners & drawish games reaching more than 200 turns
            winner = game_winner(board)

            if winner == 1 or (winner == 0 and turn >= 20):
                if winner == 1:
                    win += 1
                else:
                    draw += 1
                reward = 10
                old_prediction = reinforced_model(torch.from_numpy(temp_data[1:]).float())
                optimal_future_value = torch.ones_like(old_prediction)
                temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction)
                data = np.vstack((data, temp_data[1:]))
                labels = np.vstack((labels, temp_labels.detach().numpy()))
                break
            elif winner == -1:
                lose += 1
                reward = -10
                old_prediction = reinforced_model(torch.from_numpy(temp_data[1:]).float())
                optimal_future_value = -torch.ones_like(old_prediction)
                temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction)
                data = np.vstack((data, temp_data[1:]))
                labels = np.vstack((labels, temp_labels.detach().numpy()))
                break
            turn += 1

        if (game + 1) % 20 == 0:
            permutation = torch.randperm(data.shape[0] - 1)
            data_tensor = torch.from_numpy(data[1:]).float()
            labels_tensor = torch.from_numpy(labels[1:]).float()

            # Check if data_tensor and labels_tensor are not empty
            if data_tensor.size(0) > 0 and labels_tensor.size(0) > 0:
                for i in range(0, data_tensor.size(0), batch_size_reinforced):
                    start_index = i
                    end_index = min(i + batch_size_reinforced, data_tensor.size(0))
                    indices = permutation[start_index:end_index]

                    # Check if indices is not empty
                    if indices.size(0) > 0:
                        batch_data = data_tensor[indices]
                        batch_labels = labels_tensor[indices]

                        optimizer_reinforced.zero_grad()
                        outputs = reinforced_model(batch_data)
                        loss = criterion_reinforced(outputs, batch_labels)
                        loss.backward()
                        optimizer_reinforced.step()

                data = np.zeros((1, 32))
                labels = np.zeros(1)

    winrate = int((win + draw) / (win + draw + lose) * 100)
    winrates.append(winrate)

torch.save(reinforced_model.state_dict(), 'reinforced_model.pth')
print('Reinforcement Model updated by reinforcement learning and saved as "reinforced_model.pth".')

