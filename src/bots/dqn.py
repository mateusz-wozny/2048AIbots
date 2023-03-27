import random
from game import Game
import torch.nn as nn
import torch
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, hidden_dim, drop_out):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(208, hidden_dim),
                        nn.Dropout(drop_out),

            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        x = x.to(device)
        return self.fc(x)
    

    
def train_game(game, it, gamma=0.95):
    batch_label, batch_output = [], []
    step = 1
    while True:
        Q_values = model(game.vector())
        Q_valid_values = [Q_values[a] if game.is_action_available(a) else float('-inf') for a in range(4)]
        best_action = torch.argmax(torch.tensor(Q_valid_values))

        reward = game.do_action(best_action)
        Q_star = Q_valid_values[best_action]
        vec = game.vector()
        with torch.no_grad():
            Q_next = model(vec)
        batch_output.append(Q_star)
        batch_label.append(reward + gamma * max(Q_next))

        if step % batch_size == 0 or game.game_over():
            if len(batch_label) == 0: 
                return
            optimizer.zero_grad()

            label_tensor = torch.stack(batch_label)
            output_tensor = torch.stack(batch_output)

            loss = criterion(output_tensor, label_tensor)
            loss.backward()
            optimizer.step()
            batch_output, batch_label=[],[]

            if game.game_over():
                print("epoch: {}, game score: {}".format(it, game.state().max()))
                return
        step += 1
        
def eval_game(game):
    model.eval()
    with torch.no_grad():
        for i in range(n_eval):
            game = Game()
            while not game.game_over():
                Q_values = model(game.vector())
                Q_valid_values = [Q_values[a] if game.is_action_available(a) else float('-inf') for a in range(4)]
                best_action = torch.argmax(torch.tensor(Q_valid_values))
                game.do_action(best_action)
            print(game.state())

batch_size = 128
hidden_dim = 256
drop_out = 0.2
n_epoch = 50
n_eval = 10
gamma = 0.95
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = Net(hidden_dim, drop_out)
model = model.to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
criterion = criterion.to(device)
print(sum(p.numel() for p in model.parameters()))
losses = []
scores = []
randoms = []

if __name__=="__main__":
    model.train()
    for it in range(n_epoch):
        game = Game()
        train_game(game, it, gamma)
    eval_game(game)
        