import scipy
from torch import nn
import torch
from torch import optim
import time
import numpy as np
from collections import Counter, deque
import os
from pathlib import Path
import pickle

from game_bots.game.wrappers import DQNGame, TDGame


class Net(nn.Module):
    def __init__(self, hidden_dim, drop_out, device):
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(208, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = x.to(self.device)
        return self.fc(x)


class DQNBot:
    def __init__(
        self,
        hidden_dim: int,
        drop_out: float,
        batch_size: int,
        lr: float,
        file_path: str = "weights/dqn_agent.pth",
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = file_path
        print(self.device)
        self.model = Net(hidden_dim, drop_out, self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(self.device)
        self.batch_size = batch_size

    def save_weights(self):
        torch.save(self.model.state_dict(), self.file_path)

    def load_weights(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

    def episode(self, it: int, gamma: float = 0.95):
        batch_label, batch_output = [], []
        step = 1
        game = DQNGame()

        while True:
            Q_values = self.model(game.vector())
            Q_valid_values = [
                Q_values[a] if game.is_move_available(a) else float("-inf")
                for a in range(4)
            ]
            best_action = torch.argmax(torch.tensor(Q_valid_values))

            tmp_reward = game.game.reward
            game.game, _ = game.make_move(best_action.item())
            reward = game.game.reward - tmp_reward
            Q_star = Q_valid_values[best_action]
            vec = game.vector()
            with torch.no_grad():
                Q_next = self.model(vec)
            batch_output.append(Q_star)
            batch_label.append(reward + gamma * max(Q_next))

            if step % self.batch_size == 0 or game.game_over():
                if len(batch_label) == 0:
                    return
                self.optimizer.zero_grad()

                label_tensor = torch.stack(batch_label)
                output_tensor = torch.stack(batch_output)

                loss = self.criterion(output_tensor, label_tensor)
                loss.backward()
                self.optimizer.step()
                batch_output, batch_label = [], []

                if game.game_over():
                    print("epoch: {}, game score: {}".format(it, game.state.max()))
                    return
            step += 1

    def train_loop(self, n_epochs: int, gamma: float):
        self.model.train()
        for epoch in range(n_epochs):
            self.episode(epoch, gamma)

    @torch.no_grad()
    def eval_loop(self, n_eval: int):
        scores = []
        max_tiles = []
        self.model.eval()
        for i in range(n_eval):
            game = DQNGame()
            while not game.game_over():
                Q_values = self.model(game.vector())
                Q_valid_values = [
                    Q_values[a] if game.is_move_available(a) else float("-inf")
                    for a in range(4)
                ]
                best_action = torch.argmax(torch.tensor(Q_valid_values))
                game.make_move(best_action.item())
            scores.append(game.score)
            max_tiles.append(game.state.max())
        return scores, max_tiles


class TDBot:
    def __init__(
        self,
        file_path="weights/td_agent.npy",
        lr=0.25,
        decay=0.95,
        low_lr_limit=0.001,
    ):
        self.file_path = file_path

        self.lr = lr
        self.decay = decay
        self.low_lr_limit = low_lr_limit

        self.num_feat, self.size_feat = 17, 16**4
        self.step = 0

        self.init_weights()

    def tuples(self, board: np.ndarray) -> np.ndarray:
        x_vert = (
            (board[0, :] << 12) + (board[1, :] << 8) + (board[2, :] << 4) + board[3, :]
        ).ravel()
        x_hor = (
            (board[:, 0] << 12) + (board[:, 1] << 8) + (board[:, 2] << 4) + board[:, 3]
        ).ravel()
        x_sq = (
            (board[:3, :3] << 12)
            + (board[1:, :3] << 8)
            + (board[:3, 1:] << 4)
            + board[1:, 1:]
        ).ravel()
        return np.concatenate([x_vert, x_hor, x_sq])

    @property
    def num_weights(self):
        return self.size_feat * self.num_feat

    def init_weights(self):
        self.weights = np.zeros((self.num_feat, self.size_feat))

    def save_weights(self, use_sparse: bool = True):
        if use_sparse:
            sparse_weights = scipy.sparse.csr_matrix(self.weights)
            with open(Path(self.file_path).with_suffix(".pickle"), "wb") as f:
                pickle.dump(sparse_weights, f)
        else:
            np.save(self.file_path, self.weights)

    def load_weights(self, file_path: str, use_sparse: bool = True):
        if use_sparse:
            with open(file_path, "rb") as f:
                sparse_weights = pickle.load(f)
            self.weights = sparse_weights.toarray()
        else:
            self.weights = np.load(file_path)

    def evaluate(self, board: np.ndarray) -> float:
        return np.sum([self.weights[i, f] for i, f in enumerate(self.tuples(board))])

    def update(self, board: np.ndarray, delta: float, learning: bool = True):
        if learning:
            for _ in range(4):
                for i, f in enumerate(self.tuples(board)):
                    self.weights[i, f] += delta
                board = np.transpose(board)
                for i, f in enumerate(self.tuples(board)):
                    self.weights[i, f] += delta
                board = np.rot90(np.transpose(board))

    def episode(self, learning: bool = True) -> TDGame:
        game = TDGame()
        state, old_label = None, 0
        while not game.game_over():
            best_value = -np.inf
            best_board, best_score = None, None
            for direction in range(4):
                new_board, new_score, valid = game.pre_move(
                    game.board, game.score, direction
                )
                if valid:
                    value = self.evaluate(new_board)
                    if value > best_value:
                        best_value = value
                        best_board, best_score = new_board, new_score
            if state is not None:
                delta = (
                    (best_score - game.score + best_value - old_label)
                    * self.lr
                    / self.num_feat
                )
                self.update(state, delta, learning)

            game.board, game.score = best_board, best_score

            state, old_label = game.board.copy(), best_value
            game.new_tile()

        delta = -old_label * self.lr / self.num_feat
        self.update(state, delta, learning)

        self.step += 1
        return game

    def decay_lr(self):
        self.lr = round(max(self.lr * self.decay, self.low_lr_limit), 4)

    def train_loop(
        self,
        n_train: int,
        saving: bool,
    ):
        last_100_high_tile = deque(maxlen=100)
        global_start = time.time()
        for i in range(n_train):
            if self.step % 100 == 0 and self.lr > self.low_lr_limit:
                self.decay_lr()

            game = self.episode()
            last_100_high_tile.append(2 ** game.board.max())

            if (i + 1) % 100 == 0:
                print(
                    f"Best tile at step {i} is {np.max(last_100_high_tile)} with average tile {int(np.mean(last_100_high_tile))}"
                )
        total_time = int(time.time() - global_start)
        print(f"Total time = {total_time // 60} min {total_time % 60} sec")
        if saving:
            self.save_weights()

    def eval_loop(self, n_eval: int):
        scores = []
        max_tiles = []
        for _ in range(n_eval):
            game = self.episode(learning=False)
            scores.append(game.score)
            max_tiles.append(2 ** game.board.max())
        return scores, max_tiles


class TDBotsmall(TDBot):
    def __init__(
        self,
        file_path="weights/td_small_agent.npy",
        lr=0.25,
        decay=0.95,
        low_lr_limit=0.001,
    ):
        super().__init__(file_path, lr, decay, low_lr_limit)
        self.num_feat, self.size_feat = 24, 16**2
        self.init_weights()

    def tuples(self, x):
        x_vert = ((x[:3, :] << 4) + x[1:, :]).ravel()
        x_hor = ((x[:, :3] << 4) + x[:, 1:]).ravel()
        return np.concatenate([x_vert, x_hor])


def run_dqn_bot(n_train: int = 100, n_eval: int = 10, training: bool = True):
    batch_size = 64
    hidden_dim = 128
    drop_out = 0.1
    gamma = 0.99
    lr = 0.0025
    dqn_agent = DQNBot(hidden_dim, drop_out, batch_size, lr)

    if training:
        dqn_agent.train_loop(n_train, gamma)

    if not training and os.path.exists("weights/dqn_agent.pth"):
        dqn_agent.load_weights("weights/dqn_agent.pth")
    scores, max_tiles = dqn_agent.eval_loop(n_eval)
    print(np.mean(scores), np.mean(max_tiles, dtype=int))
    print(Counter(max_tiles))


def run_td_bot(
    n_train: int = 100,
    n_eval: int = 10,
    training: bool = True,
    saving: bool = False,
    resume: bool = True,
    ckpt_path: str = "weights/td_agent.npy",
    save_path: str = "weights/td_agent.npy",
    use_sparse: bool = True,
):
    td_agent = TDBotsmall(file_path=save_path)
    if resume and os.path.exists(ckpt_path):
        print("Load weights")
        td_agent.load_weights(ckpt_path, use_sparse)
    if training:
        td_agent.train_loop(n_train, saving=saving)
    scores, max_tiles = td_agent.eval_loop(n_eval)
    print(np.mean(scores), np.mean(max_tiles, dtype=int))
    print(Counter(max_tiles))


if __name__ == "__main__":
    run_td_bot(
        n_train=2000,
        n_eval=50,
        training=True,
        resume=False,
        ckpt_path="weights/td_agent.pickle",
        save_path="weights/tdsmall_agent.npy",
    )
