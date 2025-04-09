import socket
import threading
import random
import numpy as np
import json
import gym
from gym import spaces
import matplotlib.pyplot as plt

# ============ VANETEnv Class ============
class VANETEnv(gym.Env):
    def __init__(self, alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb):
        super(VANETEnv, self).__init__()

        # Action & Observation spaces
        self.action_space = spaces.Discrete(4)  # 0: inc beacon, 1: dec beacon, 2: inc power, 3: dec power
        self.observation_space = spaces.Tuple((
            spaces.Discrete(20),  # beacon rate
            spaces.Discrete(30),  # power
            spaces.Discrete(100)  # density
        ))

        # Q-Learning hyperparams
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = 1000
        self.num_steps = 10
        self.convergence_threshold = 0.01

        # Reward weights
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb

        # Q-table
        self.Q = np.zeros((20, 30, 4))

        # For persisting states across multiple calls
        self.current_state = None

        # Tracking for optional plotting
        self.timestamps = []
        self.rewards = []
        self.global_step_count = 0  # total step count across all training

    def select_action(self):
        b_idx = self.current_state[0] - 1
        p_idx = self.current_state[1] - 1
        # density = self.current_state[2] # not used explicitly in Q indexing
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[b_idx, p_idx])

    def action_to_delta(self, action):
        if action == 0:
            return 1, 0   # inc beacon
        elif action == 1:
            return -1, 0  # dec beacon
        elif action == 2:
            return 0, 3   # inc power
        elif action == 3:
            return 0, -3  # dec power

    def compute_neighbors(self, rho, p, new_p, beta=3):
        # Just an example from your code:  new_rho = rho * ((p / new_p)**(1/beta))
        # Make sure new_p != 0 to avoid division by zero
        if new_p <= 0:
            new_p = 1
        return rho * ((p / new_p) ** (1 / beta))

    def reward_function(self, cbr, p, p_prime):
        def H(x):
            return 1 if x >= 0 else 0

        def g(x, k):
            return x * (H(x) - 2 * H(x - k))

        # Weighted sum
        reward = (self.pi_b * g(cbr, self.kb)) \
                 - (self.pi_p_prime * abs(p - p_prime)) \
                 - (self.pi_p2 * g(p_prime, 20))
        return reward

    def update_q_table(self, action, reward, next_state):
        b_idx = next_state[0] - 1
        p_idx = next_state[1] - 1
        max_future_q = np.max(self.Q[b_idx, p_idx])

        old_Q = self.Q[self.current_state[0] - 1, self.current_state[1] - 1][action]
        new_Q = (1 - self.alpha) * old_Q + self.alpha * (reward + self.gamma * max_future_q)
        self.Q[self.current_state[0] - 1, self.current_state[1] - 1][action] = new_Q

    def step_once(self):
        action = self.select_action()
        b, p, rho = self.current_state

        delta_b, delta_p = self.action_to_delta(action)
        new_b = max(1, min(b + delta_b, 20))
        new_p = max(1, min(p + delta_p, 30))

        new_rho = self.compute_neighbors(rho, p, new_p, beta=3)
        next_state = (new_b, new_p, new_rho)

        # cbr ~ new_rho assumed
        cbr = new_rho
        reward = self.reward_function(cbr, new_p, p)

        # store for plotting
        self.global_step_count += 1
        self.timestamps.append(self.global_step_count)
        self.rewards.append(reward)

        # Q-table update
        self.update_q_table(action, reward, next_state)
        self.current_state = next_state
        return new_b, new_p

    def train_once(self, b, p, cbr_density):
        """
        Melakukan beberapa episode & steps untuk 1 data log.
        Di akhir, kembalikan beaconRate & power barunya.
        """
        # Init state
        self.current_state = (b, p, cbr_density)

        for episode in range(self.num_episodes):
            old_Q_table = np.copy(self.Q)
            for _ in range(self.num_steps):
                self.step_once()

            # Cek konvergensi
            if np.max(np.abs(self.Q - old_Q_table)) < self.convergence_threshold:
                break

        # Return final updated b, p
        return self.current_state[0], self.current_state[1]

    def plot_reward(self):
        """
        Opsional, panggil ini misal setelah beberapa request atau command khusus
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.rewards, label="Reward")
        plt.xlabel("Global Step")
        plt.ylabel("Reward")
        plt.title("Reward vs. Steps (Global)")
        plt.grid(True)
        plt.legend()
        plt.show()


# ============ Server Class ============
class Server:
    def __init__(self, host='localhost', port=5000,
                 alpha=0.1, gamma=0.9, epsilon=0.1,
                 pi_b=75, pi_p_prime=5, pi_p2=20, kb=0.6):
        self.host = host
        self.port = port

        # Buat environment persisten
        self.env = VANETEnv(alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb)

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}...")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            decoded_data = json.loads(data.decode('utf-8'))

            # if user sends e.g. {"plot": true}, we can show the plot
            if 'plot' in decoded_data:
                self.env.plot_reward()
                continue

            beacon_rate = decoded_data['beaconRate']
            power_transmission = decoded_data['transmissionPower']
            cbr_value = decoded_data.get('CBR', 0.5)  # default 0.5
            cbr_density = cbr_value * 100            # 0..1 -> 0..100

            # Jalankan training (atau "step" ringkas) pada environment persisten
            new_beacon_rate, new_power_transmission = self.env.train_once(
                beacon_rate, power_transmission, cbr_density
            )

            # Buat respon
            response = {
                "transmissionPower": new_power_transmission,
                "beaconRate": new_beacon_rate,
                "MCS": decoded_data.get("MCS", "N/A")
            }
            client_socket.sendall(json.dumps(response).encode('utf-8'))

        client_socket.close()


def main():
    host = 'localhost'
    port = 5000
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    pi_b = 75
    pi_p_prime = 5
    pi_p2 = 20
    kb = 0.6

    server = Server(
        host=host,
        port=port,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        pi_b=pi_b,
        pi_p_prime=pi_p_prime,
        pi_p2=pi_p2,
        kb=kb,
    )
    server.start_server()


if __name__ == "__main__":
    main()
