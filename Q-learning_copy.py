import socket
import threading
import random
import numpy as np
import json
import gym
from gym import spaces
import matplotlib.pyplot as plt

# Define beta value (used in vehicle density adjustment formula)
beta = 3

# Define a custom OpenAI Gym environment for Q-Learning in VANETs (Vehicular Ad-Hoc Networks)
class VANETEnv(gym.Env):
    def __init__(self, car_logs, alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb):
        super(VANETEnv, self).__init__()  # Initialize the parent class (gym.Env)

        # Define the action space and observation space for the RL agent
        self.action_space = spaces.Discrete(4)  # Actions: increase/decrease beacon rate or power transmission
        self.observation_space = spaces.Tuple((
            spaces.Discrete(20),  # Beacon rate
            spaces.Discrete(30),  # Power transmission
            spaces.Discrete(100)  # Vehicle density (dianggap mewakili CBR * 100)
        ))

        # Q-Learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = 1000
        self.num_steps = 10
        self.convergence_threshold = 0.01

        # Weights for the reward function
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb

        # Initialize the Q-table with zeros
        self.Q = np.zeros((20, 30, 4))

        # Store car logs for initializing states
        self.car_logs = car_logs
        self.current_state = None

        # MODIFIKASI: simpan CBR terakhir yang dihitung di step_once
        self.last_cbr = 0.0

    def select_action(self):
        b_idx = self.current_state[0] - 1
        p_idx = self.current_state[1] - 1
        # density = self.current_state[2] (tidak dipakai eksplisit di Q-index)
        if random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[b_idx, p_idx])

    def action_to_delta(self, action):
        # Action map: 0 = increase beacon, 1 = decrease beacon, 2 = increase power, 3 = decrease power
        if action == 0:
            return 1, 0
        elif action == 1:
            return -1, 0
        elif action == 2:
            return 0, 3
        elif action == 3:
            return 0, -3

    def compute_neighbors(self, rho, p, new_p, beta):
        if new_p <= 0:
            new_p = 1
        return rho * ((p / new_p) ** (1 / beta))

    def reward_function(self, cbr, p, p_prime):
        def H(x):
            return 1 if x >= 0 else 0

        def g(x, k):
            return x * (H(x) - 2 * H(x - k))

        # Reward function based on CBR and power differences
        reward = (
            self.pi_b * g(cbr, self.kb)
            - (self.pi_p_prime * abs(p - p_prime))
            - (self.pi_p2 * g(p_prime, 20))
        )
        return reward

    def update_q_table(self, action, reward, next_state):
        b_idx = next_state[0] - 1  # new beacon index
        p_idx = next_state[1] - 1  # new power index

        max_future_q = np.max(self.Q[b_idx, p_idx])
        current_q = self.Q[self.current_state[0] - 1, self.current_state[1] - 1][action]
        self.Q[self.current_state[0] - 1, self.current_state[1] - 1][action] = \
            (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)

    def step_once(self):
        action = self.select_action()
        b, p, rho = self.current_state

        delta_b, delta_p = self.action_to_delta(action)
        new_b = max(1, min(b + delta_b, 20))
        new_p = max(1, min(p + delta_p, 30))

        # Optional: print info changes
        if delta_b == -1:
            print(f"Decreased Beacon Rate from {b} to {new_b}")
        if delta_p == -3:
            print(f"Decreased Power Transmission from {p} to {new_p}")

        # Adjust vehicle density (treated as cbr * 100) based on the new power transmission
        new_rho = self.compute_neighbors(rho, p, new_p, beta)
        next_state = (new_b, new_p, new_rho)

        # cbr = new_rho (skala [0..100]?). Assume cbr in [0..100]
        cbr = new_rho
        self.last_cbr = cbr  # MODIFIKASI: simpan cbr terakhir
        reward = self.reward_function(cbr, new_p, p)

        self.update_q_table(action, reward, next_state)
        self.current_state = next_state
        return new_b, new_p

    def train_once(self, b, p, cbr_density):
        """
        Menjalankan beberapa episode & steps untuk data log;
        Mengembalikan (beaconRate, power, cbr_terakhir).
        """
        self.current_state = (b, p, cbr_density)

        for episode in range(self.num_episodes):
            old_Q_table = np.copy(self.Q)
            for _ in range(self.num_steps):
                new_beacon_rate, new_power_transmission = self.step_once()

            # cek konvergensi
            if np.max(np.abs(self.Q - old_Q_table)) < self.convergence_threshold:
                break

        # Kembalikan final state + cbr
        return self.current_state[0], self.current_state[1], self.last_cbr

    def plot_reward(self):
        plt.ion()
        plt.figure(figsize=(10, 6))
        plt.plot(self.timestamps, self.rewards, label="Reward")
        plt.xlabel("Global Step")
        plt.ylabel("Reward")
        plt.title("Reward vs. Steps (Global)")
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.pause(0.1)
        plt.ioff()


# ============ Server Class ============ 
class Server:
    def __init__(self, host='localhost', port=5000,
                 alpha=0.1, gamma=0.9, epsilon=0.1,
                 pi_b=75, pi_p_prime=5, pi_p2=20, kb=0.6):
        self.host = host
        self.port = port
        self.env = VANETEnv([], alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb)
        self.request_count = 0
        self.plot_frequency = 100

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

            self.request_count += 1
            print(f"Request Count: {self.request_count}")

            # Ambil param input
            car_id = decoded_data.get('car_id', 0)
            beacon_rate = decoded_data['beaconRate']
            power_transmission = decoded_data['transmissionPower']
            cbr_value = decoded_data.get('CBR', 0.5)
            cbr_density = cbr_value * 100

            # Jalankan train_once => dapatkan (beacon, power, cbr_terakhir)
            new_beacon_rate, new_power_transmission, last_cbr = \
                self.env.train_once(beacon_rate, power_transmission, cbr_density)

            # Tampilkan cbr ke terminal dengan 5 digit
            print(f"[INFO] Updated: Beacon Rate={new_beacon_rate}, "
                  f"Power={new_power_transmission}, "
                  f"CBR={last_cbr/100:.5f}")

            # Plot setiap 100 request (opsional)
            if self.request_count % self.plot_frequency == 0:
                self.env.plot_reward()

            # Kirim JSON response (5 digit di belakang koma)
            response = {
                "transmissionPower": new_power_transmission,
                "beaconRate": new_beacon_rate,
                # Bawa kembali cbr dengan skala [0..1], dan 5 digit desimal
                "CBR": float(f"{(last_cbr/100.0):.5f}"),
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
