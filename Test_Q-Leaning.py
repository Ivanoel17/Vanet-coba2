import socket
import threading
import random
import numpy as np
import json
import gym
from gym import spaces
import logging

# Konfigurasi Logging agar outputnya sama detailnya dengan SAC
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define beta value (used in vehicle density adjustment formula)
beta = 3  

# Define a custom OpenAI Gym environment for Q-Learning in VANETs
class VANETEnv(gym.Env):
    def __init__(self, alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb):
        super(VANETEnv, self).__init__()

        # Define the action space and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(20),
            spaces.Discrete(30),
            spaces.Discrete(100)
        ))

        # Q-Learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Weights for the reward function
        self.pi_b = pi_b
        self.pi_p_prime = pi_p_prime
        self.pi_p2 = pi_p2
        self.kb = kb

        # Initialize the Q-table
        self.Q = np.zeros((20, 30, 4))
        self.current_state = None

    def select_action(self):
        action = random.choice(["increase beacon rate", "decrease beacon rate", "increase power", "decrease power"])
        return action
        
    def action_to_delta(self, action):
        if action == "increase beacon rate":
            return 1, 0
        elif action == "decrease beacon rate":
            return -1, 0
        elif action == "increase power":
            return 0, 3
        elif action == "decrease power":
            return 0, -3
        return 0, 0

    def compute_neighbors(self, rho, p, new_p):
        return rho * ((p / new_p) ** (1 / beta)) if new_p != 0 else rho
    
    def reward_function(self, cbr, p, p_prime):
        return -abs(cbr - 0.65) - abs(p - p_prime)

    def step(self, beacon_rate, power, vehicle_density):
        action = self.select_action()
        delta_b, delta_p = self.action_to_delta(action)

        new_b = max(1, min(beacon_rate + delta_b, 20))
        new_p = max(1, min(power + delta_p, 30))

        new_rho = self.compute_neighbors(vehicle_density, power, new_p)
        cbr = new_rho / 100  
        snr = random.uniform(30, 45)  

        reward = self.reward_function(cbr, new_p, power)
        
        return new_b, new_p, cbr, snr, action
    
# Server class to handle incoming connections
class Server:
    def __init__(self, host='127.0.0.1', port=5000, alpha=0.1, gamma=0.9, epsilon=0.1, pi_b=75, pi_p_prime=5, pi_p2=20, kb=0.6):
        self.host = host
        self.port = port
        self.env = VANETEnv(alpha, gamma, epsilon, pi_b, pi_p_prime, pi_p2, kb)

    def start_server(self):
        """Start the Q-learning server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        logger.info(f"Server listening on {self.host}:{self.port}...")

        while True:
            client_socket, client_address = server_socket.accept()
            logger.info(f"Connection from {client_address}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        """Handle incoming client requests."""
        while True:
            try:
                data = client_socket.recv(4096)
                if not data:
                    break
                decoded_data = json.loads(data.decode('utf-8'))

                response = self.process_car_log(decoded_data)  # 🔥 FIXED: Panggil method yang benar
                client_socket.sendall(json.dumps(response).encode())
            except Exception as e:
                logger.error(f"Error while handling client data: {e}")
                break
        client_socket.close()

    def process_car_log(self, car_log):  # 🔥 FIXED: Tambahkan method ini di dalam Server
        """Process incoming car data."""
        while isinstance(car_log, list):  
            car_log = car_log[0]  

        beacon_rate = car_log.get('beaconRate', 10)
        power_transmission = car_log.get('transmissionPower', 15)
        vehicle_density = car_log.get('CBR', 0.5) * 100
        neighbors = car_log.get('neighbors', {}).get('distance', random.uniform(5, 50))
        vehicle_id = car_log.get("vehID", "Unknown")

        new_beacon_rate, new_power_transmission, cbr, snr, action = self.env.step(
            beacon_rate, power_transmission, vehicle_density
        )

        logger.debug(f"Vehicle ID: {vehicle_id} | Selected action: {action}")
        logger.debug(f"CBR: {cbr:.4f}, SNR: {snr:.1f}, Neighbors: {neighbors:.2f}m")

        response = {
            "vehID": vehicle_id,
            "transmissionPower": new_power_transmission,
            "beaconRate": new_beacon_rate,
            "MCS": car_log.get("MCS", "N/A"),
            "CBR": round(cbr, 4),
            "SNR": round(snr, 2),
            "neighbors": {"distance": round(neighbors, 2)}
        }
        return response


# Main function
def main():
    server = Server()
    server.start_server()

if __name__ == "__main__":
    main()
