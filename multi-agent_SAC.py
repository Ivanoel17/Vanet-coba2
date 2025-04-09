import socket
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
HOST = '127.0.0.1'
PORT = 5000
CBR_TARGET = 0.65
SNR_TARGET = 10.0  # Target SNR (for simplicity, this can be adjusted)
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1  # Fixed exploration rate
ENTROPY_WEIGHT = 0.2  # Entropy weight for exploration

# Simplified state discretization
POWER_BINS = [5, 15, 25, 30]
BEACON_BINS = [1, 5, 10, 20]
CBR_BINS = [0.0, 0.3, 0.6, 1.0]

# Initialize Q-table (state-action value function)
q_table = np.zeros((len(POWER_BINS), len(BEACON_BINS), len(CBR_BINS), 2))

# Define the actor (policy network) and critic (value network)
class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output 2 actions (increase or decrease)
            nn.Softmax(dim=-1)  # Softmax for probability distribution
        )
        self.critic = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output (state value)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

# Initialize the Actor-Critic model and optimizer
model = ActorCriticNetwork()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def discretize(value, bins):
    return np.digitize(value, bins) - 1

class SACServer:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((HOST, PORT))
        self.server.listen(1)
        print(f"Server listening on {HOST}:{PORT}")

    def calculate_reward(self, cbr, snr):
        # Reward based on both CBR and SNR (target values for each)
        reward_cbr = -abs(cbr - CBR_TARGET) * 100
        reward_snr = -abs(snr - SNR_TARGET) * 10  # Adjust the weight of SNR accordingly
        return reward_cbr + reward_snr

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs, _ = model(state_tensor)
        
        # Exploration (using epsilon-greedy with action probabilities)
        if random.random() < EPSILON:
            return random.choice([0, 1])  # 0: decrease, 1: increase
        return torch.argmax(action_probs).item()  # Select action with highest probability

    def update_model(self, state, action, reward, new_state):
        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        new_state_tensor = torch.tensor(new_state, dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        
        # Compute the predicted values from the model
        action_probs, state_value = model(state_tensor)
        _, next_state_value = model(new_state_tensor)

        # Compute the target value using the Bellman equation (value + entropy term)
        target_value = reward_tensor + DISCOUNT_FACTOR * next_state_value
        entropy_term = -torch.sum(action_probs * torch.log(action_probs))  # Entropy for exploration

        # Calculate the critic loss (Mean Squared Error)
        critic_loss = (state_value - target_value).pow(2).mean()

        # Calculate the actor loss (negative log likelihood with entropy)
        actor_loss = -torch.log(action_probs[action]) * (target_value - state_value) - ENTROPY_WEIGHT * entropy_term

        # Total loss
        total_loss = critic_loss + actor_loss.mean()

        # Update model weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    def handle_client(self, conn):
        while True:
            data = conn.recv(1024)
            if not data:
                break
            
            try:
                state = json.loads(data.decode())
                print(f"Received: {state}")
                
                # Current parameters
                current_power = state['transmissionPower']
                current_beacon = state['beaconRate']
                current_cbr = state['CBR']
                current_snr = state['SNR']
                current_mcs = state['MCS']  # MCS is not used in the learning process
                
                # Select action
                action = self.select_action((current_power, current_beacon, current_cbr))
                
                # Determine new values
                new_power = max(5, min(30, current_power + (-1 if action == 0 else 1)))
                new_beacon = max(1, min(20, current_beacon + (-1 if action == 0 else 1)))
                
                # Calculate reward
                reward = self.calculate_reward(current_cbr, current_snr)
                
                # Update model
                self.update_model(
                    (current_power, current_beacon, current_cbr),
                    action,
                    reward,
                    (new_power, new_beacon, current_cbr)
                )
                
                # Send response (no reward, but include MCS)
                response = {
                    'power': new_power,
                    'beacon': new_beacon,
                    'MCS': current_mcs  # Only send MCS back
                }
                conn.send(json.dumps(response).encode())
                print(f"Sent: {response}")
                
            except Exception as e:
                print(f"Error: {e}")
                break

    def start(self):
        while True:
            conn, addr = self.server.accept()
            print(f"Connected: {addr}")
            self.handle_client(conn)
            conn.close()

if __name__ == "__main__":
    server = SACServer()
    server.start()
