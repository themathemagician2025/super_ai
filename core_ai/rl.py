# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from collections import deque
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime
import gym
import neat
from config import PROJECT_CONFIG, MODELS_DIR, LOG_DIR, RAW_DIR, export_config_to_dict
from model import SelfModifyingGenome, MathemagicianModel  # Assumed modules
from agents import MathemagicianAgent  # Assumed module

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.join(BASE_DIR, 'rl')

# Ensure directories exist
for directory in [LOG_DIR, RAW_DIR, MODELS_DIR, RL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'rl.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global variables
population: List[neat.DefaultGenome] = []
config: Optional[neat.Config] = None
model_version: str = "0.0.1"
buffer: deque = deque(maxlen=200)
prediction_count: int = 0
agent: Optional[MathemagicianAgent] = None

class DQN(nn.Module):
    """Deep Q-Network for RL."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    """DQN agent with replay buffer and target network."""
    def __init__(self, env: gym.Env, lr: float = 0.001, gamma: float = 0.99, epsilon: float = 1.0, buffer_size: int = 10000):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = 32

    def act(self, state: np.ndarray) -> int:
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Update Q-network with experience."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self) -> None:
        """Sync target network with Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

def load_model(genome_path: str = "winner.pkl", config_path: str = "config.txt") -> Tuple[List[Any], Optional[neat.Config], str]:
    """Load or initialize a NEAT population."""
    global agent
    try:
        genome_path = os.path.join(MODELS_DIR, genome_path)
        config_path = os.path.join(BASE_DIR, config_path)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        if not os.path.exists(genome_path):
            logger.warning(f"Model file not found at {genome_path}, creating dummy population")
            population = [config.genome_type(i) for i in range(10)]
            for genome in population:
                genome.configure_new(config.genome_config)
            version = "0.0.1"
        else:
            with open(genome_path, "rb") as f:
                winner_genome = pickle.load(f)
            population = [winner_genome]
            for i in range(9):
                new_genome = neat.DefaultGenome(i + 1)
                new_genome.configure_crossover(winner_genome, winner_genome, config.genome_config)
                new_genome.mutate(config.genome_config)
                population.append(new_genome)
            version = getattr(winner_genome, 'version', '1.0.0')

        best_genome = max(population, key=lambda g: g.fitness if hasattr(g, 'fitness') else 0)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        agent = MathemagicianAgent(model=net)
        logger.info(f"Model version {version} loaded with population size {len(population)}")
        return population, config, version
    except Exception as e:
        logger.error(f"Error loading NEAT model: {e}")
        return [], None, "0.0.1"

def save_model(filename: str = "winner.pkl") -> None:
    """Save the best genome."""
    global population
    try:
        best_genome = max(population, key=lambda g: g.fitness if hasattr(g, 'fitness') else 0)
        best_genome.version = model_version
        filepath = os.path.join(MODELS_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(best_genome, f)
        logger.info(f"Best model saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for predictions."""
    global prediction_count, buffer
    try:
        data = request.json
        x = data.get("x", 0)
        result = agent.solve(x)
        buffer.append((x, result))
        prediction_count += 1
        return jsonify({"result": result})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    """Return current model status."""
    try:
        fitnesses = [g.fitness for g in population if hasattr(g, 'fitness')]
        avg_fitness = np.mean(fitnesses) if fitnesses else "Not evaluated"
        return jsonify({
            "version": model_version,
            "population_size": len(population),
            "prediction_count": prediction_count,
            "average_fitness": avg_fitness,
            "status": "running"
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

def train_dqn(env_name: str, num_episodes: int = 100) -> List[float]:
    """Train a DQN agent."""
    env = gym.make(env_name)
    dqn_agent = DQNAgent(env)
    rewards = []
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            dqn_agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        dqn_agent.update_target_network()
        rewards.append(total_reward)
        logger.info(f"DQN Episode {episode}/{num_episodes}, reward={total_reward:.2f}")
    env.close()
    return rewards

def integrate_neat_with_rl(env_name: str, config_path: str, num_episodes: int = 10) -> Dict:
    """Integrate NEAT with RL environment."""
    env = gym.make(env_name)
    config_path = os.path.join(BASE_DIR, config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    def eval_genome(genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        state = env.reset()
        for _ in range(200):  # Max steps per episode
            action = int(np.argmax(net.activate(state)))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    winner = p.run(eval_genome, num_episodes)
    env.close()
    return {'winner': winner, 'stats': stats}

if __name__ == "__main__":
    # Initialize globals
    population, config, model_version = load_model()
    if not agent:
        agent = MathemagicianAgent(model=neat.nn.FeedForwardNetwork.create(population[0], config))

    # Train DQN
    logger.info("Training with DQN...")
    dqn_rewards = train_dqn("CartPole-v1")
    logger.info(f"DQN rewards: {dqn_rewards[-5:]}")

    # Train NEAT
    logger.info("Training with NEAT...")
    neat_result = integrate_neat_with_rl("CartPole-v1", "config.txt")
    logger.info(f"NEAT winner fitness: {neat_result['winner'].fitness:.2f}")

    # Run Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)

"""
Reinforcement learning module for the Mathemagician system.
"""

# Placeholder for future RL implementation
def train_rl_agent():
    print("Reinforcement learning module is under development.")