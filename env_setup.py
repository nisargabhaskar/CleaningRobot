from typing import Optional
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        """Initialize the GridWorld environment."""
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=int)
        self._completed_squares = np.zeros((size, size), dtype=int)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "completed_squares": gym.spaces.Box(-1, 1, shape=(self.size, self.size), dtype=int)
                # "dumpzone": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
                # "chargestation": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )
        # Define the actions the agent can take
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   # Move right (positive x)
            1: np.array([1, 0]),   # Move up (positive y)
            2: np.array([0, -1]),  # Move left (negative x)
            3: np.array([-1,0]),  # Move down (negative y)
        }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "completed_squares": self._completed_squares}
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with manhattan distance between agent and target
        """
        return {
            "squares_left": np.count_nonzero(self._completed_squares == 0)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self._completed_squares = np.zeros((self.size, self.size), dtype=int)
        num_obstacles = self.size
        rows = np.random.randint(0, self.size, num_obstacles)
        cols = np.random.randint(0, self.size, num_obstacles)
        for r, c in zip(rows, cols):
            self._completed_squares[r, c] = -1
        
        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._completed_squares[self._agent_location[0]][self._agent_location[1]] = 1

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        direction = self._action_to_direction[action]
        new_pos = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        if self._completed_squares[new_pos[0], new_pos[1]] != -1:
            if self._completed_squares[new_pos[0], new_pos[1]] == 0:
                self._completed_squares[new_pos[0], new_pos[1]] = 1
                self._agent_location = new_pos
                terminated = np.count_nonzero(self._completed_squares == 0) == 0 
                truncated = False
                reward = 10  if terminated else 1
            elif np.array_equal(new_pos, self._agent_location):
                reward = -0.05  
                terminated = False
                truncated = False
            else :
                self._agent_location = new_pos
                reward = -0.02 
                terminated = False
                truncated = False
            observation = self._get_obs()
            info = self._get_info()
            self.render_graphical()
            return observation, reward, terminated, truncated, info
        
        else:
            terminated = False      
            truncated = True
            reward = -1            
            self._agent_location = new_pos
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, truncated, info

    def render_graphical(self):
        """Render the environment as a simple grid using matplotlib."""

        grid = np.zeros((self.size, self.size, 3), dtype=int)
        grid[:, :, :] = 255
        for r in range(self.size):
            for c in range(self.size):
                if self._completed_squares[r, c] == -1:
                    grid[r, c] = [0, 0, 0] 
                elif self._completed_squares[r, c] == 1:
                    grid[r, c] = [0, 255, 0] 
        ar, ac = self._agent_location
        grid[ar, ac] = [255, 0, 0]
        # plt.imshow(grid)
        # plt.xticks([])
        # plt.yticks([])
        # plt.pause(0.1)