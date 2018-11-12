"""Modified from https://github.com/seanbae/gym-snake."""

from collections import deque
from enum import IntEnum

import gym
from gym.envs.classic_control import rendering
from gym.error import InvalidAction, ResetNeeded
from gym.spaces import Box, Discrete
from gym.utils import seeding
import numpy as np


# Actions
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Rewards
STEP_REWARD = -1
FOOD_REWARD = 100
WIN_REWARD = 1000
DEATH_REWARD = -1000


def _opposite_dir(dir):
    if dir == LEFT: return RIGHT
    if dir == UP: return DOWN
    if dir == RIGHT: return LEFT
    if dir == DOWN: return UP
    return None


class _Snake:
    def __init__(self, x, y):
        self.pos = (x, y)
        self.body = deque([(x, y)])
        self.dir = None


class SnakeEnv(gym.Env):
    """A Gym environment implementing a clone of the Snake game."""
    metadata = {'render.modes': ['human']}

    def __init__(self, width=5, height=5):
        assert width >= 1 and height >= 1, f'Invalid size: ({width}, {height})'

        self.width = width
        self.height = height
        self.num_cells = width * height

        self.snake = None
        self.food = None
        self.rng = None
        self.viewer = None

        self.seed()
        self.reset()

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.width, self.height, 3))

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.snake = _Snake(x=self.rng.randint(0, self.width),
                            y=self.rng.randint(0, self.height))
        self._generate_food()
        return self._observation()

    def step(self, action):
        # Error handling
        if self._game_over():
            raise ResetNeeded()

        if action not in [LEFT, UP, RIGHT, DOWN]:
            raise InvalidAction(action)

        # Compute new position
        dir = action
        if _opposite_dir(action) == self.snake.dir:
            dir = self.snake.dir

        v = {LEFT: (-1, 0), UP: (0, 1), RIGHT: (1, 0), DOWN: (0, -1)}
        dx, dy = v[dir]
        x, y = self.snake.pos
        x += dx
        y += dy

        # Check for collisions
        if x < 0 or x >= self.width or y < 0 or y >= self.height or (x, y) in self.snake.body:
            return self._observation(), DEATH_REWARD, True, None

        # Update snake
        self.snake.pos = (x, y)
        self.snake.body.appendleft((x, y))
        self.snake.dir = dir

        # Check food
        if (x, y) == self.food:
            if self._game_over():
                return self._observation(), WIN_REWARD, True, None
            self._generate_food()
            return self._observation(), FOOD_REWARD, False, None

        self.snake.body.pop()
        return self._observation(), STEP_REWARD, False, None

    def _observation(self):
        """Generate the observation for the current state.

        We represent each observation as an array of size width * height * 3
        as follows.
            1st layer: multi-hot encoding of the snake's body
            2nd layer: one-hot encoding of the snake head's position
            3rd layer: one-hot encoding of the food position

        """
        # snake body
        snake_layer = np.zeros((self.width, self.height))
        for x, y in self.snake.body:
            snake_layer[x, y] = 1

        # snake head position
        head_layer = np.zeros((self.width, self.height))
        snake_x, snake_y = self.snake.pos
        head_layer[snake_x, snake_y] = 1

        # food position
        food_layer = np.zeros((self.width, self.height))
        food_x, food_y = self.food
        food_layer[food_x, food_y] = 1

        return np.stack((snake_layer, head_layer, food_layer), axis=2).astype(np.uint8)

    def _game_over(self):
        """Check if the player won."""
        return len(self.snake.body) >= self.num_cells

    def _generate_food(self):
        """Set the food location to a random empty cell.

        Raises:
            ResetNeeded: When there is no more space.

        """
        all_cells = {(x, y) for x in range(self.width) for y in range(self.height)}
        full_cells = set(self.snake.body)
        empty_cells = list(all_cells - full_cells)

        if len(empty_cells) == 0:
            raise ResetNeeded('No space to generate food.')

        random_index = self.rng.randint(len(empty_cells))
        self.food = empty_cells[random_index]

    def render(self, mode='human', close=False):
        SCALE = 20
        width = SCALE * self.width
        height = SCALE * self.height

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        # draw snake
        for x, y in self.snake.body:
            self._draw_square(x, y, scale=20, color=(0, 0, 0))

        # draw food
        food_x, food_y = self.food
        self._draw_square(food_x, food_y, scale=20, color=(0, 1, 0))

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def _draw_square(self, x, y, scale, color=(0, 0, 0)):
        l, r, t, b = scale * x, scale * (x + 1), scale * y, scale * (y + 1)
        square = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        square.set_color(*color)
        self.viewer.add_onetime(square)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
