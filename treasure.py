import gym
import numpy as np
import matplotlib.pyplot as pp

class Map(object):
	''' This class is for the 'Treasure' environment. Given a grid with positive reward values specified, 
		the agent can move around the grid normally. Each timestep gets a reward of negative one. 
		Finding a "treasure" ends the episode with the appropriate positive reward.
	'''
	def __init__(self, grid, random=0):
		self.grid = np.array(grid)
		self.position = [3,0]
		self.actions = ['left', 'right', 'up', 'down']
		self.bounds = {'x': (0, self.grid.shape[1]-1),
						'y': (0, self.grid.shape[0]-1)}
		# Find all the "treasures" according the grid definition.
		self.goals = {}
		for i in xrange(self.grid.shape[0]):
			for j in xrange(self.grid.shape[1]):
				if self.grid[i,j] > 0:
					self.goals[(i,j)] = self.grid[(i,j)]
		# In case we want to add stochasticity to the environment.
		self.random = random
		self.reset()

	def step(self, action):
		reward = -1
		ended = False
		# Check if the environment should be stochastic.
		if np.random.uniform() < self.random:
			action = np.random.choice(self.actions)
		pos = tuple(self.position)
		# Check if we have reached a goal and end the episode if so.
		if pos in self.goals:
			self.reset()
			reward = self.goals[pos]
			ended = True
		elif action == 'left':
			if self.position[1] > self.bounds['x'][0]:
				self.position[1] -= 1
		elif action == 'right':
			if self.position[1] < self.bounds['x'][1]:
				self.position[1] += 1
		elif action == 'up':
			if self.position[0] > self.bounds['y'][0]:
				self.position[0] -= 1
		elif action == 'down':
			if self.position[0] < self.bounds['y'][1]:
				self.position[0] += 1

		return tuple(self.position), reward, ended

	def reset(self):
		self.position = [3,0]

	def render(self, mode='human', close=False):
		print
		for i in xrange(self.grid.shape[0]):
			print
			for j in xrange(self.grid.shape[1]):
				if self.position == [i,j]:
					print 'o',
				elif (i,j) in self.goals:
					print 'G',
				else:
					print '-',
# Initialize a Q function as a two level dictionary Q[state][action] = 0.
def init_Q(env):
	Q = {}
	for i in xrange(env.grid.shape[0]):
		for j in xrange(env.grid.shape[1]):
			Q[(i,j)] = {'left':0, 'right':0, 'up':0, 'down':0}
	return Q

def sarsa(env, n_episodes, eps=0.1, alpha=0.1, gamma=1):
	def get_action(Q, p):
		if np.random.uniform() < eps:
			action = np.random.choice(env.actions)
		else:
			action = None
			val = -np.inf
			for a, v in Q[p].iteritems():
				if v > val: action, val = a, v
		return action

	# Discretize the state space.
	Q = init_Q(env)
	env.reset()
	position_t = tuple(env.position)
	# Keep track of the returns for each episode.
	returns = []
	for ix in xrange(n_episodes):
		render = False
		if ix == n_episodes - 1: render = False
		ended = False
		ret = 0
		# Get the first action.
		action_t = get_action(Q, position_t)
		while not ended:
			if render: env.render()
			# Execute the current action and get a reward.
			position_tp1, reward, ended = env.step(action_t)
			# Choose the next action but do not execute yet.
			action_tp1 = get_action(Q, position_tp1)
			# Take the action.
			ret += reward
			# Calculate the update.
			if not ended:
				Q[position_t][action_t] = Q[position_t][action_t] + alpha*(reward + gamma*Q[position_tp1][action_tp1] - Q[position_t][action_t])
			else:
				Q[position_t][action_t] = Q[position_t][action_t] + alpha*(reward - Q[position_t][action_t])
			position_t = position_tp1
			action_t = action_tp1
		returns.append(ret)
	return returns, Q

def exp_sarsa(env, n_episodes, eps=0.1, alpha=0.1, gamma=1):
	Q = init_Q(env)
	env.reset()
	position = tuple(env.position)
	returns = []
	for ix in xrange(n_episodes):
		render = False
		if ix == n_episodes - 1: render = False
		ended = False
		ret = 0
		while not ended:
			if render: env.render()
			# Get action using eps-soft policy.
			if np.random.uniform() < eps:
				action = np.random.choice(env.actions)
			else:
				action = None
				val = -np.inf
				for a, v in Q[position].iteritems():
					if v > val: action, val = a, v
			# Take the action.
			new_position, reward, ended = env.step(action)
			ret += reward

			# Calculate the update. First get the expected value.
			V = 0
			max_a, max_q = None, -np.inf
			for a, v in Q[new_position].iteritems():
				if v > max_q: max_a, max_q = a, v
			for a, v in Q[new_position].iteritems():
				if a == max_a:
					V += ((1-eps) + eps/4.)*v
				else:
					V += eps/4.*v
			if ended: V = 0
			# Perform the update.
			Q[position][action] =  Q[position][action] + alpha*(reward + gamma*V - Q[position][action])
			position = new_position
		returns.append(ret)
	return returns, Q

if __name__ == '__main__':
	# 0's are neutral cells, -1's cause you to return to the start, 1 is the goal.
	grid = [[0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
			[0,0,0,0,5,0,0,0,0,50,0,0,0,0,500],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0]]

	cliff = Map(grid, random=0.)
	
	repeats = 100
	length = 10000
	rs_exp_sarsa = np.zeros((repeats,length))
	rs_sarsa = np.zeros((repeats,length))

	best_exp, a_exp = -np.inf, -1
	best_s, a_s = -np.inf, -1
	# Uncomment to do grid search across alpha.
	for a in [1.0]:#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
		print a, 
		for i in xrange(0, repeats):
			rs_exp_sarsa[i,:], Q_exp = exp_sarsa(cliff, length, eps=0.5, alpha=1.0, gamma=1)
			rs_sarsa[i,:], Q_s = sarsa(cliff, length, eps=0.5, alpha=0.1, gamma=1)

		exp = np.mean(rs_exp_sarsa[:,-1])
		s = np.mean(rs_sarsa[:,-1])

		if exp > best_exp: best_exp, a_exp = exp, a
		if s > best_s: best_s, a_s = s, a

		print a, exp, s
		for i in xrange(cliff.grid.shape[0]):
			print
			for j in xrange(cliff.grid.shape[1]):
				print '%.2f' % (sum(Q_s[(i,j)].values())/4.),

		means_e = np.mean(rs_exp_sarsa, axis=0)
		means_s = np.mean(rs_sarsa, axis=0)
		pp.xlabel('Number of Trials')
		pp.ylabel('Average Returns')
		pp.plot(np.arange(0, len(means_e)), means_e, 'k-')
		pp.plot(np.arange(0, len(means_s)), means_s, 'b-')
		pp.show()
	print best_exp, a_exp
	print best_s, a_s

	