import random
from tqdm import tqdm
import numpy as np

def explore(env, agent, eps, replay_buffer, num_episodes, train=True, show_bar=True):
    """
    Explore using a given agent in a given environment.

    Parameters:
        Env (object): Environment object used to interact with agents.
        Agent (object): Agent object, including policy network, epsilon value, etc.
        EPS (float): The epsilon value of the current round, used to balance exploration and utilization.
        Replay_buffer (object): An experience replay buffer used to store data generated during the exploration process.
        Num_ episodes (int): The number of sequences to explore.
        Train (boolean, optional): Whether in training mode, defaults to True. In training mode, data will be added to the playback buffer.
        Show_mar (boolean, optional): Whether to display a progress bar, default to True.

    return:
        If train is False, return an episode return of float type, which is the accumulated reward during the testing process;
        Otherwise, no value will be returned (None).
    """

    agent.epsilon = eps

    if train:
        if show_bar:
            bar = tqdm(total=num_episodes, desc=f'epsilon={eps}时探索{num_episodes}条序列')  # 初始化进度条

        for _ in range(num_episodes):
            state = env.reset()  # Reset the environment to obtain the initial state
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env)  # Select action based on current state and epsilon value
                reward, next_state, done = env.step(action)  # Execute actions, receive rewards, and move on to the next state
                episode_return += reward
                state = next_state

            # After the exploration sequence is completed, add the n-step reward to the experience replay buffer
            env.n_step_add_buffer(replay_buffer)

            if show_bar:
                bar.update(1)

        if show_bar:
            bar.close()

    else:
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env)
                reward, next_state, done = env.step(action)
                episode_return += reward
                state = next_state

        return episode_return

def compute(H, seed_nodes, node_threshold_ratio=0.5, edge_threshold_ratio=0.5):
    """
    超图线性阈值传播模型（基于关联矩阵）
    
    参数:
    H: 超图关联矩阵 (nodes x edges)，H[i,j]=1表示节点i在超边j中
    seed_nodes: 种子节点索引列表
    node_threshold_ratio: 节点阈值比例 (0-1)，阈值 = floor(比例 * 节点度)
    edge_threshold_ratio: 超边阈值比例 (0-1)，阈值 = floor(比例 * 超边基数)
    
    返回:
    activated_nodes: 最终激活的节点索引列表
    activated_edges: 最终激活的超边索引列表
    propagation_history: 传播过程历史记录
    """
    
    n_nodes, n_edges = H.shape
    
    # 初始化节点和超边状态 (0=未激活, 1=已激活)
    node_states = np.zeros(n_nodes, dtype=int)
    edge_states = np.zeros(n_edges, dtype=int)
    
    # 计算节点度（每个节点参与的超边数）
    node_degrees = H.sum(axis=1)
    
    # 计算超边基数（每个超边包含的节点数）
    edge_cardinalities = H.sum(axis=0)
    
    
    node_thresholds = np.floor(node_threshold_ratio * node_degrees).astype(int)
    edge_thresholds = np.floor(edge_threshold_ratio * edge_cardinalities).astype(int)
    
    
    node_thresholds = np.maximum(1, node_thresholds)
    edge_thresholds = np.maximum(1, edge_thresholds)
    
    
    node_states[seed_nodes] = 1
    
    
    propagation_history = []
    activated_nodes_history = [set(seed_nodes)]
    activated_edges_history = [set()]
    
    t = 0
    changed = True
    
    while changed:
        changed = False
        t += 1
        
        
        step_info = {
            'step': t,
            'new_activated_nodes': [],
            'new_activated_edges': []
        }
        
        
        new_activated_edges = []
        for e in range(n_edges):
            if edge_states[e] == 0:  
                
                activated_nodes_in_edge = np.sum(node_states * H[:, e])
                
                
                if activated_nodes_in_edge >= edge_thresholds[e]:
                    edge_states[e] = 1
                    new_activated_edges.append(e)
                    changed = True
        
        
        new_activated_nodes = []
        for v in range(n_nodes):
            if node_states[v] == 0: 
                
                activated_edges_of_node = np.sum(edge_states * H[v, :])
                
                
                if activated_edges_of_node >= node_thresholds[v]:
                    node_states[v] = 1
                    new_activated_nodes.append(v)
                    changed = True
        
        
        step_info['new_activated_nodes'] = new_activated_nodes
        step_info['new_activated_edges'] = new_activated_edges
        propagation_history.append(step_info)
        
        
        activated_nodes_history.append(activated_nodes_history[-1].union(new_activated_nodes))
        activated_edges_history.append(activated_edges_history[-1].union(new_activated_edges))
        
        
        if not new_activated_nodes and not new_activated_edges:
            break
    
    
    activated_nodes = list(np.where(node_states == 1)[0])
    activated_edges = list(np.where(edge_states == 1)[0])
    
    return len(activated_nodes), len(activated_edges) , propagation_history

class HyperGraphEnvironment:
    def __init__(self, Hyergraphs, k, is_train=True, gamma=0.99, n_steps=1, method='MC', R=1, num_workers=8):
        """
        G: The graph of networkx, Graph or DiGraph;
        k: Seed set size;
        n_steps: the step size used to calculate rewards;
        method: Method for calculating rewards;
        R: Use Monte Carlo to estimate the number of reward rounds;
        numw_workers: How many cores are used to calculate propagation range
        """
        self.Hyergraphs = Hyergraphs  
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.method = method
        self.R = R
        self.num_workers = num_workers
        # Current status, each position represents whether a node has been selected, with 1 indicating selected and 0 indicating unselected
        self.state = None
        # Reward for the previous state
        self.preview_reward = 0
        # Record the status, actions, rewards, and next steps of each exploration to calculate the reward for n steps
        self.states = []
        self.actions = []
        self.rewards = []
        # self.next_states = []

        self.seeds = []
        self.state_records = {}

        self.is_train = is_train

    def reset(self):
        """
        Reset the environment.
        """

        self.Hyergraph = random.choice(self.Hyergraphs)
        self.seeds = []
        self.state = [0] * self.Hyergraph.node_num
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        
        return self.state

    def step(self, action):
        """
        Transfer to a new state based on the given action.
        """
        self.states.append(self.state.copy())
        self.state[action] = 1
        self.seeds.append(action)
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        if done:
            self.states.append(self.state.copy())

        self.actions.append(action)
        self.rewards.append(reward)
        # self.next_states.append(self.state)
        return reward, self.state, done

    def compute_reward(self):
        str_seeds = str(id(self.Hyergraph)) + str(sorted(self.seeds))
        if self.method == 'MC':
            if str_seeds in self.state_records:
                current_reward = self.state_records[str_seeds]
            else:
                current_reward,_,_ = compute(self.Hyergraph.H.cpu().numpy(), self.seeds)
            r = current_reward - self.preview_reward
            self.preview_reward = current_reward
            self.state_records[str_seeds] = current_reward
            return r
        else:
            pass

    def n_step_add_buffer(self, buffer):
        states = self.states
        rewards = self.rewards
        n = self.n_steps
        gamma = self.gamma
        
        # Directly limit the cycle range and avoid processing the situation of insufficient n steps
        for i in range(len(states) - n):
            # Determines whether it is terminated
            done = (i + n) == (len(states) - 1)
            next_state = states[i + n]
            
            # Calculate the n-step reward
            n_reward = sum(rewards[i + j] * (gamma ** j) for j in range(n))
            
            buffer.add(states[i], self.actions[i], n_reward, next_state, done, self.Hyergraph)
