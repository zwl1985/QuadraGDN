from torch import nn 
import torch
import torch.nn.functional as F
import random
from Models import Qnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, num_features, gamma, epsilon, lr, device, target_update=50, n_steps=1,
                 encoder_param_path='node_encoder.pth', ntype='DQN', training=True):
        self.num_features = num_features
        # Q Network
        self.q_net = Qnet(num_features, 1).to(device)
        # self.q_net.apply(self.init_weights)
        # Load pre trained parameters
        # self.q_net.encoder.load_state_dict(torch.load(encoder_param_path))

        # Target Q Network
        self.target_q_net = Qnet(num_features, 1).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 0
        self.target_update = target_update
        self.n_steps = n_steps
        self.ntype = ntype
        self.training = training

    @torch.no_grad()
    def take_action(self, state, env):
        selectable_nodes = list(set(env.Hyergraph.nodes) - set(env.seeds))
        if random.random() < self.epsilon:
            node = random.choice(selectable_nodes)
        else:
            selectable_nodes_t = torch.tensor(selectable_nodes, dtype=torch.long, device=self.device)
            states = torch.tensor(state, dtype=torch.float, device=self.device)
            # data = get_q_net_input([env.graph], self.num_features, self.device)
            self.q_net.eval()
            # q_values = self.q_net(data.x, data.edge_index, data.edge_weight, data.batch, states)
            q_values = self.q_net(torch.ones(env.Hyergraph.H.shape[0], 128, device=device), env.Hyergraph.H, states).squeeze(0)
            q_values_selectable = q_values[selectable_nodes]
            max_q_value, _ = q_values_selectable.max(0)
            # max_indices = (q_values_selectable == max_q_value).nonzero(as_tuple=False).squeeze()
            max_indices = torch.where(q_values_selectable == max_q_value)[0]

            if max_indices.numel() == 1:
                max_index = max_indices.item()
            else:
                random_idx = torch.randint(0, max_indices.numel(), (1,)).item()
                max_index = max_indices[random_idx].item()
            node = selectable_nodes[max_index]
        return node
    

    def update(self, replay_buffer, learning_time=8, batch_size=8):  
        currentList = torch.Tensor([]).to(self.device)
        currentList = torch.unsqueeze(currentList, 1)
        targetList = torch.Tensor([]).to(self.device)
        targetList = torch.unsqueeze(targetList, 1)

        self.q_net.train()
        samples = list(zip(*replay_buffer.sample(batch_size=batch_size)))

        for state, action, reward, next_state, done, graph in samples:
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        
            # 使用当前网络选择动作
            with torch.no_grad():
                current_q_next = self.q_net(
                    torch.ones(graph.H.shape[0], 128, device=self.device),
                    graph.H,
                    next_state
                )
                best_action = current_q_next.argmax()

            # 使用目标网络评估该动作的价值
            with torch.no_grad():
                target_q_next = self.target_q_net(
                    torch.ones(graph.H.shape[0], 128, device=self.device),
                    graph.H,
                    next_state
                )
                next_q_value = target_q_next[best_action]

            # 计算目标值
            if done:
                target = reward
            else:
                target = reward + self.gamma * next_q_value

            # 计算当前Q值
            current_q = self.q_net(
                torch.ones(graph.H.shape[0], 128, device=self.device),
                graph.H,
                state
            )[action]
            
            # 收集数据
            currentList = torch.cat((currentList, current_q.view(1, 1)), 0)
            targetList = torch.cat((targetList, target.view(1, 1)), 0)

        # 计算损失并更新
        self.optim.zero_grad()
        loss = self.loss(currentList, targetList)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optim.step()

        # 更新目标网络
        self.count += 1
        if self.count % self.target_update == 0:
            print("Updating target network...")
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # print(f"Current Q: min={currentList.min().item():.2f}, max={currentList.max().item():.2f}")
        # print(f"Target Q: min={targetList.min().item():.2f}, max={targetList.max().item():.2f}")
        
        return loss.item()  
       

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
