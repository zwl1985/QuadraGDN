from HyperGraphEnvironment import HyperGraphEnvironment, explore
from Hypergraph import Hypergraph
import os
import torch
from Models import Qnet
from Agent import Agent





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
   
    base_dir = os.getcwd()
    test_hypergraph_dir = os.path.join(base_dir, 'TestHypergraph')
    fn = os.path.join(test_hypergraph_dir, 'Restaurant-rev.txt')

    K = 20
    test_env = HyperGraphEnvironment([Hypergraph(fn, None)], K, 0.99)

    q_net = Qnet(32, 1).to(device)
    state_dict = torch.load('q_net.pth', weights_only=False)
    q_net.load_state_dict(state_dict) 
    q_net.eval()
    agent = Agent(32, 0.99, 1.0, 0.001, device)
    agent.q_net = q_net
    rewards = explore(test_env, agent, 0, None, 1, False, False)
    print(f"Seed set:{test_env.seeds}, Influence spread:{rewards}")
