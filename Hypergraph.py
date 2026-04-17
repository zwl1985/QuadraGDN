import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

# 定义节点和边的结构
class NodeStruct:
    def __init__(self):
        self.edge_list = []  # 与节点相关的超边
        self.state = 0  # 节点的状态
        self.degree = 0  # 节点的度
        self.mark = 0  # 标记节点是否加入过队列
        self.id = 0  # 超图修正


# 定义超边的数据结构
class EdgeStruct:
    def __init__(self):
        self.node_list = []
        self.state = 0
        self.cardinality = 0

def get_hypergraph_incidence_matrix(file_path):
    """
    构建超图关联矩阵 - 适配每行第一个元素是超边ID的数据集格式
    
    参数:
        file_path: 数据集文件路径
        
    返回:
        H: 超图关联矩阵 (节点×超边)
        node_count: 节点总数
        hyperedge_count: 超边总数
        node_to_index: 节点ID到矩阵索引的映射
        edge_to_index: 超边ID到矩阵索引的映射
    """
    hyperedges = {}  # 使用字典存储超边：{超边ID: [节点列表]}
    all_nodes = set()  # 存储所有节点ID
    
    with open(file_path, 'r') as file:
        for line in file:
            # 跳过空行
            if not line.strip():
                continue
                
            # 分割行数据
            data = line.split()
            
            # 至少需要超边ID和一个节点
            if len(data) < 2:
                continue
                
            try:
                # 第一个元素是超边ID
                edge_id = data[0]
                
                # 剩余元素是节点列表
                nodes = list(map(int, data[1:]))
                
                # 存储超边信息
                hyperedges[edge_id] = nodes
                
                # 收集所有节点
                all_nodes.update(nodes)
                
            except ValueError:
                print(f"警告: 行 '{line.strip()}' 包含非数字字符，已跳过")
                continue
    
    # 节点映射
    sorted_nodes = sorted(all_nodes)
    node_count = len(sorted_nodes)
    node_to_index = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # 超边映射
    sorted_edge_ids = sorted(hyperedges.keys())
    hyperedge_count = len(sorted_edge_ids)
    edge_to_index = {edge_id: idx for idx, edge_id in enumerate(sorted_edge_ids)}
    
    # 创建关联矩阵
    H = np.zeros((node_count, hyperedge_count), dtype=int)
    
    # 填充关联矩阵
    for edge_id, nodes in hyperedges.items():
        col_idx = edge_to_index[edge_id]
        for node in nodes:
            row_idx = node_to_index[node]
            H[row_idx, col_idx] = 1
    
    return torch.tensor(H, device=device, dtype=torch.float), node_count, hyperedge_count

# 构造超图,传入的参数为文件名
def construct_hypergraph(filename, NodeSize, EdgeSize):
    Node_array = [NodeStruct() for _ in range(NodeSize)]
    Edge_array = [EdgeStruct() for _ in range(EdgeSize)]
    # 读取超图
    # 用于存储结果的列表
    result = []
    edgenum = 0
    nodenum = 0
    # 打开文件
    with open(filename, "r") as file:
        # 逐行读取文件内容
        for line in file:
            # 将每行按空格分割成字符串列表
            numbers_str = line.strip().split()
            # 将字符串列表转换为整数列表
            numbers = [int(num) for num in numbers_str]
            # 将整数列表添加到结果列表中
            result.append(numbers)
    # 构建超图
    for line_list in result:
        edgenum += 1
        Edge_array[line_list[0]].node_list.extend(line_list[1:])
        Edge_array[line_list[0]].cardinality = len(line_list) - 1
        for t in line_list[1:]:
            Node_array[t].edge_list.append(line_list[0])
            Node_array[t].degree += 1
    # 统计节点数
    for i in range(NodeSize):
        if Node_array[i].degree != 0:
            nodenum += 1
    return Node_array, nodenum, Edge_array, edgenum


class Hypergraph:
    def __init__(self, filename, y):
         self.filename = filename
         self.H, node_num, self.edge_num = get_hypergraph_incidence_matrix(self.filename)
         self.Node_array, _, self.Edge_array, _ = construct_hypergraph(self.filename, node_num, self.edge_num)
        #  y,_ = compute_individual_node_spread(self.H.cpu().numpy())
        #  y_normalized = (y - y.min()) / (y.max() - y.min())
         self.y = y
         self.node_num = node_num
         self.nodes = [i for i in range(node_num)]