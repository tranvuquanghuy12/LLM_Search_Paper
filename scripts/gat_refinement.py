import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import argparse
import os
from sklearn.neighbors import NearestNeighbors

class SmartGAT(torch.nn.Module):
    def __init__(self, channels):
        super(SmartGAT, self).__init__()
        # 1 lớp GAT để tránh Oversmoothing cho dataset nhỏ
        self.gat = GATConv(channels, channels, heads=4, concat=False)
        # Alpha: 20% thông tin từ đồ thị, 80% từ nội dung gốc SPECTER
        self.alpha = 0.2 

    def forward(self, x, edge_index):
        identity = x
        refined = self.gat(x, edge_index)
        # Kết hợp Residual Connection
        return (1 - self.alpha) * identity + self.alpha * refined

def build_graph(embeddings_path, metadata_path):
    print("--- 1. Đang xây dựng Đồ thị (KNN Graph) ---")
    paper_ids, embeds = [], []
    with open(embeddings_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            paper_ids.append(d['paper_id'])
            embeds.append(d['embedding'])
    
    x = torch.tensor(embeds, dtype=torch.float)
    
    # Sử dụng KNN để tự động nối các bài báo có nội dung gần nhau thành đồ thị
    # K=5: Mỗi bài báo nối với 4 bài hàng xóm gần nhất
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(embeds)
    distances, indices = nbrs.kneighbors(embeds)
    
    edges = []
    for i in range(len(paper_ids)):
        for neighbor_idx in indices[i][1:]: # Bỏ chính nó (index 0)
            edges.append([i, neighbor_idx])
            edges.append([neighbor_idx, i]) # Đồ thị vô hướng (mạnh hơn cho Message Passing)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Đồ thị: {len(paper_ids)} Nodes, {edge_index.shape[1]} Edges.")
    return Data(x=x, edge_index=edge_index), paper_ids

def train_refiner(data, epochs=50):
    print("--- 2. Đang huấn luyện GAT Refiner (Train trọng số của anh Huy) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmartGAT(channels=data.x.shape[1]).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Loss: Làm mịn các bài báo có liên kết (KNN) sao cho chúng gần nhau hơn
        loss = F.mse_loss(out[data.edge_index[0]], out[data.edge_index[1]])
        loss.backward()
        optimizer.step()
    return model, data

def main():
    parser = argparse.ArgumentParser(description="GNN Refinement Module using GAT")
    parser.add_argument('--embeddings', required=True, help='Path to raw embeddings .jsonl')
    parser.add_argument('--metadata', required=True, help='Path to metadata .json')
    parser.add_argument('--output', required=True, help='Path for refined embeddings .jsonl')
    parser.add_argument('--model_save', default='gat_model.pt', help='Where to save GAT weights')
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()

    # 1. Dựng đồ thị thông minh
    graph_data, paper_ids = build_graph(args.embeddings, args.metadata)
    
    # 2. Train trọng số GAT
    model, data = train_refiner(graph_data, args.epochs)
    
    # 3. Lưu Model (để dùng cho Search Demo sau này)
    print(f"Lưu trọng số GAT vào: {args.model_save}")
    torch.save(model.state_dict(), args.model_save)
    
    # 4. Xuất kết quả
    model.eval()
    with torch.no_grad():
        refined_embeds = model(data.x, data.edge_index).cpu().numpy()

    with open(args.output, 'w') as f:
        for i, pid in enumerate(paper_ids):
            res = {"paper_id": pid, "embedding": refined_embeds[i].tolist()}
            f.write(json.dumps(res) + '\n')
    print("--- 3. HOÀN THÀNH NÂNG CẤP GAT ---")

if __name__ == "__main__":
    main()
