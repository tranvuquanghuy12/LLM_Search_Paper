import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os

# Định nghĩa lại kiến trúc lớp GAT để load Model
class SmartGAT(torch.nn.Module):
    def __init__(self, channels):
        super(SmartGAT, self).__init__()
        self.gat = GATConv(channels, channels, heads=4, concat=False)
        self.alpha = 0.2 
    def forward(self, x, edge_index):
        # Trộn 80% gốc, 20% GAT
        return (1 - self.alpha) * x + self.alpha * self.gat(x, edge_index)

def load_data(embeddings_path, metadata_path):
    embeddings, paper_ids = [], []
    with open(embeddings_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_ids.append(data['paper_id'])
            embeddings.append(data['embedding'])
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        metadata = {item['paper_id']: item for item in raw} if isinstance(raw, list) else raw
    return np.array(embeddings), paper_ids, metadata

def search(query, embeddings, paper_ids, metadata, spec_model, tokenizer, gat_model, device, top_k=5):
    # 1. Mã hóa câu hỏi (Query) bằng SPECTER gốc
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        query_v = spec_model(**inputs).last_hidden_state[:, 0, :]
        
        # 2. CHÚ Ý: ĐỒNG BỘ HÓA GAT CHO CÂU HỎI
        # Để tìm kiếm chính xác, câu hỏi cũng phải được xử lý qua bộ lọc GAT mà anh Huy đã train
        # Vì đây là query đơn lẻ (không có đồ thị hàng xóm), ta áp dụng identity padding
        if gat_model:
            # edge_index rỗng vì chỉ có 1 node query
            edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0).to(device)
            # Chạy qua lớp GAT (Refinement: 80% nguyên bản + 20% bộ lọc GAT)
            query_v = gat_model(query_v, edge_index)
            print("--- [GAT Mode:] Đã áp dụng bộ lọc GAT cho câu hỏi ---")

    query_v = query_v.cpu().numpy()

    # 3. Tính toán độ tương đồng
    scores = cosine_similarity(query_v, embeddings)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    print("\n" + "="*50)
    print(f"KẾT QUẢ TÌM KIẾM ĐỒNG BỘ GAT CHO: '{query}'")
    print("="*50)
    
    for rank, idx in enumerate(top_indices, 1):
        p_id = paper_ids[idx]
        title = metadata.get(p_id, {}).get('title', 'Unknown Title')
        print(f"{rank}. [{scores[idx]:.4f}] - {title}")
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True, help='Vector file .jsonl (nên dùng file sau khi GAT)')
    parser.add_argument('--metadata', required=True, help='Dataset gốc .json')
    parser.add_argument('--query', required=True)
    parser.add_argument('--model_path', default='gat_model.pt', help='Đường dẫn file .pt đã train')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dữ liệu và trọng số
    e, ids, m = load_data(args.embeddings, args.metadata)
    
    # Init SPECTER
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    spec_model = AutoModel.from_pretrained('allenai/specter').to(device)
    spec_model.eval()

    # Init GAT (Load trọng số anh Huy đã train)
    gat_model = None
    if os.path.exists(args.model_path):
        gat_model = SmartGAT(768).to(device)
        gat_model.load_state_dict(torch.load(args.model_path, map_location=device))
        gat_model.eval()
    
    search(args.query, e, ids, m, spec_model, tokenizer, gat_model, device)
