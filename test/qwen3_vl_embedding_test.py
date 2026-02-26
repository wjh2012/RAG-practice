import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-VL-Embedding-2B", "scripts"))

import numpy as np
from qwen3_vl_embedding import Qwen3VLEmbedder

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-VL-Embedding-2B")
    model = Qwen3VLEmbedder(model_name_or_path=model_path)

    # 이미지 2장 임베딩
    embeddings = model.process([
        {
            "image": r"C:\Users\wjh\Desktop\test1.jpg",
            "instruction": "Classify the type of this document image.",
        },
        {
            "image": r"C:\Users\wjh\Desktop\test2.jpg",
            "instruction": "Classify the type of this document image.",
        },
    ])

    # 유사도 확인
    a, b = embeddings[0].detach().cpu().numpy(), embeddings[1].detach().cpu().numpy()
    cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"유사도: {cosine_sim:.4f}")
