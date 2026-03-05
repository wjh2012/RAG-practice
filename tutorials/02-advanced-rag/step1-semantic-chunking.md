# Step 1: Semantic Chunking

## 원리

Naive RAG: 200글자씩 기계적으로 자름
Semantic Chunking: **문장 간 의미 유사도**를 기준으로 자름

```
문장1: "이란이 미사일을 발사했다"
문장2: "쿠르드족이 지상전을 시작했다"     ← 유사도 높음 (같은 주제)
문장3: "코스피가 5500선을 회복했다"      ← 유사도 급락! → 여기서 자름
문장4: "밸류업 지수가 사상 최고치를 기록했다"
```

인접 문장의 임베딩 유사도를 계산 → **유사도가 급격히 떨어지는 지점**에서 자름

## 실습 코드

### 셀 1: 문장 분리

```python
import re
from openai import OpenAI
import numpy as np
import pandas as pd

client = OpenAI()
df = pd.read_excel("Articles_20260305_125404.xlsx")

text = df.iloc[0]['content']

sentences = [s.strip() for s in re.split(r'[.!?\n다]\s', text) if len(s.strip()) > 10]
print(f"문장 수: {len(sentences)}")
for i, s in enumerate(sentences[:5]):
    print(f"  {i}: {s[:60]}")
```

### 셀 2: 인접 문장 간 유사도 계산

```python
response = client.embeddings.create(
    input=sentences,
    model="text-embedding-3-small"
)
embeddings = [d.embedding for d in response.data]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = []
for i in range(len(embeddings) - 1):
    sim = cosine_similarity(embeddings[i], embeddings[i+1])
    similarities.append(sim)
    print(f"문장 {i} ↔ {i+1}: {sim:.4f}")
```

### 셀 3: 유사도 급락 지점에서 청킹

```python
threshold = np.mean(similarities) - np.std(similarities)
print(f"임계값: {threshold:.4f}\n")

chunks = []
current_chunk = [sentences[0]]

for i, sim in enumerate(similarities):
    if sim < threshold:
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i+1]]
        print(f"✂ 문장 {i}~{i+1} 사이에서 자름 (유사도: {sim:.4f})")
    else:
        current_chunk.append(sentences[i+1])

chunks.append(" ".join(current_chunk))

print(f"\n총 청크 수: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n--- 청크 {i} ({len(chunk)}자) ---")
    print(chunk[:120])
```

## 관찰할 것

1. Naive 청킹(200글자)과 비교: 문장이 중간에 잘리는 문제가 해결됐나?
2. 각 청크가 **하나의 주제**를 담고 있나?
3. threshold를 조절하면 청크 수가 어떻게 변하나?
