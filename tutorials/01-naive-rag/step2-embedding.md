# Step 2: 임베딩 (Embedding)

## 핵심 개념

컴퓨터는 텍스트를 이해 못한다. 숫자만 안다.
그래서 텍스트를 **숫자 배열(벡터)**로 변환하는 게 임베딩이다.

```
"오늘 날씨가 좋다"  →  [0.12, -0.34, 0.78, ..., 0.56]  (1536개 숫자)
"날씨가 화창하다"   →  [0.11, -0.31, 0.80, ..., 0.54]  (비슷한 숫자!)
"주가가 폭락했다"   →  [-0.45, 0.67, -0.12, ..., 0.89] (다른 숫자!)
```

- 의미가 비슷한 문장 → 비슷한 숫자 배열
- 의미가 다른 문장 → 다른 숫자 배열

두 벡터가 얼마나 비슷한지 측정: **코사인 유사도** (1에 가까우면 비슷, 0에 가까우면 다름)

## 실습 코드

### 셀 4: OpenAI 임베딩 해보기

```python
from openai import OpenAI

client = OpenAI()  # .env의 API키 자동 로드

texts = [
    "오늘 날씨가 좋다",
    "날씨가 화창하다",
    "주가가 폭락했다"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

for i, text in enumerate(texts):
    vec = response.data[i].embedding
    print(f"'{text}' → 벡터 길이: {len(vec)}, 앞 5개: {vec[:5]}")
```

### 셀 5: 코사인 유사도 직접 계산

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = response.data[0].embedding  # "오늘 날씨가 좋다"
v2 = response.data[1].embedding  # "날씨가 화창하다"
v3 = response.data[2].embedding  # "주가가 폭락했다"

print(f"날씨 좋다 vs 화창하다: {cosine_similarity(v1, v2):.4f}")
print(f"날씨 좋다 vs 주가 폭락: {cosine_similarity(v1, v3):.4f}")
print(f"화창하다  vs 주가 폭락: {cosine_similarity(v2, v3):.4f}")
```

## 핵심 포인트

- "날씨 좋다" vs "화창하다" → 유사도 **높음**
- "날씨" vs "주가 폭락" → 유사도 **낮음**
- 이게 작동하면 → **질문과 비슷한 의미의 청크를 찾을 수 있다!**

이게 RAG에서 "검색"이 작동하는 원리다.
질문을 임베딩 → 저장된 청크 임베딩들과 유사도 비교 → 가장 비슷한 것을 골라냄.
