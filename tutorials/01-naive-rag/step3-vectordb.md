# Step 3: 벡터DB (Vector Database)

## 왜 벡터DB가 필요한가?

임베딩으로 유사도를 계산할 수 있지만, 청크가 많아지면 전부 비교하는 건 비효율적이다.

```
청크 1000개 → 질문과 1000번 비교 → 느림
```

벡터DB는 인덱스를 만들어서 **전부 비교하지 않고도** 가장 유사한 벡터를 빠르게 찾아준다.

ChromaDB를 사용한다. 설치 간단, 파일 기반, 서버 불필요.

## 실습 코드

### 셀 6: ChromaDB 설치 & 임포트

```python
# 터미널에서 먼저 설치: pip install chromadb
import chromadb

# 디스크에 저장 (다른 노트북에서도 불러올 수 있도록)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="news")
print("컬렉션 생성 완료!")
```

### 셀 7: 기사 청크들을 벡터DB에 저장

```python
from openai import OpenAI
import pandas as pd

openai_client = OpenAI()
df = pd.read_excel("Articles_20260305_125404.xlsx")

def simple_chunk(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
all_ids = []
all_metadatas = []

for idx, row in df.iterrows():
    if not isinstance(row['content'], str):
        continue
    chunks = simple_chunk(row['content'])
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"article_{idx}_chunk_{i}")
        all_metadatas.append({"title": row['title'], "category": row['category']})

print(f"총 청크 수: {len(all_chunks)}")
```

### 셀 8: 임베딩 후 ChromaDB에 저장

```python
batch_size = 100

for i in range(0, len(all_chunks), batch_size):
    batch_chunks = all_chunks[i:i+batch_size]
    batch_ids = all_ids[i:i+batch_size]
    batch_meta = all_metadatas[i:i+batch_size]

    response = openai_client.embeddings.create(
        input=batch_chunks,
        model="text-embedding-3-small"
    )
    embeddings = [d.embedding for d in response.data]

    collection.add(
        ids=batch_ids,
        embeddings=embeddings,
        documents=batch_chunks,
        metadatas=batch_meta
    )
    print(f"{i + len(batch_chunks)}/{len(all_chunks)} 저장 완료")

print("벡터DB 저장 완료!")
```

### 셀 9: 검색 테스트

```python
question = "중동 전쟁 상황은?"
q_response = openai_client.embeddings.create(
    input=[question],
    model="text-embedding-3-small"
)
q_embedding = q_response.data[0].embedding

results = collection.query(
    query_embeddings=[q_embedding],
    n_results=3
)

print(f"질문: {question}\n")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"--- 결과 {i+1} [{meta['category']}] {meta['title'][:30]} ---")
    print(doc[:150])
    print()
```

## 관찰할 것

1. 질문과 **관련 있는** 기사 청크가 나오나?
2. 질문을 바꿔보세요: `"경제 상황"`, `"날씨 예보"`, `"IT 기술"` 등
3. **관련 없는 결과가 나오는 경우**가 있나? → Naive RAG의 한계
