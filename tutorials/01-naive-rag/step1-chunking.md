# Step 1: 문서 로드 & 청킹 (Chunking)

## 왜 문서를 쪼개야 하나?

크롤링한 뉴스 기사 1개가 보통 500~2000자. 60개면 전체 텍스트가 엄청 크다.
통째로 LLM에 넣으면:

- 토큰 제한에 걸림 (비용도 폭증)
- 관련 없는 내용까지 섞여서 답변 품질 저하
- 어떤 부분이 관련 있는지 LLM이 찾기 어려움

→ **작은 조각(chunk)으로 쪼개고, 질문과 관련된 조각만 골라서** LLM에 넘긴다.

## 실습 코드

### 셀 1: 기사 로드

```python
import pandas as pd

df = pd.read_excel("Articles_20260305_125404.xlsx")
print(f"기사 수: {len(df)}")
print(f"컬럼: {list(df.columns)}")
```

### 셀 2: 기사 1개 살펴보기

```python
text = df.iloc[0]['content']
print(f"글자 수: {len(text)}")
print(f"내용 미리보기:\n{text[:300]}...")
```

### 셀 3: 직접 청킹해보기 (가장 단순한 방식)

```python
def simple_chunk(text, chunk_size=200, overlap=50):
    """텍스트를 chunk_size 글자씩 쪼개기. overlap만큼 겹치게."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = simple_chunk(text)
print(f"청크 수: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- 청크 {i} ({len(chunk)}자) ---")
    print(chunk[:100])
    print()
```

## 생각해볼 것

1. **chunk_size를 100으로 줄이면?** 500으로 늘리면? 청크 수가 어떻게 변하나?
2. **overlap은 왜 있나?** overlap=0으로 하면 문장이 중간에 잘리는 경우가 생기나?
3. **글자 수 기준으로 자르는 게 좋은 방법인가?** 문장 중간에 잘리는 문제는?
