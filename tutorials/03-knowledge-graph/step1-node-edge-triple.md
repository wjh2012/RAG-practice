# Step 1: 노드, 엣지, 트리플 개념

## 지식 그래프란?

정보를 **"누가 - 어떤 관계로 - 누구와 연결"** 형태로 저장.

```
(트럼프) --[대통령]--> (미국)
(이란)   --[전쟁중]--> (미국)
(쿠르드족) --[참전]--> (이란 전쟁)
```

## 3가지 핵심 개념

| 개념 | 의미 | 예시 |
|---|---|---|
| 노드(Node) | 개체 | 트럼프, 미국, 이란 |
| 엣지(Edge) | 관계 | "대통령이다", "전쟁중" |
| 트리플(Triple) | (주어, 관계, 목적어) | (트럼프, 대통령, 미국) |

## 왜 그래프인가?

```
질문: "이란과 전쟁 중인 나라의 대통령은?"
그래프: 이란 --[전쟁중]--> 미국 --[대통령]--> 트럼프
        2홉(hop)만에 답 도출!
```

**관계를 따라가며 추론** → 그래프의 핵심 장점

## 실습 코드

### 셀 1: 뉴스 기사에서 트리플 추출

```python
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = OpenAI()
df = pd.read_excel("Articles_20260305_125404.xlsx")

text = df.iloc[0]['content']

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": """뉴스 기사에서 (주어, 관계, 목적어) 형태의 트리플을 추출하세요.
각 줄에 하나씩, 형식: (주어, 관계, 목적어)
최대 10개만 추출하세요."""},
        {"role": "user", "content": text}
    ]
)

print("=== 추출된 트리플 ===")
print(response.choices[0].message.content)
```

### 셀 2: 여러 기사에서 트리플 추출

```python
all_triples = []

for idx in range(5):
    text = df.iloc[idx]['content']
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """뉴스 기사에서 (주어, 관계, 목적어) 트리플을 추출하세요.
각 줄에: (주어, 관계, 목적어)
최대 5개만."""},
            {"role": "user", "content": text}
        ]
    )

    triples_text = response.choices[0].message.content
    print(f"\n--- 기사 {idx}: {df.iloc[idx]['title'][:30]} ---")
    print(triples_text)

    for line in triples_text.strip().split("\n"):
        line = line.strip().strip("()")
        parts = [p.strip().strip("()") for p in line.split(",")]
        if len(parts) == 3:
            all_triples.append(tuple(parts))

print(f"\n총 트리플 수: {len(all_triples)}")
```

### 셀 3: 자주 등장하는 엔티티 확인

```python
from collections import Counter

entities = []
for s, r, o in all_triples:
    entities.append(s)
    entities.append(o)

counter = Counter(entities)
print("=== 자주 등장하는 엔티티 ===")
for entity, count in counter.most_common(10):
    print(f"  {entity}: {count}회")
```

## 관찰할 것

1. LLM이 추출한 트리플이 **정확한가?**
2. **기사 간 공통 엔티티**가 있나?
3. 이 관계를 따라가면 질문에 답할 수 있을까?
