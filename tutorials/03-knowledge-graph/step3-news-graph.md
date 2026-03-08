# Step 3: 뉴스 데이터로 그래프 만들기

LLM으로 추출한 트리플을 Neo4j에 저장.

## 실습 코드

### 셀 1: LLM으로 트리플 추출 (구조화)

```python
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import json

load_dotenv()
client = OpenAI()
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

df = pd.read_excel("Articles_20260305_125404.xlsx")

def extract_triples(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """뉴스 기사에서 엔티티와 관계를 추출하세요.
JSON 배열로 출력. 각 항목: {"subject": "주어", "subject_type": "타입", "relation": "관계", "object": "목적어", "object_type": "타입"}
타입은: Person, Organization, Country, Event, Weapon, Place 중 하나.
최대 7개만. JSON만 출력하세요."""},
            {"role": "user", "content": text}
        ]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return []

triples = extract_triples(df.iloc[0]['content'])
for t in triples:
    print(f"({t['subject']}) -[{t['relation']}]-> ({t['object']})")
```

### 셀 2: 전체 기사에서 추출

```python
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

all_triples = []

for idx, row in df.iterrows():
    if not isinstance(row['content'], str):
        continue

    triples = extract_triples(row['content'])
    print(f"[{idx+1}/{len(df)}] {row['title'][:30]}... → {len(triples)}개 트리플")

    for t in triples:
        t['source_article'] = row['title']
        t['category'] = row['category']
        all_triples.append(t)

print(f"\n총 트리플 수: {len(all_triples)}")
```

### 셀 3: Neo4j에 저장

```python
with driver.session() as session:
    for t in all_triples:
        try:
            session.run("""
                MERGE (s {name: $subject})
                SET s:""" + t['subject_type'] + """
                MERGE (o {name: $object})
                SET o:""" + t['object_type'] + """
                MERGE (s)-[r:""" + t['relation'].replace(" ", "_") + """]->(o)
            """, subject=t['subject'], object=t['object'])
        except Exception as e:
            print(f"오류: {e}")
            continue

print("Neo4j 저장 완료!")
```

### 셀 4: 그래프 확인

```python
with driver.session() as session:
    nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
    rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
    print(f"노드: {nodes}개, 관계: {rels}개")

    result = session.run("""
        MATCH (n)-[r]-()
        RETURN n.name AS name, labels(n) AS type, count(r) AS connections
        ORDER BY connections DESC
        LIMIT 10
    """)
    print("\n=== 허브 엔티티 (연결 많은 순) ===")
    for record in result:
        print(f"  {record['name']} ({record['type'][0]}): {record['connections']}개 연결")
```

### 셀 5: 그래프 탐색

```python
with driver.session() as session:
    result = session.run("""
        MATCH (n)-[r]-(target {name: "이란"})
        RETURN n.name AS entity, type(r) AS relation
    """)
    print("=== 이란과 관련된 엔티티 ===")
    for record in result:
        print(f"  {record['entity']} - {record['relation']}")

    result = session.run("""
        MATCH (start {name: "트럼프"})-[r1]-(mid)-[r2]-(end)
        RETURN DISTINCT end.name AS entity, type(r1) AS rel1, mid.name AS via, type(r2) AS rel2
        LIMIT 10
    """)
    print("\n=== 트럼프에서 2홉 이내 ===")
    for record in result:
        print(f"  트럼프 -{record['rel1']}-> {record['via']} -{record['rel2']}-> {record['entity']}")
```

## 관찰할 것

1. **허브 엔티티**는 뭔가? (연결 많은 노드 = 핵심 인물/조직)
2. Neo4j 브라우저에서 시각화: `MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50`
3. 2홉 검색으로 간접 관계 발견 가능?

## 3단계 Knowledge Graph 기초 완료!
