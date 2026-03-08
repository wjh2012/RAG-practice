# Step 2: Neo4j 설치 & Cypher 쿼리 기초

## Neo4j란?

그래프 데이터베이스. 노드와 관계를 저장, **Cypher** 쿼리 언어로 검색.

## 설치: Docker

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

브라우저에서 `http://localhost:7474` 접속 (ID: neo4j / PW: password123)

> Docker 없으면: Neo4j Desktop 또는 Neo4j Sandbox (무료 웹)

## Cypher 기초

### 1. 노드 만들기

```cypher
CREATE (t:Person {name: "트럼프", role: "대통령"})
CREATE (u:Country {name: "미국"})
CREATE (i:Country {name: "이란"})
CREATE (k:Group {name: "쿠르드족"})
RETURN t, u, i, k
```

### 2. 관계 만들기

```cypher
MATCH (t:Person {name: "트럼프"}), (u:Country {name: "미국"})
CREATE (t)-[:대통령]->(u)

MATCH (u:Country {name: "미국"}), (i:Country {name: "이란"})
CREATE (u)-[:전쟁중]->(i)

MATCH (k:Group {name: "쿠르드족"}), (i:Country {name: "이란"})
CREATE (k)-[:참전_반대편]->(i)
```

### 3. 검색

```cypher
// 모든 노드와 관계
MATCH (n)-[r]->(m) RETURN n, r, m

// "이란과 전쟁 중인 나라의 대통령은?"
MATCH (p:Person)-[:대통령]->(c:Country)-[:전쟁중]->(i:Country {name: "이란"})
RETURN p.name

// 이란과 관련된 모든 관계
MATCH (n)-[r]-(i:Country {name: "이란"})
RETURN n.name, type(r)
```

### 4. 정리

```cypher
MATCH (n) DETACH DELETE n
```

## Cypher 핵심 문법

```
CREATE  → 노드/관계 생성
MATCH   → 패턴 매칭 (검색)
RETURN  → 결과 반환
WHERE   → 조건 필터
()      → 노드
-[]->   → 관계 (방향 있음)
-[]-    → 관계 (방향 무관)
```

## Python에서 Neo4j 연결

### 셀 1: 연결

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))

with driver.session() as session:
    result = session.run("RETURN 'Neo4j 연결 성공!' AS message")
    print(result.single()["message"])
```

### 셀 2: Python으로 노드/관계 생성 & 검색

```python
with driver.session() as session:
    session.run("CREATE (t:Person {name: $name, role: $role})", name="트럼프", role="대통령")
    session.run("CREATE (c:Country {name: $name})", name="미국")
    session.run("CREATE (c:Country {name: $name})", name="이란")

    session.run("""
        MATCH (p:Person {name: "트럼프"}), (c:Country {name: "미국"})
        CREATE (p)-[:대통령]->(c)
    """)
    session.run("""
        MATCH (a:Country {name: "미국"}), (b:Country {name: "이란"})
        CREATE (a)-[:전쟁중]->(b)
    """)

    result = session.run("""
        MATCH (p:Person)-[:대통령]->(c:Country)-[:전쟁중]->(target)
        RETURN p.name AS person, target.name AS target
    """)
    for record in result:
        print(f"{record['person']}은 {record['target']}과 전쟁 중인 나라의 대통령")
```

## 관찰할 것

1. Neo4j 브라우저에서 그래프가 시각적으로 보이나?
2. Cypher로 관계를 따라가는 검색이 SQL보다 직관적인가?
3. 뉴스 데이터에 적용하면 어떻게 될지 상상해보세요
