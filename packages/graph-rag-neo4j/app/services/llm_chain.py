from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings

SYSTEM_PROMPT = """\
당신은 지식 그래프 기반 PDF 문서 질의응답 전문가입니다.
주어진 컨텍스트(문서 청크 + 지식 그래프 관계)를 활용하여 질문에 정확하게 답변하세요.

규칙:
1. 컨텍스트에 없는 내용은 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답하세요.
2. 답변 시 출처(문서명, 페이지)를 언급하세요.
3. 지식 그래프 관계 정보가 있으면 이를 활용하여 엔티티 간의 연결을 설명하세요.
4. 테이블 데이터는 구조를 유지하여 답변하세요.
5. 한국어로 답변하되, 고유명사나 기술 용어는 원문 그대로 사용하세요."""


class LLMChain:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature

    async def generate(self, query: str, context: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(query, context)},
            ],
        )
        return response.choices[0].message.content

    async def stream(self, query: str, context: str) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_prompt(query, context)},
            ],
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return f"""다음 컨텍스트를 참고하여 질문에 답변하세요.
컨텍스트에는 문서 청크와 지식 그래프 관계 정보가 포함될 수 있습니다.

## 컨텍스트
{context}

## 질문
{query}

## 답변"""
