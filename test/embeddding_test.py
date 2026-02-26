import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# API 키 설정
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def run():
    # 1. 문서 로드 & 청킹
    loader = TextLoader("test_doc.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 2. 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cpu"}
    )

    # 3. 벡터 DB 생성
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever()

    # 4. LLM 설정
    llm = ChatOpenAI(model="gpt-4o")

    # 5. 프롬프트 구성
    system_prompt = (
        "당신은 검색된 문맥(context)을 바탕으로 질문에 답하는 인공지능 도우미입니다. "
        "답변은 한국어로 작성하고, 모르는 내용은 지어내지 마세요. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 6. LCEL 방식으로 RAG 체인 구성
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. 실행
    question = "문서의 내용을 요약하고 핵심 정보를 알려줘"
    response = rag_chain.invoke(question)

    print(f"\n질문: {question}")
    print(f"답변: {response}")

if __name__ == "__main__":
    run()