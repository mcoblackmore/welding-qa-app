# -*- coding: utf-8 -*-
"""
Streamlit App for Welding Q&A Expert
"""

import os
import requests
import json
from pathlib import Path
import lancedb
from lancedb.embeddings import EmbeddingFunction
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import LanceTable
from lancedb.rerankers import Reranker
from pydantic_ai import Agent
import tiktoken
import tempfile
import numpy as np
import streamlit as st
import asyncio
import pyarrow as pa
from functools import cached_property
from typing import ClassVar, Optional, List
from lancedb.embeddings import register, TextEmbeddingFunction, get_registry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# --- 基本配置 ---
# For deployment, the API key is fetched from Streamlit's secrets management.
# For local development, it falls back to environment variables.
SILICONFLOW_API_KEY = st.secrets.get("SILICONFLOW_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"

if not SILICONFLOW_API_KEY:
    st.error("请设置您的 SiliconFlow API Key！在本地运行时，请设置环境变量 `SILICONFLOW_API_KEY`；在 Streamlit Cloud 上部署时，请在 Secrets 中设置。")
    st.stop()

# --- 自定义 Embedding 和 Reranker 模型 (来自你的原始代码) ---

@register("siliconflow")
class SiliconFlowEmbeddingFunction(TextEmbeddingFunction):
    name: str = "Qwen/Qwen3-Embedding-8B"
    api_key: Optional[str] = None

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-large-zh-v1.5": 1024,
        "netease-youdao/bce-embedding-base_v1": 768,
        "BAAI/bge-m3": 1024,
        "Qwen/Qwen3-Embedding-8B": 4096
    }

    @cached_property
    def _resolved_api_key(self) -> str:
        key = self.api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise ValueError("SiliconFlow API key is required.")
        return key

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        url = f"{BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._resolved_api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": self.name, "input": texts}

        try:
            # Add a timeout to prevent the request from hanging indefinitely
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            embeddings_data = data.get('data', [])
            sorted_embeddings = sorted(embeddings_data, key=lambda x: x['index'])
            embeddings = [item['embedding'] for item in sorted_embeddings]
            if len(embeddings) != len(texts):
                raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            return embeddings
        except requests.exceptions.RequestException as e:
            st.error(f"调用 SiliconFlow Embedding API 失败: {e}")
            raise
        except (KeyError, IndexError) as e:
            st.error(f"解析 SiliconFlow Embedding API 响应失败: {e}")
            raise

    def ndims(self) -> int:
        dim = self.MODEL_DIMENSIONS.get(self.name)
        if dim is not None:
            return dim
        try:
            test_embedding = self.generate_embeddings(["test"])
            return len(test_embedding[0])
        except Exception as e:
            raise RuntimeError(f"无法确定模型维度 {self.name}: {e}")


class SiliconFlowReranker(Reranker):
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-8B", api_key: str = SILICONFLOW_API_KEY, return_score="_relevance_score"):
        super().__init__()
        self._score_column = return_score
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")

    def _rerank(self, query: str, documents_table: pa.Table) -> pa.Table:
        if not self._api_key:
            raise ValueError("SiliconFlow API key is required.")

        docs_list = documents_table.to_pylist()
        if not docs_list:
            return documents_table

        texts = [str(doc.get("text", "")) for doc in docs_list]
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "query": query, "documents": texts, "return_documents": False}
        scores = [0.0] * len(docs_list)

        try:
            response = requests.post(f"{BASE_URL}/rerank", json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            api_results = response.json().get("results", [])
            for result in api_results:
                original_index = result.get('index')
                score = result.get('relevance_score')
                if score is not None and 0 <= original_index < len(scores):
                    scores[original_index] = score
        except requests.exceptions.RequestException as e:
            print(f"Reranker API request failed: {e}. Documents will not be reranked.")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Failed to parse reranker API response: {e}. Documents will not be reranked.")

        scores_array = pa.array(scores, type=pa.float64())
        if self._score_column in documents_table.column_names:
            return documents_table.set_column(documents_table.column_names.index(self._score_column), self._score_column, scores_array)
        else:
            return documents_table.append_column(self._score_column, scores_array)

# --- LanceDB 和知识库函数 ---

embed_func = SiliconFlowEmbeddingFunction.create(api_key=SILICONFLOW_API_KEY)

class Document(LanceModel):
    id: str
    text: str = embed_func.SourceField()
    vector: Vector(4096) = embed_func.VectorField()

def chunk_text(text: str, max_tokens: int = 256, overlap_tokens: int = 50, encoding_name: str = "cl100k_base"):
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(text))
    step = max_tokens - overlap_tokens
    for i in range(0, len(tokens), step):
        yield encoding.decode(tokens[i:i + max_tokens])

def create_lancedb_table(db_path: str, table_name: str, overwrite: bool = True):
    db = lancedb.connect(db_path)
    mode = 'overwrite' if overwrite else 'create'
    table = db.create_table(table_name, schema=Document, mode=mode)
    table.create_fts_index("text", replace=overwrite)
    return table

def add_documents_to_table(table: LanceTable, knowledge_base_dir: str, max_tokens: int = 256):
    docs = []
    knowledge_base = Path(knowledge_base_dir)
    md_files = list(knowledge_base.glob("*.md"))

    if not md_files:
        st.warning(f"在 '{knowledge_base_dir}' 目录下没有找到 Markdown 文件 (.md)。知识库将为空。")
        return

    with st.spinner(f"正在处理 {len(md_files)} 个知识库文件..."):
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    text = f.read()
                for i, chunk in enumerate(chunk_text(text, max_tokens=max_tokens)):
                    doc_id = f"{md_file.stem}_{i}"
                    clean_chunk = str(chunk).strip()
                    if clean_chunk:
                        docs.append({"id": doc_id, "text": clean_chunk})
            except Exception as e:
                st.error(f"处理文件 {md_file.name} 时出错: {e}")

    if docs:
        with st.spinner(f"正在向数据库添加 {len(docs)} 个文档块..."):
            batch_size = 32
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                try:
                    table.add(batch)
                except Exception as e:
                    st.error(f"添加文档批次时出错: {e}")
    st.success(f"知识库加载完成，共添加 {len(docs)} 个文档块。")


def retrieve_similar_docs(table: LanceTable, query: str, query_type: str = 'hybrid', limit: int = 100):
    try:
        reranker = SiliconFlowReranker()
        results = (
            table.search(query, query_type=query_type)
            .limit(limit)
            .rerank(reranker=reranker)
            .to_list()
        )
        return results
    except Exception as e:
        st.error(f"检索或重排序文档时出错: {e}")
        return table.search(query, query_type=query_type).limit(limit).to_list()

# --- Agent 设置 ---

@st.cache_resource
def get_llm_provider():
    return OpenAIProvider(base_url=BASE_URL, api_key=SILICONFLOW_API_KEY)

@st.cache_resource
def setup_knowledge_query_agent():
    model = OpenAIModel('Qwen/Qwen3-235B-A22B-Instruct-2507', provider=get_llm_provider())
    return Agent(
        name="Knowledge Query Agent",
        model=model,
        system_prompt="From the input text string, please generate a concise query string to pass to the knowledge base."
    )

@st.cache_resource
def setup_main_agent():
    model = OpenAIModel('Qwen/Qwen3-235B-A22B-Instruct-2507', provider=get_llm_provider())
    system_prompt = (
        "You are a helpful assistant. "
        "Please answer the user's question based on the context provided within their prompt."
    )
    return Agent(name="Main Agent", model=model, system_prompt=system_prompt)

@st.cache_resource
def setup_database():
    db_path = "db"
    table_name = "knowledge"
    knowledge_base_dir = "knowledge-file"
    
    # 检查知识库目录是否存在
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)
        st.info(f"已创建 '{knowledge_base_dir}' 目录。请将您的知识库文件（.md）放入其中并刷新页面。")

    db = lancedb.connect(db_path)
    
    # 检查表是否存在，如果不存在或为空则创建
    if table_name not in db.table_names():
        st.info("未找到现有知识库，正在创建新库...")
        table = create_lancedb_table(db_path, table_name, overwrite=True)
        add_documents_to_table(table, knowledge_base_dir)
    else:
        table = db.open_table(table_name)
        if len(table) == 0:
            st.info("知识库为空，正在加载文档...")
            add_documents_to_table(table, knowledge_base_dir)
        else:
            st.success("已成功加载现有知识库。")
            
    return table

# --- 主应用逻辑 ---

async def get_answer(query: str, agent, knowledge_query_agent, knowledge_table, message_history):
    # 1. 生成知识库查询
    res = await knowledge_query_agent.run(query)
    knowledge_query = res.output
    
    # 2. 检索文档
    retrieved_docs = retrieve_similar_docs(knowledge_table, knowledge_query, limit=10)
    
    # 3. 构建上下文
    relevant_docs = [doc for doc in retrieved_docs if doc.get("_relevance_score", 0) > 0.5]
    if relevant_docs:
        knowledge_context = "\n".join(doc["text"] for doc in relevant_docs)
    else:
        knowledge_context = "\n".join(doc["text"] for doc in retrieved_docs[:3]) if retrieved_docs else "No relevant documents found."

    # 4. 构建最终提示
    final_user_prompt = (
        f"Here is some context from the knowledge base:\n"
        f"---CONTEXT_START---\n"
        f"{knowledge_context}\n"
        f"---CONTEXT_END---"

        f"Based on the context above, please answer my question: {query}"
    )
    
    # 5. 获取最终答案
    response = await agent.run(user_prompt=final_user_prompt, message_history=message_history)
    
    return response.output, response.new_messages()


# --- Streamlit UI ---

st.title("焊接标准问答专家")

# 初始化
try:
    knowledge_table = setup_database()
    knowledge_query_agent = setup_knowledge_query_agent()
    main_agent = setup_main_agent()
except Exception as e:
    st.error(f"应用初始化失败: {e}")
    st.stop()


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pydantic_history" not in st.session_state:
    st.session_state.pydantic_history = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您关于焊接标准的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成并显示AI回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("思考中..."):
            try:
                # 在Streamlit中运行异步函数
                answer, new_pydantic_history = asyncio.run(get_answer(
                    prompt,
                    main_agent,
                    knowledge_query_agent,
                    knowledge_table,
                    st.session_state.pydantic_history
                ))
                message_placeholder.markdown(answer)
                st.session_state.pydantic_history = new_pydantic_history
            except Exception as e:
                error_message = f"处理请求时出错: {e}"
                st.error(error_message)
                answer = "抱歉，我遇到了一些麻烦，无法回答您的问题。"

    st.session_state.messages.append({"role": "assistant", "content": answer})
