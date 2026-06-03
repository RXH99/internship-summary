# rag-bench 代码逐行全解（零基础版）

> 目标：看完能说出每个 `.py` 文件里每一行代码在干什么。
> 前提：知道什么是变量、列表、字典、for 循环、if 判断。

---

# 项目结构总览

```
rag-bench/
│
├── rag/                          # 核心 RAG 库
│   ├── __init__.py               # 空文件，让 rag/ 成为 Python 包
│   ├── config.py                 # 集中配置管理（dataclass）
│   ├── loader.py                 # 文档加载器（纯 Python）
│   └── pipeline.py               # RAGPipeline 核心类
│
├── eval/                         # 评估系统
│   ├── __init__.py               # 空文件
│   ├── benchmark.py              # 对比实验（6 种配置）
│   └── questions.json            # 15 个 QA 测试对
│
├── scripts/                      # 可执行入口
│   ├── query.py                  # 交互式问答 CLI
│   ├── build_db.py               # 构建向量库
│   └── app.py                    # Streamlit Web 界面
│
├── data/                         # 知识库文档（4 篇 .txt）
├── results/                      # Benchmark 报告
├── requirements.txt              # Python 依赖
└── README.md                     # 项目文档
```

---

# 第一部分：rag/config.py — 集中配置管理

## 1.1 文件头

```python
"""
rag/config.py — 集中管理所有配置参数
"""

from dataclasses import dataclass, field
from typing import List
```

### from dataclasses import dataclass, field

Python 的 `dataclass` 是一个**装饰器**，能自动生成 `__init__`（构造函数）、`__repr__`（打印）、`__eq__`（比较）等方法。

```python
# 不用 dataclass 的写法：
class EmbeddingConfig:
    def __init__(self, model="nomic-embed-text", dimensions=768):
        self.model = model
        self.dimensions = dimensions

# 用 dataclass 的写法：
@dataclass
class EmbeddingConfig:
    model: str = "nomic-embed-text"
    dimensions: int = 768
# Python 自动帮你写了 __init__，完全等价！
```

### dataclass 的优点

1. **省代码** — 不用手写 `__init__`
2. **类型提示** — `model: str` 自动标注类型
3. **默认值** — `= "nomic-embed-text"` 指定默认值
4. **不可变选项** — 加 `frozen=True` 就不能修改了

---

## 1.2 EmbeddingConfig — 嵌入模型配置

```python
@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model: str = "nomic-embed-text"
    dimensions: int = 768
    base_url: str = "http://localhost:11434"
```

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"nomic-embed-text"` | Ollama 上的嵌入模型名称 |
| `dimensions` | `768` | 嵌入向量的维度（每个文档转成 768 个数字） |
| `base_url` | `"http://localhost:11434"` | Ollama 服务的地址和端口 |

### 什么是嵌入向量？

```
"RAG是检索增强生成"  →  嵌入模型  →  [0.12, -0.45, 0.78, ..., 0.33]
                      (nomic-embed-text)     ↑ 768 个浮点数 ↑
```

嵌入向量就是把**文字变成一串数字**。语义相近的文字，它们的数字向量在"向量空间"里也离得近。

### 为什么 base_url 是 localhost:11434？

Ollama 默认在本地 11434 端口启动服务。这个配置告诉代码：去 `http://localhost:11434` 找 Ollama 要嵌入向量。

```python
# 如果你在另一台机器上跑 Ollama，可以改：
EmbeddingConfig(base_url="http://192.168.1.100:11434")
```

---

## 1.3 LLMConfig — 大语言模型配置

```python
@dataclass
class LLMConfig:
    """大语言模型配置"""
    model: str = "qwen:4b"
    temperature: float = 0.1
    base_url: str = "http://localhost:11434"
```

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `model` | `"qwen:4b"` | 通义千问 40 亿参数版本（轻量） |
| `temperature` | `0.1` | 生成随机性（0=确定，1=创意） |
| `base_url` | `"http://localhost:11434"` | Ollama 服务地址 |

### temperature 是什么？

```
temperature=0.0  →  每次都选最可能的词  →  回答很"稳"
temperature=0.1  →  偶尔选次可能的词  →  回答有少量变化
temperature=1.0  →  经常选不太可能的词  →  回答很"有创意"
```

RAG 问答中 `temperature=0.1` 是合适的——我们希望回答基于检索到的文档，不要自己编造，所以确定性高一点好。

---

## 1.4 ChunkConfig — 文档分块配置

```python
@dataclass
class ChunkConfig:
    """文档分割配置"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", " ", ""])
```

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | `500` | 每个文本片段的字符数 |
| `chunk_overlap` | `50` | 相邻片段重叠字符数（防止切断语义） |
| `separators` | `["\n\n", "\n", "。", " ", ""]` | 分割优先级 |

### field(default_factory=lambda: ...) 

这里有一个关键语法点：

```python
# ❌ 错误的写法（所有实例共享同一个列表！）
separators: List[str] = ["\n\n", "\n", "。", " ", ""]

# ✅ 正确的写法（每个实例创建自己的列表）
separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", "。", " ", ""])
```

**为什么？** Python 的函数默认值只在定义时创建一次。用 `field(default_factory=...)` 确保每次创建实例都生成新的列表。

### lambda 是什么？

```python
lambda: ["\n\n", "\n", "。", " ", ""]
# 等价于：
def _工厂函数():
    return ["\n\n", "\n", "。", " ", ""]
```

`lambda` 是一种**匿名函数**，只用一行就能定义。格式是 `lambda 参数: 返回值`。

### 分割优先级图解

```
文档字符串如："段落1\n\n段落2\n这里还有内容。句子A。句子B"

第一步：尝试用 "\n\n" 切  →  分成 [段落1] [段落2\n这里还有内容。句子A。句子B]
                              (但不一定刚好 500 字符，所以继续)
第二步：对超长的片段用 "\n" 切
第三步：还不够，用 "。" 切（中文句号）
第四步：再不行用 " " 切（英文空格）
第五步：最后用 "" 切（逐字符，保证不超过 chunk_size）
```

---

## 1.5 RetrievalConfig — 检索配置

```python
@dataclass
class RetrievalConfig:
    """检索配置"""
    search_type: str = "similarity"  # "similarity" | "mmr"
    k: int = 3
    # MMR 专用参数
    fetch_k: int = 10
    lambda_mult: float = 0.5
```

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `search_type` | `"similarity"` | 检索策略（Top-K / MMR） |
| `k` | `3` | 返回的最相关文档数 |
| `fetch_k` | `10` | MMR 先取 10 个候选，再选 3 个不同的 |
| `lambda_mult` | `0.5` | MMR 多样性平衡参数 |

### Top-K vs MMR

```
Top-K（相似度检索）：
  找与问题最像的 3 个文档
  → [doc_A, doc_B, doc_C]  ← 全是相似内容

MMR（最大边际相关性）：
  先拿 10 个候选，再从中挑 3 个"又相关又不同"的
  → [doc_A, doc_D, doc_E]  ← 覆盖更广
```

### lambda_mult 的作用

```
lambda_mult = 0.5  →  相关性和多样性 50:50 平衡
lambda_mult = 1.0  →  只看相关性（和 Top-K 一样）
lambda_mult = 0.0  →  只看多样性（可能完全不相关）
```

---

## 1.6 DataConfig 和 PromptConfig

```python
@dataclass
class DataConfig:
    """数据路径配置"""
    data_dir: str = "data"
    chroma_dir: str = "chroma_db"
```

| 属性 | 默认值 | 说明 |
|------|--------|------|
| `data_dir` | `"data"` | 原始文本文件夹路径 |
| `chroma_dir` | `"chroma_db"` | Chroma 向量库存储路径 |

```python
@dataclass
class PromptConfig:
    """提示词模板"""
    template: str = """你是一个知识库助手，请严格根据以下上下文内容回答问题。
如果上下文中没有提供相关信息，请不要编造，直接回答"根据现有知识库，我无法回答这个问题"。

上下文：
{context}

问题：{question}

回答（只基于以上上下文）："""
```

### 模板里的 {context} 和 {question}

这是 Python 字符串模板格式。

```python
template = "你好，{name}！"
template.format(name="小明")  # → "你好，小明！"

# 在 RAG 中：
template.format(
    context="RAG是检索增强生成...\n[来源: rag_deep_dive.txt]",
    question="什么是RAG？",
)
# 结果：
# "你是一个知识库助手...上下文：RAG是检索增强生成... 问题：什么是RAG？"
```

### 为什么 prompt 要强调"不要编造"？

大语言模型有一个问题叫**幻觉（Hallucination）**：它不知道答案时会自己编造看起来合理的回答。RAG 的核心目的就是减少幻觉——让模型**只基于检索到的文档**回答，找不到就说不知道。

---

## 1.7 config.py 总结

| 类 | 管什么 |
|-----|--------|
| `EmbeddingConfig` | 把文字变成向量的模型 |
| `LLMConfig` | 生成答案的模型 |
| `ChunkConfig` | 如何把文档切成小段 |
| `RetrievalConfig` | 如何找到相关的段 |
| `DataConfig` | 文件存在哪里 |
| `PromptConfig` | 怎么问模型才能得到好答案 |

---

# 第二部分：rag/loader.py — 文档加载器

## 2.1 文件头

```python
"""
rag/loader.py — 文档加载器

纯 Python 实现，不依赖 langchain-community，无弃用警告。
"""

from pathlib import Path
from typing import List, Dict
```

### 为什么强调"纯 Python 实现"？

`langchain-community` 曾经提供文本加载功能，但被标记为**弃用**（DeprecationWarning）。弃用意味着未来版本会删除。所以项目自己写了一个简单的加载器，不依赖这个库，也就没有弃用警告。

---

## 2.2 load_text_files() 函数

```python
def load_text_files(data_dir: str) -> List[Dict[str, str]]:
    """
    加载 data/ 目录下所有 .txt 文件

    返回:
        [{"source": "filename.txt", "content": "文件内容..."}, ...]
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    docs = []
    for file in sorted(data_path.glob("*.txt")):
        content = file.read_text(encoding="utf-8")
        docs.append({
            "source": file.name,
            "content": content,
        })
        print(f"  [load] {file.name} ({len(content)} chars)")

    print(f"  共 {len(docs)} 个文档, "
          f"{sum(len(d['content']) for d in docs)} 总字符数")
    return docs
```

### 逐行拆解

```python
def load_text_files(data_dir: str) -> List[Dict[str, str]]:
# data_dir: str  →  参数是一个字符串（文件夹路径）
# -> List[Dict[str, str]]  →  返回值是一个列表，每个元素是{"source": str, "content": str}
```

```python
data_path = Path(data_dir)
# Path("data")  →  创建一个路径对象
# 比 os.path.join() 更简洁好用
# Path("data") / "sub" / "file.txt"  →  "data/sub/file.txt"
```

```python
if not data_path.exists():
    raise FileNotFoundError(f"数据目录不存在: {data_dir}")
# 如果文件夹不存在，立即报错退出
# raise → 抛出一个异常，程序会停止
# 这是"防御性编程"——早失败早发现
```

```python
docs = []
# 创建一个空列表，用来存储文档
```

```python
for file in sorted(data_path.glob("*.txt")):
    # data_path.glob("*.txt")  →  匹配所有 .txt 文件
    # sorted(...)  →  按文件名排序（确保顺序一致）
    # file 是一个 Path 对象，如 Path("data/ai_agent_intro.txt")
```

```python
    content = file.read_text(encoding="utf-8")
    # file.read_text()  →  Path 对象的读取方法
    # encoding="utf-8"  →  用 UTF-8 编码读取（中文必须）
    # 等价于：
    # with open(file, "r", encoding="utf-8") as f:
    #     content = f.read()
```

```python
    docs.append({
        "source": file.name,
        # file.name  →  "ai_agent_intro.txt"（不含路径）
        "content": content,
        # content  →  文件的全部文本
    })
```

```python
    print(f"  [load] {file.name} ({len(content)} chars)")
    # 打印：  [load] ai_agent_intro.txt (1234 chars)
```

```python
print(f"  共 {len(docs)} 个文档, "
      f"{sum(len(d['content']) for d in docs)} 总字符数")
# len(docs)  →  文档数量
# sum(...)  →  所有文档内容字符数的总和
#   len(d['content']) for d in docs  →  生成器表达式
#   等价于：
#   total = 0
#   for d in docs:
#       total += len(d['content'])
```

```python
return docs
# 返回文档列表给调用者
```

---

# 第三部分：rag/pipeline.py — RAG Pipeline 核心类

## 3.1 文件头 — import 语句

```python
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma

from .config import (
    EmbeddingConfig, LLMConfig, ChunkConfig,
    RetrievalConfig, DataConfig, PromptConfig,
)
from .loader import load_text_files
```

### import hashlib

```python
import hashlib

# 计算字符串的 SHA256 哈希
content = "RAG是检索增强生成"
hash_obj = hashlib.sha256(content.encode())
hash_hex = hash_obj.hexdigest()
# → "a1b2c3d4e5f6g7h8..."（64个十六进制字符）
```

**哈希是什么？** 任何一个输入都产生一个固定长度的指纹。输入稍微一变，指纹完全不同。用于验证文件没有被篡改。

### 相对导入 from .xxx import ...

```python
from .config import EmbeddingConfig
# . 表示"当前包目录"
# 在 pipeline.py 中，.config 表示 rag/config.py
# 因为是 rag 包内部的导入，所以用相对路径
```

```python
from .loader import load_text_files
# .loader 表示 rag/loader.py
```

### langchain_xxx 导入

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 文档分割器，按优先级（段落→句子→词语）切割

from langchain_ollama import OllamaEmbeddings, OllamaLLM
# OllamaEmbeddings  →  调用 Ollama 的嵌入模型生成向量
# OllamaLLM         →  调用 Ollama 的大模型生成文本

from langchain_chroma import Chroma
# Chroma 向量数据库的操作接口
```

### __init__.py 的作用

```python
# rag/__init__.py
# 空文件即可
# 作用是告诉 Python：rag/ 是一个"包"（package）
# 没有这个文件，就不能 from .xxx import yyy
```

---

## 3.2 RAGPipeline 类定义

```python
class RAGPipeline:
    """
    RAG Pipeline — 检索增强生成全流程
    
    用法：
        pipe = RAGPipeline()
        pipe.build_vectorstore()        # 加载文档 → 分割 → 建向量库
        pipe.retrieve("什么是RAG？")      # 仅检索
        pipe.query("什么是RAG？")         # 检索 + 生成
    """
```

这个类封装了 RAG 的**完整流程**：

```
加载文档 → 分割成片段 → 向量化 → 存入向量库
                                           ↓
用户提问 → 问题向量化 → 向量库检索 → 组装上下文 → LLM 生成答案
```

---

## 3.3 __init__() 方法

```python
def __init__(
    self,
    embedding_cfg: Optional[EmbeddingConfig] = None,
    llm_cfg: Optional[LLMConfig] = None,
    chunk_cfg: Optional[ChunkConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    prompt_cfg: Optional[PromptConfig] = None,
):
    self.embedding_cfg = embedding_cfg or EmbeddingConfig()
    self.llm_cfg = llm_cfg or LLMConfig()
    self.chunk_cfg = chunk_cfg or ChunkConfig()
    self.data_cfg = data_cfg or DataConfig()
    self.prompt_cfg = prompt_cfg or PromptConfig()

    self._embeddings = OllamaEmbeddings(
        model=self.embedding_cfg.model,
        base_url=self.embedding_cfg.base_url,
    )
    self._vectordb: Optional[Chroma] = None
    self._llm: Optional[OllamaLLM] = None
```

### Optional[Type] = None

```python
embedding_cfg: Optional[EmbeddingConfig] = None
# 意思是：这个参数可以是 EmbeddingConfig 类型，也可以是 None
# 默认是 None
```

### A or B 技巧

```python
self.embedding_cfg = embedding_cfg or EmbeddingConfig()
# 如果 embedding_cfg 不是 None（传入了配置），就用传入的
# 如果 embedding_cfg 是 None（没传），就用默认的 EmbeddingConfig()
# 
# Python 中 or 的机制：
#   None or "默认值"  →  "默认值"
#   "传入的值" or ... →  "传入的值"
```

### 好处是什么？

```python
# 用户可以自定义配置：
custom_cfg = EmbeddingConfig(model="bge-large", dimensions=1024)
pipe = RAGPipeline(embedding_cfg=custom_cfg)

# 也可以完全用默认值：
pipe = RAGPipeline()
```

### self._vectordb = None

```python
self._vectordb: Optional[Chroma] = None
# 一开始向量库为空（还没建）
# 等 build_vectorstore() 调用后才赋值
# 下划线 _vectordb 表示"内部使用"，外部不应该直接访问
```

---

## 3.4 build_vectorstore() — 构建向量库

```python
def build_vectorstore(self, chunk_size: Optional[int] = None) -> int:
    if chunk_size:
        self.chunk_cfg.chunk_size = chunk_size

    print(f"\n[build] chunk_size={self.chunk_cfg.chunk_size}, "
          f"overlap={self.chunk_cfg.chunk_overlap}")
```

### 1. 加载文档

```python
    raw_docs = load_text_files(self.data_cfg.data_dir)
    # 调用 loader.py 中的函数
    # 返回：[{"source": "ai_agent_intro.txt", "content": "..."}, ...]
```

### 2. 分割文档

```python
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=self.chunk_cfg.chunk_size,     # 每段 500 字符
        chunk_overlap=self.chunk_cfg.chunk_overlap, # 重叠 50 字符
        separators=self.chunk_cfg.separators,       # 分割优先级
    )
```

#### RecursiveCharacterTextSplitter 原理

```
输入文本（2000 字符）
    │
    ├─ 尝试用 "\n\n"（段落分隔）切 → 可能有片段 >500
    │      │
    │      └─ 对超长的片段改用 "\n"（行分隔）切
    │             │
    │             └─ 还超长？用 "。"（句号）切
    │                    │
    │                    └─ 还超长？用 " "（空格）切
    │                           │
    │                           └─ 最后用 ""（逐字符）切
    │
    输出：多个 500 字符的片段，相邻之间保留 50 字符重叠
```

### 3. 转换为 LangChain Document

```python
    lc_docs = [
        {"page_content": d["content"], "metadata": {"source": d["source"]}}
        for d in raw_docs
    ]
    # 把 loader.py 返回的格式转成 LangChain 需要的格式
```

### 4. 包装为 Doc 对象

```python
    class Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    lc_docs_obj = [Doc(d["page_content"], d["metadata"]) for d in lc_docs]
```

这里定义了一个**临时类** `Doc`，只有两个属性：`page_content`（内容）和 `metadata`（元数据）。为什么要自己定义？因为 `RecursiveCharacterTextSplitter` 的 `split_documents()` 方法需要传入的文档有 `page_content` 和 `metadata` 属性，而不是字典。

### 5. 执行分割

```python
    chunks = splitter.split_documents(lc_docs_obj)
    # 分割后，1 篇文档可能变成多个片段
    # chunks → [Doc1, Doc2, ...]
    # 每个 Doc 的 page_content 是 500 字左右的文本片段
    print(f"  → {len(chunks)} 个文本片段")
```

### 6. 构建向量库并持久化

```python
    chroma_dir = Path(self.data_cfg.chroma_dir)
    chroma_dir.mkdir(exist_ok=True)
    # mkdir(exist_ok=True)  →  如果目录已存在，不报错

    self._vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=self._embeddings,          # 用 nomic-embed-text 做嵌入
        persist_directory=str(chroma_dir),   # 保存到 chroma_db/ 目录
    )
    print(f"  → 已保存到 {chroma_dir}/")
    return len(chunks)
```

### Chroma.from_documents() 内部发生了什么？

```
每个文档片段 → nomic-embed-text → 768 维向量 → 存入 Chroma
                   ↑
            Ollama 服务 (localhost:11434)
```

Chroma 会：
1. 对每个片段调用 `self._embeddings.embed_documents()`
2. 把向量和原文一起存入 `chroma_db/` 目录
3. 下次启动时可以直接加载，不用重新计算

---

## 3.5 load_vectorstore() — 加载已有向量库

```python
def load_vectorstore(self) -> bool:
    chroma_dir = Path(self.data_cfg.chroma_dir)
    if not chroma_dir.exists():
        return False  # 库不存在，返回 False

    self._vectordb = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=self._embeddings,
    )
    print(f"  [load] 从 {chroma_dir}/ 加载向量库")
    return True
```

### 为什么需要这个方法？

构建向量库很耗时（文档越多越慢）。如果之前已经构建过了，直接加载就能用，不用重新算一遍。

### persist_directory 参数

```python
Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# 告诉 Chroma：去 chroma_db/ 目录读之前存好的数据
# embedding_function 是查询时用的，不是加载时用的
```

---

## 3.6 retrieve() — 检索方法

```python
def retrieve(
    self,
    query: str,
    search_type: str = "similarity",
    k: int = 3,
    fetch_k: int = 10,
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    检索与 query 最相关的文档片段
    
    参数:
        query: 用户问题
        search_type: "similarity" | "mmr"
        k: 返回的文档数量
        fetch_k: (MMR) 预取数量
        lambda_mult: (MMR) 多样性系数
    
    返回:
        [{"content": "...", "source": "...", "score": ...}, ...]
    """
    if not self._vectordb:
        loaded = self.load_vectorstore()
        if not loaded:
            raise RuntimeError("向量库未构建，请先调用 build_vectorstore()")
```

### 自动加载逻辑

```python
if not self._vectordb:
    # self._vectordb 是 None（还没建或没加载）
    loaded = self.load_vectorstore()
    # 尝试加载已有向量库
    if not loaded:
        # 加载失败（目录不存在），报错
        raise RuntimeError("...")
    # 加载成功，继续执行
```

### 设置检索参数

```python
    kwargs = {"k": k}  # 基础参数：返回 3 个文档
    if search_type == "mmr":
        kwargs["fetch_k"] = fetch_k      # 预取 10 个
        kwargs["lambda_mult"] = lambda_mult  # 多样性 0.5
```

### 创建检索器

```python
    retriever = self._vectordb.as_retriever(
        search_type=search_type,   # "similarity" 或 "mmr"
        search_kwargs=kwargs,      # 上面的参数
    )
```

`.as_retriever()` 是 Chroma 的一个方法，把向量数据库包装成一个"检索器"对象。这个检索器只有一个功能：给定问题，返回最相关的文档。

### 执行检索

```python
    docs = retriever.invoke(query)
    # invoke → 执行检索
    # query → "什么是RAG？"
    # 返回 → 3 个 Document 对象
```

### 格式化结果

```python
    results = []
    for doc in docs:
        results.append({
            "content": doc.page_content,          # 文档内容
            "source": doc.metadata.get("source", "unknown"),  # 来源文件名
        })
    return results
```

---

## 3.7 query() — 完整 RAG 问答

```python
def query(
    self,
    question: str,
    search_type: str = "similarity",
    k: int = 3,
    return_sources: bool = True,
) -> Dict[str, Any]:
```

### 1. 检索

```python
    retrieved = self.retrieve(question, search_type=search_type, k=k)
    # 返回：[{"content": "...", "source": "..."}, ...]
```

### 2. 处理空结果

```python
    if not retrieved:
        return {
            "answer": "未能检索到相关文档，无法回答问题。",
            "sources": [],
        }
```

### 3. 组装上下文

```python
    context = "\n\n".join(
        f"[来源: {r['source']}]\n{r['content']}" for r in retrieved
    )
    # 把 3 个文档拼接成一个字符串，用两个换行分隔
    # 例如：
    # [来源: rag_deep_dive.txt]
    # RAG是检索增强生成...
    #
    # [来源: ai_agent_intro.txt]
    # AI编程助手是一种...
```

### format() 方法

```python
    prompt = self.prompt_cfg.template.format(
        context=context,
        question=question,
    )
    # 把模板中的 {context} 替换为实际上下文
    # 把模板中的 {question} 替换为用户问题
```

### 4. 调用 LLM

```python
    if not self._llm:
        self._llm = OllamaLLM(
            model=self.llm_cfg.model,               # qwen:4b
            temperature=self.llm_cfg.temperature,    # 0.1
            base_url=self.llm_cfg.base_url,          # localhost:11434
        )
    # 懒加载：第一次调 query() 时才创建 LLM 实例
```

### 5. 生成回答

```python
    answer = self._llm.invoke(prompt)
    # invoke → 把 prompt 发给 Ollama → qwen:4b 生成回答
```

### 6. 返回结果

```python
    result = {"answer": answer.strip()}
    # answer.strip() → 去掉首尾空白
    if return_sources:
        result["sources"] = [
            {"source": r["source"]} for r in retrieved
        ]
    return result
    # 返回：
    # {
    #     "answer": "RAG是检索增强生成...",
    #     "sources": [{"source": "rag_deep_dive.txt"}, ...]
    # }
```

---

## 3.8 get_vectorstore_info() — 工具方法

```python
def get_vectorstore_info(self) -> Dict[str, Any]:
    """获取向量库信息"""
    if not self._vectordb:
        return {"status": "not_loaded"}

    try:
        count = self._vectordb._collection.count()
    except Exception:
        count = "unknown"

    return {
        "status": "loaded",
        "chroma_dir": self.data_cfg.chroma_dir,
        "embedding_model": self.embedding_cfg.model,
        "document_count": count,
    }
```

### try/except 保护

```python
try:
    count = self._vectordb._collection.count()
    # 尝试获取文档数量
except Exception:
    count = "unknown"
    # 如果失败（比如数据库还没完全初始化），用 "unknown"
    # 而不是让程序崩溃
```

---

# 第四部分：eval/benchmark.py — 对比实验

这是整个项目中最复杂的文件，实现了自动化的 RAG 效果对比实验。

## 4.1 文件头

```python
import sys
import json
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import RAGPipeline
from rag.config import ChunkConfig, RetrievalConfig, DataConfig
from rag.loader import load_text_files
```

### sys.path.insert(0, str(ROOT))

```python
ROOT = Path(__file__).resolve().parent.parent
# __file__  →  当前文件路径（eval/benchmark.py）
# .resolve()  →  转绝对路径
# .parent  →  eval/
# .parent.parent  →  rag-bench/（项目根目录）

sys.path.insert(0, str(ROOT))
# 把项目根目录加到 Python 模块搜索路径的最前面
# 这样就能直接 from rag.pipeline import RAGPipeline
```

### import shutil

```python
import shutil
# shutil 是标准库，提供高级文件操作
# shutil.rmtree(path)  →  递归删除整个目录（像 rm -rf）
```

---

## 4.2 配置参数

```python
CHUNK_SIZES = [300, 500, 1000]
CHUNK_OVERLAP = 50
TOP_K = 3

RETRIEVAL_CONFIGS = [
    RetrievalConfig(search_type="similarity", k=TOP_K),
    RetrievalConfig(search_type="mmr", k=TOP_K, fetch_k=10, lambda_mult=0.5),
]

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_DIR = ROOT / "results"
```

### 实验设计

```
6 种配置组合：
┌────────────────┬────────────┬────────────┐
│ chunk_size     │ Top-K      │ MMR        │
├────────────────┼────────────┼────────────┤
│ 300            │ cs=300,TK  │ cs=300,MMR │
│ 500            │ cs=500,TK  │ cs=500,MMR │
│ 1000           │ cs=1000,TK │ cs=1000,MMR│
└────────────────┴────────────┴────────────┘
```

---

## 4.3 evaluate() 函数

```python
def evaluate(chroma_dir: str, questions: list,
             search_type: str, k: int, label: str) -> dict:
```

### 基本设置

```python
    cfg = RetrievalConfig(search_type=search_type, k=k)
    data_cfg = DataConfig(chroma_dir=chroma_dir)
    pipe = RAGPipeline(data_cfg=data_cfg)
    pipe.load_vectorstore()
```

### 逐题评估

```python
    total = len(questions)  # 15 个问题
    hits = 0                # 命中数
    top1_scores = []        # 每个问题的 Top-1 精度

    for q in questions:
        keywords = q["expected_keywords"]  # 预期的关键词列表
        retrieved = pipe.retrieve(q["question"], search_type=search_type, k=k)
```

### 命中判断

```python
        if not retrieved:
            top1_scores.append(0.0)
            continue  # 没检索到任何文档，跳过

        # 命中判断：top_k 中是否有任一篇档包含全部关键词
        is_hit = False
        for doc in retrieved:
            text = doc["content"]
            if all(kw.lower() in text.lower() for kw in keywords):
                is_hit = True
                break
```

### all() 函数

```python
keywords = ["RAG", "检索增强生成"]
text = "RAG的全称是Retrieval-Augmented Generation"

# 检查每个关键词是否都在 text 中
all(kw.lower() in text.lower() for kw in keywords)
# → "rag" in "rag的全称是retrieval-augmented generation" → True
# → "检索增强生成" in "rag的全称是retrieval-augmented generation" → False
# → all([True, False]) → False
```

### Top-1 精度

```python
        # Top-1 精度：第一个结果中包含多少比例的关键词
        top1 = retrieved[0]["content"]
        matched = sum(1 for kw in keywords if kw.lower() in top1.lower())
        top1_scores.append(matched / len(keywords))
        # 例如：5 个关键词匹配了 3 个 → 3/5 = 0.6
```

### 汇总统计

```python
    hit_rate = hits / total if total > 0 else 0
    avg_top1 = sum(top1_scores) / len(top1_scores) if top1_scores else 0

    return {
        "label": label,
        "chunk_size": int(chroma_dir.split("_")[-1]),
        "retrieval_type": "MMR" if search_type == "mmr" else "Top-K",
        "top_k": k,
        "total": total,
        "hits": hits,
        "hit_rate": round(hit_rate, 4),
        "avg_top1": round(avg_top1, 4),
    }
```

### chroma_dir.split("_")[-1]

```python
chroma_dir = "chroma_db_eval_500"
chroma_dir.split("_")
# → ["chroma", "db", "eval", "500"]
chroma_dir.split("_")[-1]
# → "500"
int("500")
# → 500
```

---

## 4.4 print_report() — 打印报告

```python
def print_report(all_results: list, questions: list, docs_count: int):
    print()
    print("=" * 65)
    print("  RAG 检索效果对比报告")
    print("=" * 65)
    print(f"  测试集: {len(questions)} 个 QA 对")
    print(f"  文档:   {docs_count} 个源文件")
    print(f"  嵌入:   nomic-embed-text (768维)")
    print()
```

### 表格头部

```python
    print(f"  {'配置':<28} {'Hit Rate':<12} {'Top-1 Prec':<12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")
    # :<28  →  左对齐，占 28 个字符宽度
    # {'-'*28}  →  28 个 "-" 字符
```

### max() 找最佳

```python
    best = max(all_results, key=lambda r: r["hit_rate"])
    # max() 的 key 参数指定"按什么比较"
    # lambda r: r["hit_rate"]  →  取每个结果的 hit_rate 字段
    # 返回 hit_rate 最大的那个结果
```

### 按维度分析

```python
    for ct in ["MMR", "Top-K"]:
        ct_results = [r for r in all_results if r["retrieval_type"] == ct]
        if ct_results:
            avg = sum(r["hit_rate"] for r in ct_results) / len(ct_results)
            print(f"  {ct} 平均: {avg:.2%}")
    # 分别计算 MMR 和 Top-K 的平均命中率
```

---

## 4.5 main() — 主流程

```python
def main():
    print("🚀 RAG Benchmark")
    print("=" * 65)

    # 1. 加载数据
    print("\n[1/3] 加载数据和测试集...")
    docs = load_text_files(str(ROOT / "data"))

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"  测试问题: {len(questions)} 个")
```

### 2. 运行对比实验

```python
    for chunk_size in CHUNK_SIZES:
        chroma_dir = f"chroma_db_eval_{chunk_size}"

        # 清理并重建向量库（保证每个 chunk_size 是独立的实验）
        db_path = ROOT / chroma_dir
        if db_path.exists():
            shutil.rmtree(db_path)

        cfg = ChunkConfig(chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP)
        data_cfg = DataConfig(chroma_dir=chroma_dir)
        pipe = RAGPipeline(chunk_cfg=cfg, data_cfg=data_cfg)
        pipe.build_vectorstore(chunk_size=chunk_size)
```

### shutil.rmtree() 的作用

```python
if db_path.exists():
    shutil.rmtree(db_path)
    # 删除整个 chroma_db_eval_500/ 目录
    # 下次 build_vectorstore() 会重新创建
    # 确保实验是在干净的环境下运行的
```

### 对每种检索策略执行评估

```python
        for rc in RETRIEVAL_CONFIGS:
            label = f"cs={chunk_size}, {rc.search_type}(k={rc.k})"
            print(f"  ▶ {label}...", end=" ", flush=True)
            # flush=True  →  立即打印，不缓存（因为后面要等一会儿）
```

### time.time() 计时

```python
            start = time.time()
            result = evaluate(chroma_dir, questions, rc.search_type, rc.k, label)
            elapsed = time.time() - start
            # time.time() → 1970 年至今的秒数
            # 差值 → 代码执行了多少秒
            result["time_sec"] = round(elapsed, 2)
```

### 3. 输出报告

```python
    print("\n[3/3] 生成报告...")
    print_report(all_results, questions, len(docs))

    # 保存 JSON 报告
    RESULTS_DIR.mkdir(exist_ok=True)
    report = {
        "config": {
            "chunk_sizes": CHUNK_SIZES,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieval_configs": [
                {"type": rc.search_type, "k": rc.k}
                for rc in RETRIEVAL_CONFIGS
            ],
            "embedding_model": "nomic-embed-text",
        },
        "results": all_results,
        "best": max(all_results, key=lambda r: r["hit_rate"])["label"],
    }
    report_path = RESULTS_DIR / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已保存: {report_path}")
```

### ensure_ascii=False

```python
json.dump(report, f, ensure_ascii=False, indent=2)
# ensure_ascii=False  →  中文正常显示，不转成 \uXXXX
# indent=2  →  格式化成 2 空格缩进，便于阅读
# 不加的话：{"name": "\u5c0f\u660e"}
# 加了的话：{"name": "小明"}
```

---

# 第五部分：eval/questions.json — 测试数据集

```json
[
  {
    "id": "q1",
    "question": "AI 编程助手的工作流程包含哪些步骤？",
    "expected_keywords": ["拆解", "子任务", "规划", "编辑", "运行"]
  },
  ...
]
```

### 为什么用 expected_keywords 而不是 expected_answer？

用关键词判断比用完整答案判断更灵活。例如：

```
检索到的文档包含"拆解子任务、规划、编辑、运行"
用户问"AI编程助手的工作流程"

关键词匹配：["拆解", "子任务", "规划", "编辑", "运行"] → 全部命中！
完整答案匹配："第一步拆解子任务，第二步..." → 很难精确匹配
```

---

# 第六部分：scripts/ 目录 — 用户入口

## 6.1 scripts/build_db.py — 构建向量库

```python
def main():
    parser = argparse.ArgumentParser(description="构建向量库")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="文本片段大小（默认 500）")
    parser.add_argument("--overlap", type=int, default=50,
                        help="片段重叠字符数（默认 50）")
    args = parser.parse_args()
```

### argparse — 命令行参数解析

```python
parser = argparse.ArgumentParser(description="构建向量库")
# 创建一个解析器

parser.add_argument("--chunk-size", type=int, default=500)
# 添加一个可选参数 --chunk-size，类型是 int，默认 500

args = parser.parse_args()
# 解析用户输入的命令行
# python build_db.py --chunk-size 1000 --overlap 30
# → args.chunk_size = 1000, args.overlap = 30
```

```python
    cfg = ChunkConfig(chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    pipe = RAGPipeline(chunk_cfg=cfg)
    count = pipe.build_vectorstore(chunk_size=args.chunk_size)
    print(f"\n✅ 构建完成: {count} 个文本片段已向量化")
```

---

## 6.2 scripts/query.py — 交互式问答

```python
def main():
    pipe = RAGPipeline()

    # 尝试加载已有向量库，没有则构建
    loaded = pipe.load_vectorstore()
    if not loaded:
        print("\n[info] 首次运行，构建向量库...")
        pipe.build_vectorstore(chunk_size=500)

    info = pipe.get_vectorstore_info()
```

### 主循环

```python
    search_type = "similarity"
    k = 3

    while True:
        try:
            user_input = input("问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # EOFError → Linux 按 Ctrl+D
        # KeyboardInterrupt → 按 Ctrl+C
```

### 输入处理

```python
        if user_input.lower() == "q":
            break  # 退出
        if user_input == "/mmr":
            search_type = "mmr"
            print("  → 切换到 MMR 检索")
            continue  # 跳到下一次循环
        if user_input == "/sim":
            search_type = "similarity"
            print("  → 切换到 Top-K 检索")
            continue
        if not user_input:
            continue  # 空输入，跳过
```

### 执行问答

```python
        result = pipe.query(user_input, search_type=search_type, k=k)
        print(f"\n回答: {result['answer']}\n")
        if "sources" in result and result["sources"]:
            print("来源:")
            for s in result["sources"]:
                print(f"  - {s['source']}")
        print()
```

---

## 6.3 scripts/app.py — Streamlit Web 界面

### @st.cache_resource

```python
@st.cache_resource
def init_pipeline():
    pipe = RAGPipeline()
    loaded = pipe.load_vectorstore()
    if not loaded:
        with st.spinner("首次运行，构建向量库中..."):
            pipe.build_vectorstore(chunk_size=500)
    return pipe
```

`@st.cache_resource` 是一个**装饰器**，作用是把函数的结果缓存起来。

```python
# 没有缓存：每次页面刷新都重新构建向量库（慢！）
pipe = init_pipeline()

# 有缓存：只有第一次调用时执行函数，之后直接返回缓存结果
@st.cache_resource
def init_pipeline():
    ...
```

### st.set_page_config

```python
st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="centered",
)
```

设置页面标题、图标和布局。

### 侧边栏

```python
with st.sidebar:
    st.header("⚙️ 检索配置")
    search_type = st.radio(
        "检索策略",
        options=["similarity", "mmr"],
        format_func=lambda x: "Top-K（相似度）" if x == "similarity" else "MMR（多样性）",
        index=0,
    )
    k = st.slider("返回文档数 (k)", min_value=1, max_value=5, value=3)
```

| Streamlit 组件 | 作用 |
|----------------|------|
| `st.header()` | 显示标题 |
| `st.radio()` | 单选按钮 |
| `st.slider()` | 滑块选择 |
| `st.divider()` | 分隔线 |
| `st.caption()` | 小字说明 |

### 会话状态

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

`st.session_state` 是 Streamlit 的会话状态。每次页面交互时，脚本会从头执行，但 `st.session_state` 会保留之前的值，保证聊天记录不丢失。

### 聊天消息显示

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("来源文档"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']}")
```

### 用户输入处理

```python
if prompt := st.chat_input("输入你的问题..."):
    # 这是一个"海象运算符" :=
    # 等价于：
    # prompt = st.chat_input("输入你的问题...")
    # if prompt:
```

### 海象运算符 := (Walrus Operator)

```python
# Python 3.8+ 新增的语法
# 同时做两件事：赋值 + 判断

if prompt := st.chat_input("输入..."):
    # 1. 把 st.chat_input() 的结果赋给 prompt
    # 2. 判断 prompt 是否为真（非空）

# 等价于：
prompt = st.chat_input("输入...")
if prompt:
```

### 消息存储

```python
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
    })
```

---

# 第七部分：知识点总结

## 7.1 关键 Python 语法点

| 语法 | 位置 | 作用 |
|------|------|------|
| `@dataclass` | config.py | 自动生成类的 `__init__`、`__repr__` |
| `field(default_factory=lambda: ...)` | config.py | 为 dataclass 的 list/dict 创建独立的默认值 |
| `Optional[Type] = None` | pipeline.py | 类型提示：可以是该类型或 None |
| `try/except` | pipeline.py | 捕获异常，防止程序崩溃 |
| `sys.path.insert(0, str(ROOT))` | benchmark.py | 将项目根目录加入模块搜索路径 |
| `format()` | pipeline.py | 字符串模板替换 |
| `map()` | benchmark.py | 找到最大值的元素 |
| `all()` | benchmark.py | 检查所有条件是否同时满足 |
| `lambda` | 多处 | 匿名函数，一行定义简单函数 |
| `:=` 海象运算符 | app.py | 在表达式内部赋值 |
| `shutil.rmtree()` | benchmark.py | 递归删除整个目录树 |

## 7.2 项目架构设计模式

### 分层架构

```
scripts/  (用户界面层)
    ↓ 调用
rag/      (核心业务逻辑层)
    ↓ 依赖
config.py (配置管理层)
```

### 依赖注入

```python
class RAGPipeline:
    def __init__(self, embedding_cfg=None, llm_cfg=None, ...):
        # 不直接创建配置对象，而是由调用者传入
        # 好处：灵活替换，方便测试
```

### 懒加载

```python
self._llm: Optional[OllamaLLM] = None

def query(...):
    if not self._llm:
        self._llm = OllamaLLM(...)
    # 只有真正需要 LLM 时才创建，不是一开始就创建
```

### 链式调用 vs 常规调用

```python
# 本项目写法：每个方法独立调用
pipe = RAGPipeline()
pipe.build_vectorstore()
pipe.query("什么是RAG？")

# 对比 converter.py 的链式调用：
# c.load().fill_missing(...).select_columns(...).export(...)
# 链式调用的好处：一气呵成
# 独立调用的好处：中间可以根据需要调整参数
```

## 7.3 项目中的数据流

### 构建向量库时

```
data/*.txt  →  loader.py  →  [{"source":"...", "content":"..."}]
                                 ↓
                  RecursiveCharacterTextSplitter
                                 ↓
                  [Doc("片段1"), Doc("片段2"), ...]
                                 ↓
                  nomic-embed-text → 768维向量
                                 ↓
                  Chroma 向量数据库（chroma_db/）
```

### 回答问题时

```
用户提问  →  768维向量  →  Chroma 检索  →  Top-3 文档片段
                                               ↓
                       [来源A: 内容] + [来源B: 内容] + [来源C: 内容]
                                               ↓
                       合并为 context
                                               ↓
                       PromptTemplate.format(context=..., question=...)
                                               ↓
                       qwen:4b 生成回答
                                               ↓
                       {"answer": "RAG是...", "sources": [...]}
```

## 7.4 关键技术概念速查

| 概念 | 一句话解释 |
|------|-----------|
| RAG | 检索 + 生成，先找资料后回答，减少模型胡说八道 |
| 嵌入 (Embedding) | 把文字变成一串数字，语义相近的数字也相近 |
| Chunk | 把长文档切成小段，方便精确匹配 |
| Top-K | 找最像的 K 个结果，简单直接 |
| MMR | 既相关又多样的 K 个结果，减少重复 |
| Chroma | 快速存/查向量的数据库 |
| Ollama | 本地跑大模型，不用联网 |
| Hit Rate | 命中率 = 找到正确答案的比例 |
| Hallucination | 模型不记得就瞎编，RAG 是对抗幻觉的常用方法 |

---
