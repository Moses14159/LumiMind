# LumiMind 环境变量配置指南

本文档帮助您创建 `.env` 配置文件，以设置 LumiMind 应用程序所需的环境变量。

## 创建 .env 文件

在项目根目录下创建一个名为 `.env` 的文件，并按照下面的模板进行配置：

```
# 一般设置
APP_NAME="LumiMind"
APP_VERSION="0.1.0"
DEBUG=False

# 默认 LLM 提供商 (可选: "openai", "gemini", "ollama" 等)
DEFAULT_LLM_PROVIDER="openai"

# OpenAI 配置
OPENAI_API_KEY=""
OPENAI_DEFAULT_MODEL="gpt-4-turbo"

# Google (Gemini) 配置
GEMINI_API_KEY=""
GEMINI_DEFAULT_MODEL="gemini-pro"

# DeepSeek 配置
DEEPSEEK_API_KEY=""
DEEPSEEK_DEFAULT_MODEL="deepseek-chat"

# SiliconFlow 配置
SILICONFLOW_API_KEY=""
SILICONFLOW_DEFAULT_MODEL="sf-chat"

# InternLM 配置
INTERNLM_API_KEY=""
INTERNLM_DEFAULT_MODEL="internlm-chat-20b"

# 讯飞星火配置
IFLYTEK_SPARK_APPID=""
IFLYTEK_SPARK_API_KEY=""
IFLYTEK_SPARK_API_SECRET=""
IFLYTEK_SPARK_DEFAULT_MODEL="spark-3.5"

# Ollama 配置（本地模型）
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_DEFAULT_MODEL="llama3"

# 向量数据库设置
VECTORDB_TYPE="chroma"
VECTORDB_PATH="./vectordb"

# RAG 设置
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# 心理健康知识库
MENTAL_HEALTH_DOCS_PATH="./app/knowledge_base/mental_health_docs"
MENTAL_HEALTH_KB_NAME="mental_health_kb"

# 沟通知识库
COMMUNICATION_DOCS_PATH="./app/knowledge_base/communication_docs"
COMMUNICATION_KB_NAME="communication_kb"

# 危机检测设置
CRISIS_KEYWORDS_PATH="./app/core/utils/crisis_keywords.txt"
CRISIS_DETECTION_THRESHOLD=0.7
```

## 配置说明

### 一般设置
- `APP_NAME`：应用程序名称，可保持默认
- `APP_VERSION`：应用程序版本号，可保持默认
- `DEBUG`：调试模式开关，设置为 `True` 或 `False`

### LLM 提供商设置
- `DEFAULT_LLM_PROVIDER`：默认使用的 LLM 提供商，有效值包括：
  - `"openai"`：使用 OpenAI 的 GPT 模型
  - `"gemini"`：使用 Google 的 Gemini 模型
  - `"deepseek"`：使用 DeepSeek 模型
  - `"siliconflow"`：使用 SiliconFlow 模型
  - `"internlm"`：使用书生·浦语模型
  - `"spark"`：使用讯飞星火模型
  - `"ollama"`：使用本地 Ollama 模型（推荐初次使用选择此项）

### OpenAI 配置
- `OPENAI_API_KEY`：您的 OpenAI API 密钥
- `OPENAI_DEFAULT_MODEL`：默认使用的 OpenAI 模型，如 `"gpt-4-turbo"` 或 `"gpt-3.5-turbo"`

### Google Gemini 配置
- `GEMINI_API_KEY`：您的 Google AI Studio API 密钥
- `GEMINI_DEFAULT_MODEL`：默认使用的 Gemini 模型，如 `"gemini-pro"`

### Ollama 配置（本地模型）
- `OLLAMA_BASE_URL`：Ollama 服务的 URL，默认为 `"http://localhost:11434"`
- `OLLAMA_DEFAULT_MODEL`：默认使用的 Ollama 模型，如 `"llama3"` 或 `"mistral"`

### 其他 LLM 提供商
您可以根据需要配置其他 LLM 提供商的 API 密钥和模型设置，包括：
- DeepSeek
- SiliconFlow
- InternLM (书生·浦语)
- 讯飞星火

### 向量数据库和 RAG 设置
这些设置控制应用程序的检索增强生成 (RAG) 功能：
- `VECTORDB_TYPE`：向量数据库类型，支持 `"chroma"` 或 `"faiss"`
- `VECTORDB_PATH`：向量数据库存储路径
- `EMBEDDING_MODEL`：用于文本嵌入的模型
- `MENTAL_HEALTH_DOCS_PATH`：心理健康知识库文档路径
- `MENTAL_HEALTH_KB_NAME`：心理健康知识库名称
- `COMMUNICATION_DOCS_PATH`：沟通知识库文档路径
- `COMMUNICATION_KB_NAME`：沟通知识库名称

### 危机检测设置
- `CRISIS_KEYWORDS_PATH`：危机关键词文件路径
- `CRISIS_DETECTION_THRESHOLD`：危机检测阈值，范围 0-1，默认为 0.7

## 注意事项

1. API 密钥安全：请勿分享您的 API 密钥，确保 `.env` 文件在版本控制系统中被忽略
2. 最小配置：如果只想使用本地 Ollama 模型，只需设置 `DEFAULT_LLM_PROVIDER="ollama"`
3. 多提供商支持：您可以同时配置多个 LLM 提供商，在应用程序运行时进行切换
4. 文件路径：确保知识库路径和危机关键词文件路径正确，最好使用绝对路径 