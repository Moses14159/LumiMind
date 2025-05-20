# Langchain 与 Streamlit 驱动的心理健康与沟通辅导助手 - Cursor Prompts

## 一、项目初始化与核心架构

---

**Prompt 1: 创建项目目录结构和基础文件**

Action: 为基于 Langchain 和 Streamlit 的双模块应用（心理健康咨询、沟通辅导）创建详细的项目目录结构。
应包含以下主要目录和文件：
- `app.py` (Streamlit主入口)
- `core/`: 核心逻辑 (`chains/`, `rag/`, `prompts/`, `utils/`)
- `modules/`: Streamlit 页面模块 (`mental_health_page.py`, `communication_page.py`)
- `knowledge_base/`: RAG 文档存储 (`mental_health_docs/`, `communication_docs/`)
- `static/`: 静态资源 (CSS, images)
- `tests/`: 测试文件
- `config/`: 配置文件 (`settings.py`)
- `scripts/`: 辅助脚本 (如数据导入)
- 根目录文件: `requirements.txt`, `Dockerfile`, `.env.example`
确保在必要的子目录中创建 `__init__.py` 文件。

---

**Prompt 2: 实现 Pydantic 应用配置 (`config/settings.py`)**

Action: 在 `config/settings.py` 文件中，使用 Pydantic 的 `BaseSettings` 来管理应用配置。
确保配置可以从环境变量加载。
需要包含的配置项（但不限于）：
- 各大 LLM Provider 的 API Keys (OpenAI, Gemini, DeepSeek, SiliconFlow, InternLM)
- 讯飞星火的认证信息 (APPID, API_KEY, API_SECRET)
- Ollama 的基础 URL (`OLLAMA_BASE_URL`) 和默认模型 (`OLLAMA_DEFAULT_MODEL`)
- 默认 LLM Provider (`DEFAULT_LLM_PROVIDER`)
- 向量数据库路径及其他 RAG 相关配置 (如文档路径, 嵌入模型名称)
- 使用 `SettingsConfigDict(env_file=".env", ...)` 从 .env 文件加载。

---

**Prompt 3: 创建 `.env.example` 文件**

Action: 创建一个 `.env.example` 文件。
该文件应包含 `config/settings.py` 中定义的所有环境变量的占位符。

---

**Prompt 4: 实现 LLM Factory (`core/utils/llm_factory.py`)**

Action: 在 `core/utils/llm_factory.py` 中实现一个 `get_llm` 函数。
该函数应：
1. 接收可选的 `provider` 和 `model_name` 参数。若 `provider` 未指定，则从 `config.settings.DEFAULT_LLM_PROVIDER` 获取。
2. 根据 `provider` 的值，使用相应的 Langchain LLM 包装器实例化并返回 LLM 客户端。支持的 provider 包括："openai", "gemini", "deepseek", "siliconflow", "internlm", "spark", "ollama"。
3. 从 `config.settings` 中获取各 provider 所需的 API Keys 或其他认证信息及 URL。
4. Ollama 配置应使用 `settings.OLLAMA_BASE_URL`，模型选择优先使用传入的 `model_name`，其次是 `settings.OLLAMA_DEFAULT_MODEL`。
5. 若 API Key 或必要配置缺失，抛出 `ValueError`。
6. 若某个 provider 的 Langchain 集成尚未实现，则抛出 `NotImplementedError` (为这些 provider 预留分支)。
7. 添加一个辅助函数 `get_default_llm()`，用于获取默认配置的 LLM 实例。
确保导入所有必要的 Langchain 模块。

---

**Prompt 5: 初始化 Streamlit 多页面应用与模块化 UI (`app.py`)**

Action: 在 `app.py` 中使用 Streamlit 创建一个多页面应用。
1. 在侧边栏提供清晰的导航，允许用户在"心理健康咨询"和"沟通回应辅导"两大模块间切换。
2. 为每个模块（如 `modules/mental_health_page.py`, `modules/communication_page.py`）创建独立的 Python 文件并在 `app.py` 中进行路由。
3. `app.py` 负责全局状态管理，例如当前选择的 LLM Provider (存储在 `st.session_state`)。
4. 在 UI 中允许用户（或管理员）从已配置的 LLM Provider 中选择要使用的 Provider。
5. **关键视觉区分**：确保两个模块在视觉风格和交互模式上有明显不同。心理健康模块采用平静舒缓设计；沟通辅导模块侧重信息密度和选项呈现。使用不同的 CSS 文件或 Streamlit 主题化功能实现。

---

**Prompt 6: (可选 - FastAPI) 设计后端 API 骨架**

Action: 如果计划使用独立的 FastAPI 后端，请设计其核心路由和 Pydantic 数据模型。
主要路由应包括：
- `/chat/mental_health`: 处理心理健康咨询请求。
- `/chat/communication_coach`: 处理沟通辅导请求。
- `/rag/query`: 处理 RAG 查询。
- `/config/llm_providers`: 返回支持和配置的 LLM providers 列表。
初步设计用户会话管理和状态追踪机制。

---

## 二、心理健康咨询模块 (深度)

---

**Prompt 7: 定义共情对话 Chain (`core/chains/mental_health_chain.py`)**

Action: 在 `core/chains/mental_health_chain.py` 中，创建一个名为 `EmpatheticConversationChain` 的 Langchain Chain。
1. Chain 初始化时接收一个 LLM 实例 (通过 `llm_factory.get_llm()` 获取)。
2. 设计一个复杂的 PromptTemplate (可在 `core/prompts/mental_health_prompts.py` 中定义和管理)，明确指示 LLM 扮演富有同情心、耐心、非评判性的心理支持伙伴。提示需包含角色定义、积极倾听技巧运用、鼓励用户安全分享等元素，并借鉴 EmoLLM 和 CPsyCounD 数据集的设计理念。
3. 集成 `ConversationBufferWindowMemory` 以支持多轮对话。

---

**Prompt 8: 实现心理健康 RAG (`core/rag/`)**

Action: 在 `core/rag/` 目录下实现 RAG 功能，用于心理健康模块。
1.  **`vectorstore_manager.py`**: 实现一个 `VectorstoreManager` 类。
    * 管理向量数据库的初始化 (支持 ChromaDB 或 FAISS)。
    * 从 `knowledge_base/mental_health_docs/` 加载文档。
    * 使用 Sentence Transformers 或所选 LLM Provider 的嵌入API 进行文本嵌入。
    * 提供检索功能，并能从持久化存储中加载/保存向量库。
    * 包含为特定知识库（如 "mental_health_kb"）创建或加载 retriever 的方法。
2.  **`retriever.py` (或集成入 `EmpatheticConversationChain`)**: 创建或配置 Langchain `Retriever` 对象。
3.  **修改 `EmpatheticConversationChain`**:
    * 使其能够结合 RAG。
    * 当用户提问涉及特定心理健康知识时，能判断是否需要 RAG，从向量库检索信息，并将上下文融入回答。
4.  **知识库内容**: 知识库应包含心理健康教育材料、CBT 基础、应对策略、危机干预信息（用于判断和引导，非直接提供）。在 `scripts/ingest_data.py` 中创建数据导入脚本。

---

**Prompt 9: 定义 CBT 技术引导 Chain (`core/chains/cbt_exercise_chain.py`)**

Action: 在 `core/chains/cbt_exercise_chain.py` 中创建 `CBTExerciseChain`。
1. Chain 初始化时接收 LLM 实例。
2. **目标**：引导用户完成简化的 CBT 练习（如思维记录表、挑战消极思维）。
3. **交互流程**：设计多步骤对话流程，逐步引导用户描述情境、识别思维、评估证据、提出替代思维。
4. **Prompt Engineering**：为每个步骤设计清晰、温和、有效的提示 (可在 `core/prompts/cbt_prompts.py` 中管理)。
5. **结构化输出**：考虑让 LLM 以结构化方式 (如 JSON，使用 Pydantic 模型定义) 返回练习的关键点。

---

**Prompt 10: 实现危机检测与强制升级路径 (`core/utils/crisis_detection.py` 及集成)**

Action: 在心理健康模块的对话处理流程中，集成危机检测机制。
1.  在 `core/utils/crisis_detection.py` 中实现危机检测逻辑。
    * **方法**：综合使用关键词检测 (维护敏感词列表，例如在 `core/utils/crisis_keywords.txt`)、情感分析 (可选，使用第三方库或 API) 和 LLM 判断 (作为辅助，使用特定 prompt)。
2.  **强制升级逻辑** (在 `modules/mental_health_page.py` 的对话处理中调用)：
    * 如果检测到高风险危机信号，立即停止当前对话流程。
    * 清晰、直接地向用户显示预设的危机干预信息（本地化紧急求助热线、专业机构联系方式）。**AI 不应提供危机干预**。
    * 界面应有醒目提示。

---

## 三、沟通回应辅导模块 (深度)

---

**Prompt 11: 定义"不知如何回应"辅助与元认知循环 Chain (`core/chains/communication_coach_chain.py`)**

Action: 在 `core/chains/communication_coach_chain.py` 中创建 `ResponseCoachChain`。
1. Chain 初始化时接收 LLM 实例。
2. **核心功能**：当用户输入沟通困境时：
    * LLM 分析情境（对方、目标、氛围）。
    * 生成3-5个不同回应选项。
    * 对每个选项解释其潜在效果、语气、可能反应及适用性。
    * 若信息不足，LLM 生成引导性问题以帮助用户思考沟通目的和真实意图 (元认知)。
    * LLM 能在信息不足时承认不确定性并引导用户。
3. **Prompt Engineering** (可在 `core/prompts/communication_prompts.py` 中管理)：强调培养用户的"元认知循环"。
4. **结构化输出**：使用 Pydantic 模型定义输出，包含回应列表 (每个回应含 `text` 和 `explanation` 字段) 及元认知提示。

---

**Prompt 12: 实现沟通辅导 RAG (社交技能与礼仪指导)**

Action: 为沟通辅导模块配置独立的 RAG 系统。
1.  **知识库内容** (`knowledge_base/communication_docs/`): 包含社交规范、沟通礼仪、非暴力沟通、跨文化沟通、说服技巧、冲突管理等。在 `scripts/ingest_data.py` 中添加对应的数据导入逻辑。
2.  **`VectorstoreManager` 复用**: 使用 `core/rag/vectorstore_manager.py` 中的 `VectorstoreManager` 为此模块创建和管理独立的向量数据库实例 (例如，使用不同的 `collection_name` like "communication_kb")。
3.  **"语用假肢"功能** (在 `ResponseCoachChain` 或新 Chain 中集成)：
    * 当用户询问特定表达是否礼貌或如何委婉拒绝时，RAG 检索相关规则或礼仪。
    * LLM 结合检索到的信息解释为何某种表达更合适，突出语用考量。

---

**Prompt 13: 实现困难对话演练功能 (`modules/communication_page.py` 和 `core/chains/role_play_chain.py`)**

Action: 在 Streamlit 界面 (`modules/communication_page.py`) 实现困难对话演练。
1.  **`core/chains/role_play_chain.py`**: 创建 `RolePlayChain`。
    * Chain 初始化时接收 LLM 实例。
    * 管理对话流程和 LLM 的角色扮演行为，基于用户选择的场景。
2.  **Streamlit 界面 (`modules/communication_page.py`)**:
    * 允许用户选择预设场景（如"要求加薪"）或自定义场景。
    * LLM 扮演对话另一方，用户输入回应。
    * 演练结束后，LLM (`RolePlayChain`) 提供总结性分析（优点、改进点、策略建议）。
    * (可选) LLM 在每轮对话后给出简短即时反馈。

---

## 四、UI/UX、数据管理与通用功能

---

**Prompt 14: 设计动态伦理协议与模式切换逻辑**

Action: 设计系统逻辑以支持动态伦理协议。
1.  **情境感知**：系统需能感知用户当前模块（心理健康 vs. 沟通辅导）。
2.  **跨模块危机检测**：即使在沟通辅导模块，若用户输入透露心理危机，应能识别并触发心理健康模块的危机干预流程。在 `app.py` 或各页面模块中实现此检查逻辑，调用 `core/utils/crisis_detection.py`。
3.  **用户同意管理**：清晰管理用户对数据使用的同意，尤其在模块间可能共享匿名化洞察时，需额外明确同意。

---

**Prompt 15: 强化结构化输出与 Pydantic 集成**

Action: 对于需要 LLM 生成结构化数据的场景 (如 `ResponseCoachChain` 的多个回应选项、`CBTExerciseChain` 的练习总结)，全面使用 Langchain 的 Output Parsers，并结合 Pydantic 定义输出的数据模型。
确保所有相关 Chain 都配置了合适的 Output Parser 以保证输出格式一致性。

---

**Prompt 16: 应用高级提示工程技巧**

Action: 在各 Chain 的 Prompt 设计中，系统性应用高级提示工程技巧。
1.  **思维链 (CoT)**：对复杂分析任务（如沟通情境分析、CBT 思维挑战），引导 LLM 先分步思考再输出。
2.  **角色扮演细化**：为不同模块和场景下的 LLM 定义更细致的角色、行为指南和禁止行为。
3.  **CAF (Critical Analysis Filter) 系统探索 (高级)**：探索实现一个简易 CAF 系统。例如，一个 LLM 生成回应，另一个 LLM (或同一 LLM 使用不同提示) 对其进行批判性评估和修正。可以先设计一个简单的评估 prompt。

---

**Prompt 17: 实现用户反馈与"人在环" (HITL) 数据收集**

Action: 在 Streamlit 界面中，为重要的 LLM 回应提供明确的用户反馈机制。
1.  **UI 元素**：在 `mental_health_page.py` 和 `communication_page.py` 中，为 LLM 的关键回复添加反馈按钮 (如点赞/点踩、评分、标签选择、文本评论)。
2.  **数据存储**：将用户反馈（匿名化处理）与对应对话片段一同存储（考虑简单的 CSV、JSON Lines 文件或数据库）。
3.  **HITL 标注界面 (初步设想)**：规划未来如何开发一个简单的内部工具或流程，允许领域专家审查和标注数据，用于模型微调。

---

## 五、评估、迭代与部署

---

**Prompt 18: 搭建 LLM 性能评估框架**

Action: 设计一个初步的 LLM 评估框架的规划和相关脚本/工具的准备。
1.  **指标定义**:
    * 心理健康模块：共情性 (人工评分量表参考 PsycoLLM)、CBT 原则依从性 (专家审查)、安全性 (危机识别准确率)。
    * 沟通辅导模块：回应相关性/恰当性 (人工判断)、解释清晰度/帮助性 (用户评分)。
    * 通用：BLEU/ROUGE (辅助)、困惑度、连贯性、RAG 忠实度 (Faithfulness) 和答案相关性 (Answer Relevance)。
2.  **自动化评估脚本** (`scripts/evaluate_rag.py`): 编写脚本，使用如 `ragas` 或自定义脚本对 RAG 系统进行初步自动化评估 (如 context_precision, answer_relevancy)。
3.  **"LLM 作为裁判"探索**: 准备使用强大 LLM (如 GPT-4, Claude 3 Opus) 通过特定 Prompt 评估生成回应质量的实验方案。

---

**Prompt 19: 制定持续学习与模型更新策略**

Action: 制定模型持续学习和更新的初步策略（文档化或作为规划）。
1.  **数据收集**：明确收集用户反馈、HITL 标注数据的流程和格式。
2.  **微调频率**：规划合适的模型再微调周期 (尤其针对本地 Ollama 微调模型)。
3.  **微调技术**：研究并优先考虑参数高效微调方法 (LoRA/QLoRA)。探索使用直接偏好优化 (DPO)。
4.  **版本控制**：建立模型、数据集和 Prompt 的版本控制机制。

---

**Prompt 20: Docker 化与部署准备**

Action: 为应用创建 `Dockerfile` 和 `docker-compose.yml`。
1.  **`Dockerfile`**:
    * 包含所有 Python 依赖项 (从 `requirements.txt` 安装)。
    * 复制所有必要的应用代码、配置文件和知识库数据。
    * 设置正确的启动命令 (如 `streamlit run app.py`)。
    * 考虑为本地 LLM (如 Ollama) 和应用服务分别创建 Docker 镜像（如果 Ollama 不在同一容器运行）。
2.  **`docker-compose.yml` (用于本地开发和测试)**:
    * 编排应用服务。
    * (可选) 包含 Ollama 服务 (如果本地使用 Ollama 且希望通过 compose 管理)。
    * 配置必要的环境变量和卷挂载。