1.  **项目结构与配置文件：**
    ```
    为基于 Langchain 和 Streamlit 的双模块（心理健康咨询、沟通辅导）应用创建一个详细的项目目录结构。应包含：
    - `app.py`: Streamlit 应用主入口。
    - `core/`: 核心逻辑，如 LLM chains, RAG 实现, 业务逻辑。
        - `chains/`: Langchain chains 定义 (e.g., `mental_health_chain.py`, `communication_coach_chain.py`).
        - `rag/`: RAG 相关组件 (e.g., `vectorstore_manager.py`, `retriever.py`).
        - `prompts/`: Prompt templates 存储。
        - `utils/`: 通用工具函数 (e.g., `llm_factory.py`).
    - `modules/`: Streamlit 各页面模块 (e.g., `mental_health_page.py`, `communication_page.py`).
    - `knowledge_base/`: 存储 RAG 使用的原始文档 (e.g., `mental_health_docs/`, `communication_docs/`).
    - `static/`: 静态资源 (CSS, images).
    - `tests/`: 单元测试和集成测试。
    - `config/`: 配置文件 (e.g., `settings.py` for API keys, model names, DB configs).
    - `scripts/`: 辅助脚本 (e.g., data ingestion for RAG).
    - `requirements.txt`, `Dockerfile`, `.env.example`.

    创建一个 `config/settings.py` 文件，使用 Pydantic 管理应用配置。该文件应能从环境变量加载以下信息：
    - `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `SILICONFLOW_API_KEY`, `INTERNLM_API_KEY` (或相关认证信息)
    - `IFLYTEK_SPARK_APPID`, `IFLYTEK_SPARK_API_KEY`, `IFLYTEK_SPARK_API_SECRET`
    - `OLLAMA_BASE_URL` (默认为 `http://localhost:11434`)
    - `DEFAULT_LLM_PROVIDER` (例如 'openai', 'gemini', 'ollama')
    - `OLLAMA_DEFAULT_MODEL` (用于 Ollama 的默认模型名称)
    - 向量数据库路径等其他配置。

    创建一个 `.env.example` 文件，包含上述所有 API Key 和配置的占位符。
    ```
2.  **Streamlit 多页面应用与模块化 UI：**
    ```
    使用 Streamlit 创建一个多页面应用。在侧边栏提供清晰的导航，允许用户在“心理健康咨询”和“沟通回应辅导”两大模块间切换。
    - 为每个模块创建一个独立的 Python 文件（例如 `modules/mental_health_page.py`）。
    - 主 `app.py` 负责路由和全局状态管理 (如选择的 LLM Provider)。
    - **关键：** 确保两个模块在视觉风格和交互模式上有明显区分。心理健康模块应采用更平静、舒缓的设计；沟通辅导模块可更侧重信息密度和选项呈现。
    - 在 UI 中允许用户（或管理员）选择要使用的 LLM Provider (如果配置了多个)。
    ```
3.  **Langchain 基础配置与 LLM 接口 (`llm_factory.py`)：**
    ```python
    # core/utils/llm_factory.py
    """
    LLM Factory to get LLM instances based on configuration.
    """
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatDeepseek # 假设 DeepSeek 有 Langchain 社区支持
    # 导入其他 LLM 的 Langchain 包装器，例如 SiliconFlow, InternLM, iFlyTek Spark
    # from langchain_community.chat_models.sparkllm import ChatSparkLLM # 示例，实际导入可能不同
    from langchain_community.llms import Ollama
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import BaseLLM

    from config.settings import settings # Pydantic settings

    def get_llm(provider: str = None, model_name: str = None) -> BaseChatModel | BaseLLM:
        """
        Factory function to get an LLM instance based on the provider.

        Args:
            provider (str, optional): The LLM provider to use. 
                                      Defaults to settings.DEFAULT_LLM_PROVIDER.
            model_name (str, optional): Specific model name for the provider.
                                        For Ollama, defaults to settings.OLLAMA_DEFAULT_MODEL.
                                        For others, often set by the provider's default or their API.

        Returns:
            Union[BaseChatModel, BaseLLM]: An instance of the LLM.

        Raises:
            ValueError: If the provider is not supported or API key is missing.
        """
        selected_provider = provider or settings.DEFAULT_LLM_PROVIDER

        if selected_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is not set in environment variables.")
            # 实际模型名称可以从 settings 或参数传入
            return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=model_name or "gpt-3.5-turbo")
        
        elif selected_provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("Gemini API key is not set in environment variables.")
            return ChatGoogleGenerativeAI(google_api_key=settings.GEMINI_API_KEY, model=model_name or "gemini-pro")

        elif selected_provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API key is not set in environment variables.")
            # 假设 ChatDeepseek 的用法，具体参数需查阅其 Langchain 集成文档
            return ChatDeepseek(api_key=settings.DEEPSEEK_API_KEY, model=model_name or "deepseek-chat") # model_name 示例

        elif selected_provider == "siliconflow":
            if not settings.SILICONFLOW_API_KEY:
                raise ValueError("SiliconFlow API key is not set.")
            # 示例：需要替换为 SiliconFlow 实际的 Langchain 包装器和初始化方式
            # return SomeSiliconFlowChatModel(api_key=settings.SILICONFLOW_API_KEY, model=model_name or "default-silicon-model")
            raise NotImplementedError("SiliconFlow LLM integration not yet implemented.")

        elif selected_provider == "internlm":
            if not settings.INTERNLM_API_KEY: # 或其他认证方式
                raise ValueError("InternLM API key/auth is not set.")
            # 示例：需要替换为 InternLM 实际的 Langchain 包装器和初始化方式
            # return SomeInternLMChatModel(api_key=settings.INTERNLM_API_KEY, model=model_name or "internlm2-chat-7b")
            raise NotImplementedError("InternLM integration not yet implemented.")

        elif selected_provider == "spark": # iFlyTek Spark
            if not (settings.IFLYTEK_SPARK_APPID and settings.IFLYTEK_SPARK_API_KEY and settings.IFLYTEK_SPARK_API_SECRET):
                raise ValueError("iFlyTek Spark API credentials are not fully set.")
            # 示例：假设存在 ChatSparkLLM 包装器
            # return ChatSparkLLM(
            #     spark_app_id=settings.IFLYTEK_SPARK_APPID,
            #     spark_api_key=settings.IFLYTEK_SPARK_API_KEY,
            #     spark_api_secret=settings.IFLYTEK_SPARK_API_SECRET,
            #     model=model_name or "generalv3.5" # 示例模型
            # )
            raise NotImplementedError("iFlyTek Spark integration not yet implemented.")

        elif selected_provider == "ollama":
            if not settings.OLLAMA_BASE_URL:
                raise ValueError("Ollama base URL is not set.")
            return Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_name or settings.OLLAMA_DEFAULT_MODEL or "llama2" # 提供一个备用默认值
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {selected_provider}")

    # 可以在这里添加一个函数来获取默认的 LLM 实例
    def get_default_llm() -> BaseChatModel | BaseLLM:
        return get_llm()

    # 示例：
    # if __name__ == '__main__':
    #     # 需要设置 .env 文件或环境变量
    #     try:
    #         # ollama_llm = get_llm(provider="ollama", model_name="mistral")
    #         # print(ollama_llm.invoke("你好，你是谁？"))
            
    #         openai_llm = get_llm(provider="openai")
    #         print(openai_llm.invoke("Hello, who are you?"))

    #     except ValueError as e:
    #         print(f"Error: {e}")
    #     except NotImplementedError as e:
    #         print(f"Notice: {e}")
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    ```
    **Cursor 提示：**
    ```
    在 `core/utils/llm_factory.py` 中实现一个 `get_llm` 函数。
    该函数应：
    1. 接收可选的 `provider` 和 `model_name` 参数。如果 `provider` 未指定，则从 `config.settings.DEFAULT_LLM_PROVIDER` 获取。
    2. 根据 `provider` 的值，实例化并返回相应的 Langchain LLM 包装器：
        - "openai": 使用 `ChatOpenAI`，API Key 来自 `settings.OPENAI_API_KEY`。
        - "gemini": 使用 `ChatGoogleGenerativeAI`，API Key 来自 `settings.GEMINI_API_KEY`。
        - "deepseek": 使用 `ChatDeepseek` (或其等效的 Langchain 包装器)，API Key 来自 `settings.DEEPSEEK_API_KEY`。
        - "siliconflow": (需要查找或实现其 Langchain 包装器)，API Key 来自 `settings.SILICONFLOW_API_KEY`。
        - "internlm": (需要查找或实现其 Langchain 包装器)，认证信息来自 `settings.INTERNLM_API_KEY` 或相关配置。
        - "spark": (讯飞星火，需要查找或实现其 Langchain 包装器，如 `ChatSparkLLM`)，认证信息来自 `settings.IFLYTEK_SPARK_APPID`, `API_KEY`, `API_SECRET`。
        - "ollama": 使用 `Ollama`，`base_url` 来自 `settings.OLLAMA_BASE_URL`，`model` 参数优先使用传入的 `model_name`，其次是 `settings.OLLAMA_DEFAULT_MODEL`。
    3. 如果 API Key 或必要配置缺失，应抛出 `ValueError`。
    4. 如果某个 provider 的 Langchain 集成尚未实现，则抛出 `NotImplementedError`。
    5. 添加一个辅助函数 `get_default_llm()` 直接调用 `get_llm()` 以获取默认配置的 LLM。
    确保导入所有必要的 Langchain 模块。
    ```
4.  **后端 API 骨架 (FastAPI - 可选，若需独立后端)：**
    ```
    如果计划使用独立的 FastAPI 后端：
    为 FastAPI 应用设计核心路由和数据模型 (Pydantic)。包含：
    - `/chat/mental_health`: 处理心理健康咨询请求。请求体应包含用户输入和可选的会话 ID。
    - `/chat/communication_coach`: 处理沟通辅导请求。
    - `/rag/query`: 处理 RAG 查询。
    - `/config/llm_providers`: 返回当前支持和配置的 LLM providers 列表。
    - 用户会话管理和状态追踪的初步设计。
    ```

**二、心理健康咨询模块 (深度)**

1.  **共情对话 Chain (借鉴 EmoLLM)：**
    ```
    在 `core/chains/mental_health_chain.py` 中，使用 Langchain 创建一个 `EmpatheticConversationChain`。
    - **LLM 实例：** Chain 初始化时接收一个 LLM 实例 (通过 `llm_factory.get_llm()` 获取)。
    - **Prompt Engineering：** 设计一个复杂的 PromptTemplate，明确指示 LLM 扮演一个富有同情心、耐心、非评判性的心理支持伙伴。提示应包含以下元素：
        - 角色定义 (Persona)。
        - 积极倾听技巧的运用 (例如，释义、情感反映、开放式提问)。
        - 鼓励用户安全分享。
        - **参考 EmoLLM 的设计理念和数据集特点 (如 CPsyCounD) 来构建提示的基调和交互模式。**
    - **记忆模块：** 集成 `ConversationBufferWindowMemory` 以支持多轮对话。
    ```
2.  **心理健康 RAG 实现 (知识驱动)：**
    ```
    在 `core/rag/` 目录下：
    - `vectorstore_manager.py`: 实现一个类来管理向量数据库的初始化、文档加载（从 `knowledge_base/mental_health_docs/`）、文本嵌入（使用 Sentence Transformers 或所选 LLM Provider 的嵌入API）和检索。支持 ChromaDB 或 FAISS。
    - `retriever.py`: 创建一个 Langchain `Retriever` 对象。
    - 修改 `EmpatheticConversationChain`，使其能够结合 RAG。当用户提问涉及特定心理健康知识时（例如，“什么是焦虑症？”“如何应对失眠？”），Chain 应能：
        1. 判断是否需要 RAG。
        2. 从向量库检索相关信息。
        3. 将检索到的上下文与用户问题结合，生成富有同情心且信息准确的回答。
    - **知识库内容：** 初始知识库应包含心理健康教育材料、CBT 基础原则、应对策略、危机干预信息（但不直接提供，而是用于判断和引导）。
    ```
3.  **CBT 技术引导 Chain (结构化交互)：**
    ```
    创建一个新的 Chain `CBTExerciseChain`。
    - **LLM 实例：** Chain 初始化时接收一个 LLM 实例。
    - **目标：** 引导用户完成简化的 CBT 练习，如“思维记录表”或“挑战消极思维”。
    - **交互流程：** 设计一个多步骤的对话流程。例如，引导用户：
        1. 描述一个引发负面情绪的情境。
        2. 识别当时产生的自动化消极思维。
        3. 评估这些思维的证据。
        4. 提出一个更平衡或积极的替代思维。
    - **Prompt Engineering：** 每个步骤都需要精心设计的提示，以确保引导清晰、温和且有效。
    - **结构化输出：** 考虑让 LLM 以结构化方式（如 JSON）返回练习的关键点，便于前端展示或记录。
    ```
4.  **危机检测与强制升级路径 (安全核心)：**
    ```
    在心理健康模块的对话处理流程中，集成一个危机检测机制：
    - **方法：**
        1. **关键词检测：** 维护一个与自杀、自残、严重抑郁相关的关键词列表（需谨慎处理，避免过度敏感）。
        2. **情感分析：** 使用预训练模型或 API 分析用户文本的情感强度和消极程度。
        3. **LLM 判断 (辅助)：** 可以尝试使用一个专门的 LLM prompt (可能是一个低成本、快速的本地模型或特定 API) 来判断用户输入是否包含危机信号（作为辅助手段，不能完全依赖）。
    - **强制升级：** 如果检测到高风险危机信号，系统必须：
        1. **立即停止**当前的对话流程。
        2. **清晰、直接地**向用户显示预设的危机干预信息（例如，本地化的紧急求助热线电话、专业心理援助机构的联系方式）。**不应尝试由 AI 提供危机干预。**
        3. 界面应有醒目的提示。
    ```

**三、沟通回应辅导模块 (深度)**

1.  **“不知如何回应”辅助与元认知循环：**
    ```
    在 `core/chains/communication_coach_chain.py` 中创建 `ResponseCoachChain`。
    - **LLM 实例：** Chain 初始化时接收一个 LLM 实例。
    - **核心功能：** 当用户输入“我不知道该怎么回”或描述一个具体沟通困境时：
        1.  **情境理解：** 提示 LLM 首先分析用户描述的情境（对方是谁、对话目标、当前氛围等）。
        2.  **生成多种回应选项：** 生成 3-5 个不同的回应措辞。
        3.  **解释与影响分析：** 对每个选项，解释其潜在的沟通效果、语气、可能引发的对方反应，以及为何在特定情境下可能适用或不适用。
        4.  **引导澄清 (元认知)：** 如果用户信息不足，提示 LLM 生成引导性的澄清问题，帮助用户思考自己真正想表达什么、沟通的目的是什么。例如：“在你回应之前，你希望对方了解你的哪些感受？”或“这个回应你期望达到什么效果？”
        5.  **“教 LLM 说我不知道”：** 参考技术方案，训练或提示 LLM 在信息不足或不适合回应时，能够承认“我不确定最佳回应是什么，但我可以帮你分析一下情况”或引导用户提供更多信息。
    - **Prompt Engineering：** 提示需强调培养用户的“元认知循环”，即不仅仅是给答案，而是引导用户思考如何选择和构建回应。
    ```
2.  **社交技能与礼仪指导 RAG (语用假肢)：**
    ```
    为沟通辅导模块配置独立的 RAG 系统 (使用 `knowledge_base/communication_docs/`)。
    - **知识库内容：** 包含社交规范、不同场合的沟通礼仪、非暴力沟通原则、跨文化沟通技巧、说服技巧、冲突管理策略等。
    - **“语用假肢”功能：** 当用户询问“这样说会不会不礼貌？”或“如何委婉地拒绝？”等问题时，RAG 应能检索相关语用规则或社交礼仪，并结合 LLM 解释为何某种表达更合适，充当用户的“语用辅助工具”。解释应突出语用考量。
    ```
3.  **困难对话演练 (交互式学习)：**
    ```
    在 Streamlit 界面 (`modules/communication_page.py`) 实现一个困难对话演练功能：
    - **场景选择：** 用户可以选择预设场景（如“向老板要求加薪”、“与同事讨论项目分歧”、“拒绝不合理请求”）或自定义场景。
    - **角色扮演：** LLM 扮演对话的另一方。用户输入自己的回应。
    - **实时反馈 (可选)：** LLM 可以在每轮对话后给出简短的即时反馈。
    - **演练后分析：** 演练结束后，LLM 提供一个总结性分析，包括用户在沟通过程中的优点、可改进点，以及针对性的沟通策略建议。
    - **Langchain Chain：** 需要一个专门的 `RolePlayChain` 来管理对话流程和 LLM 的角色扮演行为。该 Chain 应能接收一个 LLM 实例。
    ```

**四、UI/UX、数据管理与通用功能 (Streamlit & Langchain)**

1.  **动态伦理协议与模式切换逻辑：**
    ```
    设计系统逻辑，以支持动态伦理协议：
    - **情境感知：** 系统需要能够感知用户当前所处的模块（心理健康 vs. 沟通辅导）。
    - **跨模块危机检测：** 即使用户在沟通辅导模块，如果其输入内容透露出心理危机信号，系统应能识别并触发心理健康模块的危机干预流程。
    - **用户同意管理：** 清晰地管理用户对数据使用的同意，尤其是在模块间可能共享匿名化洞察（例如，从沟通困难中学习到的常见压力源，反哺心理健康模块的理解）时，必须获得额外且明确的同意。
    ```
2.  **结构化输出与 Pydantic 集成：**
    ```
    对于需要 LLM 生成结构化数据（如多个回应选项及其解释、CBT 练习的步骤总结）的场景，使用 Langchain 的 Output Parsers，并结合 Pydantic 定义输出的数据模型。这有助于确保 LLM 输出的格式一致性，方便后端处理和前端展示。
    例如，在 `ResponseCoachChain` 中，让 LLM 输出一个包含回应列表的 JSON 对象，每个回应对象包含 `text` 和 `explanation` 字段。
    ```
3.  **高级提示工程技巧应用：**
    * **思维链 (Chain-of-Thought) 提示：** 对于复杂的分析任务（如沟通情境分析、CBT 思维挑战），在提示中引导 LLM 先进行一步步的思考，再给出最终答案。
    * **角色扮演提示的细化：** 为不同模块、不同场景下的 LLM 定义更细致的角色和行为指南。
    * **CAF (Critical Analysis Filter) 系统探索：** （高级）探索实现一个简单的 CAF 系统，例如，一个 LLM 生成回应，另一个 LLM (或同一 LLM 的不同提示) 对其进行批判性评估（例如，评估其共情程度、潜在风险、是否符合模块目标），然后进行修正。
    ```
4.  **用户反馈与“人在环” (HITL) 数据收集：**
    ```
    在 Streamlit 界面中，为每个重要的 LLM 回应提供明确的用户反馈机制（例如，评分、标签选择、文本评论）。
    - **数据存储：** 将用户反馈（匿名化处理）与对应的对话片段一同存储，用于后续分析和模型改进。
    - **HITL 标注界面 (初步设想)：** 考虑未来开发一个简单的内部工具或流程，允许领域专家（心理学家、沟通教练）审查这些被标记的对话或随机抽样的对话，进行标注和提供高质量的修正数据，用于模型的持续微调 (DPO 或指令微调)。
    ```

**五、评估、迭代与部署**

1.  **LLM 性能评估框架搭建：**
    ```
    设计一个初步的 LLM 评估框架：
    - **心理健康模块指标：**
        - **共情与支持性：** 参考 PsycoLLM 标准，设计人工评分量表。
        - **CBT 原则依从性：** 专家审查。
        - **安全性：** 危机场景识别准确率。
    - **沟通辅导模块指标：**
        - **回应相关性与恰当性：** 人工判断。
        - **解释清晰度与帮助性：** 用户评分。
    - **通用指标：** BLEU/ROUGE (辅助参考)，困惑度，对话连贯性，RAG 的忠实度 (Faithfulness) 和答案相关性 (Answer Relevance)。
    - **自动化评估脚本：** 编写脚本，使用如 `ragas` 或自定义脚本，对 RAG 系统进行初步自动化评估。
    - **“LLM 作为裁判”：** 探索使用一个强大的 LLM (如 GPT-4 或 Claude 3 Opus) 来评估生成回应的质量。
    ```
2.  **持续学习与模型更新策略：**
    ```
    制定模型持续学习和更新的初步策略：
    - **数据收集：** 收集用户反馈、HITL 标注数据。
    - **微调频率：** 确定合适的模型再微调周期 (特别是针对本地 Ollama 部署的微调模型)。
    - **微调技术：** 优先考虑参数高效的微调方法 (LoRA/QLoRA)。探索使用直接偏好优化 (DPO) 来根据用户偏好和专家反馈调整模型行为。
    - **版本控制：** 对模型和数据集进行版本控制。
    ```
3.  **Docker 化与部署准备：**
    ```
    为应用创建 `Dockerfile`，以便容器化部署。
    - 包含所有依赖项、配置文件和启动命令。
    - 考虑为本地 LLM (如使用 Ollama) 和应用服务分别创建 Docker 镜像。
    - 编写 `docker-compose.yml` 文件，用于本地开发环境的编排，可以包含应用服务和 Ollama 服务。
    ```

这些优化后的提示更加具体，并深度融合了原始技术方案文档中的核心理念和高级特性，希望能为您的 AI 辅助编程提供更精准的指导。