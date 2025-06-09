# LumiMind 项目架构

## 系统架构图

```mermaid
graph TB
    subgraph Frontend["前端界面 (Streamlit)"]
        UI[用户界面]
        Sidebar[侧边栏]
        Chat[聊天界面]
        Upload[文件上传]
    end

    subgraph Core["核心模块"]
        subgraph Chains["Langchain Chains"]
            MH[心理健康咨询链]
            Comm[沟通辅导链]
            CBT[CBT练习链]
            Response[回应教练链]
        end

        subgraph RAG["检索增强生成"]
            KB[知识库管理]
            VS[向量存储]
            Retriever[检索器]
        end

        subgraph Utils["工具类"]
            LLM[LLM工厂]
            Doc[文档处理器]
            Security[安全处理器]
            Error[错误处理器]
            Crisis[危机检测器]
            Emotion[情绪分析器]
        end
    end

    subgraph Data["数据层"]
        subgraph Storage["存储"]
            VDB[向量数据库]
            Cache[缓存]
            Logs[日志]
        end

        subgraph Knowledge["知识库"]
            MHDocs[心理健康文档]
            CommDocs[沟通辅导文档]
        end
    end

    subgraph External["外部服务"]
        LLMProviders["LLM提供商"]
        API[API服务]
    end

    %% 连接关系
    UI --> Sidebar
    UI --> Chat
    UI --> Upload

    Sidebar --> Chains
    Chat --> Chains
    Upload --> Doc

    Chains --> LLM
    Chains --> RAG
    Chains --> Utils

    RAG --> KB
    KB --> VS
    VS --> Retriever

    Doc --> Security
    Doc --> KB
    Security --> Error

    KB --> Storage
    Doc --> Storage
    Error --> Logs

    LLM --> External
    API --> External

    %% 样式
    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef core fill:#bbf,stroke:#333,stroke-width:2px
    classDef data fill:#bfb,stroke:#333,stroke-width:2px
    classDef external fill:#fbb,stroke:#333,stroke-width:2px

    class Frontend frontend
    class Core core
    class Data data
    class External external
```

## 组件说明

### 前端界面 (Streamlit)
- **用户界面**: 主界面布局和交互
- **侧边栏**: 模块选择、模型选择、文件上传
- **聊天界面**: 对话交互区域
- **文件上传**: 文档上传和管理

### 核心模块
#### Langchain Chains
- **心理健康咨询链**: 处理心理健康相关咨询
- **沟通辅导链**: 处理沟通技巧辅导
- **CBT练习链**: 认知行为疗法练习
- **回应教练链**: 沟通回应建议

#### 检索增强生成 (RAG)
- **知识库管理**: 管理文档和知识
- **向量存储**: 存储文档向量
- **检索器**: 检索相关文档

#### 工具类
- **LLM工厂**: 管理不同LLM提供商
- **文档处理器**: 处理上传文档
- **安全处理器**: 安全检查
- **错误处理器**: 错误处理
- **危机检测器**: 检测危机信号
- **情绪分析器**: 分析用户情绪

### 数据层
#### 存储
- **向量数据库**: 存储文档向量
- **缓存**: 缓存数据
- **日志**: 系统日志

#### 知识库
- **心理健康文档**: 心理健康知识
- **沟通辅导文档**: 沟通技巧知识

### 外部服务
- **LLM提供商**: 各种语言模型服务
- **API服务**: 外部API集成

## 数据流

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as 前端界面
    participant Chain as 处理链
    participant RAG as 知识检索
    participant LLM as 语言模型
    participant Storage as 存储

    User->>UI: 输入问题
    UI->>Chain: 发送请求
    Chain->>RAG: 检索相关知识
    RAG->>Storage: 查询向量数据库
    Storage-->>RAG: 返回相关文档
    RAG-->>Chain: 返回知识上下文
    Chain->>LLM: 生成回答
    LLM-->>Chain: 返回回答
    Chain-->>UI: 返回处理结果
    UI-->>User: 显示回答
```

## 安全架构

```mermaid
graph LR
    subgraph Security["安全层"]
        Auth[认证]
        Encrypt[加密]
        Validate[验证]
        Monitor[监控]
    end

    subgraph Components["组件"]
        Upload[文件上传]
        Process[数据处理]
        Storage[数据存储]
        API[API调用]
    end

    Upload --> Auth
    Upload --> Validate
    Process --> Encrypt
    Storage --> Encrypt
    API --> Auth
    API --> Monitor

    classDef security fill:#fbb,stroke:#333,stroke-width:2px
    classDef component fill:#bbf,stroke:#333,stroke-width:2px

    class Security security
    class Components component
``` 