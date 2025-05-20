# LumiMind 项目使用指南

## 1. 项目概述

LumiMind 是一个基于大语言模型（LLM）的双模块应用程序，专注于：
1. 心理健康咨询：提供同理心对话和认知行为疗法（CBT）练习
2. 沟通辅导：帮助用户组织语言，提供回应建议和角色扮演练习

项目使用 Langchain 进行 LLM 工作流管理，Streamlit 提供用户界面，支持多种 LLM 提供商。

## 2. 安装指南

### 2.1 系统要求
- Python 3.8 或更高版本
- 至少 4GB RAM（推荐 8GB 或更高）
- 本地运行 Ollama（可选，如果要使用本地模型）

### 2.2 安装步骤

1. 克隆仓库（或下载项目文件）到本地
```bash
git clone [仓库地址]
cd LumiMind
```

2. 创建并激活虚拟环境（推荐）
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

## 3. API 配置

LumiMind 支持多种 LLM 提供商。您需要创建一个 `.env` 文件来存储 API 密钥。

### 3.1 创建 .env 文件

在项目根目录创建一个名为 `.env` 的文件，参考以下模板：

```
# 一般设置
APP_NAME="LumiMind"
APP_VERSION="0.1.0"
DEBUG=False

# 默认 LLM 提供商 (可选: "openai", "gemini", "ollama" 等)
DEFAULT_LLM_PROVIDER="openai"

# OpenAI 配置
OPENAI_API_KEY="sk-your-openai-api-key"
OPENAI_DEFAULT_MODEL="gpt-4-turbo"

# Google (Gemini) 配置
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_DEFAULT_MODEL="gemini-pro"

# DeepSeek 配置
DEEPSEEK_API_KEY="your-deepseek-api-key"
DEEPSEEK_DEFAULT_MODEL="deepseek-chat"

# SiliconFlow 配置
SILICONFLOW_API_KEY="your-siliconflow-api-key"
SILICONFLOW_DEFAULT_MODEL="sf-chat"

# InternLM 配置
INTERNLM_API_KEY="your-internlm-api-key"
INTERNLM_DEFAULT_MODEL="internlm-chat-20b"

# 讯飞星火配置
IFLYTEK_SPARK_APPID="your-iflytek-appid"
IFLYTEK_SPARK_API_KEY="your-iflytek-api-key"
IFLYTEK_SPARK_API_SECRET="your-iflytek-api-secret"
IFLYTEK_SPARK_DEFAULT_MODEL="spark-3.5"

# Ollama 配置（本地模型）
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_DEFAULT_MODEL="llama3"
```

### 3.2 如何获取各平台 API 密钥

#### OpenAI API 密钥
1. 访问 [OpenAI 平台](https://platform.openai.com/)
2. 注册或登录账户
3. 进入 API 密钥部分
4. 创建新密钥
5. 复制密钥并添加到 `.env` 文件

#### Google Gemini API 密钥
1. 访问 [Google AI Studio](https://makersuite.google.com/)
2. 注册或登录 Google 账户
3. 进入 API 密钥管理
4. 创建新密钥
5. 复制密钥并添加到 `.env` 文件

#### 本地 Ollama 设置
1. 从 [Ollama 官网](https://ollama.ai/) 下载并安装 Ollama
2. 使用以下命令拉取模型
```bash
ollama pull llama3
```
3. Ollama 默认在 `http://localhost:11434` 运行，无需额外配置

#### 其他 API 提供商
其他提供商（如 DeepSeek、SiliconFlow、InternLM、讯飞星火）的 API 密钥需要访问各自的官方网站注册并获取。

### 3.3 切换 LLM 提供商

您可以通过两种方式切换 LLM 提供商：
1. 在 `.env` 文件中修改 `DEFAULT_LLM_PROVIDER` 设置
2. 在应用程序运行时，通过侧边栏中的下拉菜单选择不同的提供商

## 4. 运行应用程序

在项目根目录下执行：

```bash
streamlit run app.py
```

应用程序将在浏览器中打开，默认地址为 http://localhost:8501

## 5. 使用心理健康咨询模块

心理健康咨询模块提供两种主要功能：

### 5.1 同理心对话

此模式提供一个安全的空间表达您的想法和感受：
1. 在左侧导航中选择"Mental Health Support"
2. 确保选择"Empathetic Chat"模式
3. 在聊天框中输入您的想法或感受
4. AI 助手将以同理心方式回应，并在适当时提供支持性建议

### 5.2 认知行为疗法（CBT）练习

此模式引导您完成 CBT 思维记录练习：
1. 在左侧导航中选择"Mental Health Support"
2. 选择"CBT Exercise"模式
3. 系统将引导您完成以下步骤：
   - 描述引发负面情绪的具体情境
   - 识别自动产生的想法
   - 确定并评估情绪强度
   - 识别认知扭曲
   - 评估支持和反对证据
   - 发展更平衡的替代想法
   - 反思练习后的情绪变化

### 5.3 注意事项

- 系统集成了危机检测机制，如果检测到高风险信号，将提供危机干预信息
- 所有对话仅在当前会话中保存，刷新页面后将丢失
- 此功能不能替代专业心理健康服务

## 6. 使用沟通辅导模块

沟通辅导模块提供两种主要功能：

### 6.1 回应教练

此模式帮助您在各种沟通场景中组织语言：
1. 在左侧导航中选择"Communication Coach"
2. 确保选择"Response Coach"模式
3. 描述您遇到的沟通困境或需要回应的情境
4. AI 将分析情境并提供：
   - 情境分析
   - 3-5 个不同的回应选项
   - 每个选项的解释（包括语气、潜在影响和适用性）
   - 帮助您反思沟通目标的问题

### 6.2 角色扮演练习

此模式让您在安全环境中练习困难对话：
1. 在左侧导航中选择"Communication Coach"
2. 选择"Role Play Practice"模式
3. 从预设场景中选择一个（如"薪资谈判"、"给予困难反馈"等）
4. 点击"Start Role Play"开始练习
5. AI 将扮演另一方角色，您可以练习真实对话
6. 随时通过输入"How did I do?"请求反馈
7. 输入"End role play"或使用侧边栏中的按钮结束角色扮演

### 6.3 可用场景

当前提供以下预设场景：
- 薪资谈判 (Salary Negotiation)
- 给予困难反馈 (Giving Difficult Feedback)
- 设立界限 (Setting Boundaries)
- 冲突解决 (Conflict Resolution)
- 处理客户投诉 (Handling a Customer Complaint)

## 7. 故障排除

### 7.1 常见问题

1. **API 错误**
   - 确保您已正确配置 API 密钥
   - 检查 API 账户余额是否充足
   - 验证互联网连接

2. **Ollama 连接问题**
   - 确保 Ollama 服务正在运行
   - 验证默认地址是否正确 (http://localhost:11434)
   - 检查是否已安装所需模型

3. **内存错误**
   - 减少使用的上下文窗口大小
   - 关闭其他内存密集型应用程序
   - 考虑使用较小的 LLM 模型

### 7.2 清除缓存

如果应用程序运行异常，您可以尝试：
1. 点击侧边栏中的"Clear Conversation"按钮
2. 刷新浏览器页面
3. 重启 Streamlit 应用程序

## 8. 数据隐私说明

- 所有对话数据仅在本地会话中处理，不会永久存储
- 使用 API 提供商时，数据会发送至其服务器进行处理
- 使用 Ollama 本地模型时，所有数据处理在本地完成，不会离开您的设备
- 危机检测功能仅在本地进行关键词匹配，不会外发数据

## 9. 联系与支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目仓库地址]
- 电子邮件: [联系邮箱]

---

**免责声明**：LumiMind 不是专业心理健康服务或专业沟通培训的替代品。在危机情况下，请寻求专业帮助。 