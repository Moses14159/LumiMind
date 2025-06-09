# LumiMind - AI驱动的心理健康与沟通辅导平台

LumiMind 是一个基于人工智能的心理健康咨询和沟通辅导平台，集成了多种大语言模型，提供专业的心理健康支持和沟通技巧指导。

## 主要功能

### 心理健康咨询
- 共情对话：理解用户情绪，提供情感支持
- 知识普及：提供心理健康知识
- CBT指导：认知行为疗法指导
- 危机信号检测：识别潜在风险
- 对话上下文分析：理解用户需求
- 回复选项生成：提供专业建议

### 沟通辅导
- 沟通技巧指导
- 场景模拟训练
- 反馈和建议
- 个性化学习计划
- 进度跟踪

## 快速开始

### 环境要求
- Python 3.8+
- 虚拟环境（推荐）
- 必要的API密钥

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/LumiMind.git
cd LumiMind
```

2. 创建虚拟环境
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
# 复制示例配置文件
cp .env.example .env

# 编辑 .env 文件，填入必要的API密钥
```

### 启动应用

1. 初始化环境
```bash
python scripts/setup.py
```

2. 初始化知识库
```bash
python scripts/initialize_kb.py
```

3. 启动应用
```bash
python scripts/start.py
```

应用将在浏览器中自动打开，默认地址为 http://localhost:8501

## 使用指南

### 界面导航

1. 侧边栏
   - 模块选择：心理健康咨询/沟通辅导
   - AI模型选择：选择不同的语言模型
   - 知识库管理：上传和管理文档
   - 对话历史：查看历史记录

2. 主界面
   - 对话区域：与AI助手交流
   - 设置面板：调整参数
   - 反馈区域：提供反馈

### 文档管理

1. 支持的文件格式
   - TXT：纯文本文件
   - PDF：PDF文档
   - DOCX：Word文档
   - MD：Markdown文件

2. 上传步骤
   - 在侧边栏选择"知识库管理"
   - 点击"上传文档"
   - 选择文件
   - 等待处理完成

3. 注意事项
   - 文件大小限制：10MB
   - 支持批量上传
   - 自动进行安全检查
   - 自动分块处理

## 安全与隐私

1. 数据安全
   - 文件安全检查
   - 敏感信息过滤
   - 数据加密存储
   - 定期备份

2. 隐私保护
   - 用户数据隔离
   - 对话历史加密
   - 可选的匿名模式
   - 数据删除选项

## 故障排除

### 常见问题

1. 文件上传失败
   - 检查文件格式
   - 确认文件大小
   - 验证网络连接
   - 查看错误日志

2. 模型响应异常
   - 检查API密钥
   - 确认网络连接
   - 尝试其他模型
   - 查看错误日志

3. 知识库问题
   - 检查文档格式
   - 确认处理状态
   - 重新初始化
   - 查看错误日志

### 获取帮助

- 查看文档：`docs/`目录
- 提交Issue：GitHub Issues
- 联系支持：support@lumimind.com

## 使用限制

1. 功能限制
   - 单次对话长度限制
   - 文件上传大小限制
   - 知识库容量限制
   - API调用频率限制

2. 使用建议
   - 定期备份数据
   - 及时更新系统
   - 遵守使用规范
   - 保护账号安全

## 贡献指南

1. 开发流程
   - Fork项目
   - 创建分支
   - 提交更改
   - 发起PR

2. 代码规范
   - 遵循PEP 8
   - 添加注释
   - 编写测试
   - 更新文档

## 许可证

本项目采用 MIT 许可证

## 联系方式

- 项目主页：https://github.com/Moses14159/LumiMind
- 问题反馈：https://github.com/Moses14159/LumiMind/issues
