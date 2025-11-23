# AI语音聊天系统 (AI Chat Voice)

一个基于FastAPI和WebSocket的智能语音聊天系统，支持文本和语音两种交互模式，集成AI大语言模型进行智能对话。

## 🌟 项目特性

### 核心功能
- **双模式交互**: 支持文本聊天和语音聊天两种模式
- **智能AI对话**: 集成DeepSeek等大语言模型，提供智能回复
- **实时语音识别**: 使用Whisper模型进行高精度语音转文字
- **多引擎TTS**: 支持多种文本转语音引擎（gTTS、EdgeTTS、pyttsx3）
- **WebSocket实时通信**: 基于WebSocket实现低延迟的实时对话
- **多语言支持**: 支持中文和英文对话

### 技术特色
- **流式响应**: AI回复支持流式输出，提升用户体验
- **句子级处理**: 按句子边界智能分割，实现更自然的语音交互
- **性能监控**: 内置性能监控和日志系统
- **配置化管理**: 支持环境变量配置，易于部署
- **响应式UI**: 现代化Web界面，支持移动端适配

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 支持的操作系统：Windows、Linux、macOS

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd ai-chat-voice
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
创建 `.env` 文件并配置API密钥：
```env
API_KEY=your-deepseek-api-key-here
MODEL=deepseek-chat
API_BASE=https://api.deepseek.com/v1
WHISPER_MODEL=base
```

4. **启动服务**
```bash
python main.py
```

5. **访问应用**
打开浏览器访问：http://localhost:8000

## 📁 项目结构

```
ai-chat-voice/
├── main.py              # 主程序文件，包含后端逻辑
├── index.html           # 前端界面文件
├── requirements.txt     # Python依赖包列表
├── README.md           # 项目说明文档
├── .env                # 环境配置文件（需手动创建）
└── static/
    └── favicon.ico     # 网站图标
```

## 🔧 核心模块

### 后端模块
- **Config**: 配置管理类，管理API密钥和系统参数
- **ConnectionManager**: WebSocket连接管理器
- **ChatHistory**: 聊天历史管理，支持会话隔离
- **AudioProcessor**: 音频处理工具，支持格式转换
- **AIService**: AI服务核心，集成聊天和TTS功能
- **SentenceProcessor**: 句子处理，实现智能分割

### 前端功能
- **双标签页界面**: 文本聊天和语音聊天独立标签
- **实时消息显示**: 支持流式消息和语音播放
- **语音录制**: 基于Web Audio API的录音功能
- **状态指示器**: 连接状态和语音活动检测
- **响应式设计**: 适配不同屏幕尺寸

## ⚙️ 配置说明

### API配置
项目支持多种AI服务提供商，默认使用DeepSeek API：
- `API_KEY`: API访问密钥
- `MODEL`: 使用的模型名称
- `API_BASE`: API基础URL
- `WHISPER_MODEL`: 语音识别模型大小（tiny/base/small/medium/large）

### TTS引擎配置
支持多种TTS引擎，可通过前端界面选择：
- **gTTS**: Google Text-to-Speech（在线，质量好）
- **EdgeTTS**: Microsoft Edge TTS（在线，支持多种语音）
- **pyttsx3**: 离线TTS引擎（无需网络，响应快）

## 🎯 使用指南

### 文本聊天模式
1. 选择"文本聊天"标签页
2. 在输入框中输入问题
3. 点击发送按钮或按Enter键
4. 查看AI的实时回复

### 语音聊天模式
1. 选择"语音聊天"标签页
2. 点击麦克风按钮开始录音
3. 说话后松开按钮结束录音
4. 系统自动识别语音并获取AI回复
5. 回复内容将通过语音播放

### 高级功能
- **语言切换**: 支持中英文切换
- **TTS引擎选择**: 可根据需求选择不同的语音合成引擎
- **历史记录**: 对话历史自动保存，支持多会话管理

## 🔍 开发说明

### 代码结构
项目采用模块化设计，主要类包括：
- `Config`: 配置管理
- `ConnectionManager`: 连接管理
- `ChatHistory`: 历史管理
- `AIService`: AI服务核心

### 扩展开发
可以轻松扩展以下功能：
- 添加新的AI服务提供商
- 集成更多TTS引擎
- 支持更多语言
- 添加插件系统

## 🐛 故障排除

### 常见问题

**Q: 服务启动失败**
A: 检查Python版本和依赖包是否安装正确

**Q: API调用失败**
A: 验证API密钥配置是否正确，网络连接是否正常

**Q: 语音识别不准确**
A: 尝试使用更大的Whisper模型（如medium或large）

**Q: TTS没有声音**
A: 检查浏览器音频权限，尝试切换TTS引擎

### 日志查看
服务运行日志会输出到控制台，包含详细的调试信息。

## 📄 许可证

本项目基于MIT许可证开源，详见LICENSE文件。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**享受与AI的智能对话体验！** 🎉
