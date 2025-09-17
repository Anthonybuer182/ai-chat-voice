# AI语音聊天系统

一个基于FastAPI和WebSocket的双模式AI语音聊天系统，支持文本/语音混合聊天和实时语音对话。

## 功能特性

### Tab 1: 文本与语音聊天
- ✅ 文本消息输入和发送
- ✅ 流式文本响应显示
- ✅ 按钮触发录音功能
- ✅ 语音转文本（Faster-Whisper）
- ✅ 文本转语音（gTTS）
- ✅ OpenAI GPT智能对话
- ✅ 聊天历史管理

### Tab 2: 实时语音对话
- ✅ 自动语音检测（VAD）
- ✅ 连续语音流处理
- ✅ 实时语音转文本
- ✅ 快速AI响应
- ✅ 语音合成播放
- ✅ 音量控制
- ✅ 可视化音频波形

## 技术栈

- **后端**: FastAPI, WebSocket, Python 3.8+
- **AI模型**: OpenAI GPT-3.5, Faster-Whisper
- **语音合成**: Google Text-to-Speech (gTTS)
- **前端**: 原生JavaScript, HTML5, CSS3
- **音频处理**: Web Audio API, MediaRecorder API

## 安装步骤

### 1. 克隆项目
```bash
mkdir ai-voice-chat
cd ai-voice-chat
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
创建 `.env` 文件：
```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. 创建项目文件
- 将HTML内容保存为 `index.html`
- 将Python后端代码保存为 `main.py`
- 将requirements内容保存为 `requirements.txt`

## 运行项目

```bash
python main.py
```

服务器将在 http://localhost:8000 启动

## 使用说明

### Tab 1: 文本与语音聊天
1. 在输入框输入文本消息，按Enter或点击发送
2. 点击录音按钮开始录音，再次点击停止
3. AI会以流式文本形式响应，并自动播放语音

### Tab 2: 实时语音对话
1. 切换到第二个标签页
2. 允许浏览器访问麦克风
3. 直接说话，系统会自动检测和处理
4. AI会快速响应并播放语音回复

## 配置选项

在 `main.py` 中的 `Config` 类可以调整：
- `OPENAI_MODEL`: GPT模型选择
- `WHISPER_MODEL`: 语音识别模型大小
- `TTS_LANGUAGE`: 语音合成语言
- `SAMPLE_RATE`: 音频采样率

## 注意事项

1. **API密钥安全**: 不要将API密钥提交到版本控制
2. **浏览器兼容**: 需要支持WebSocket和MediaRecorder的现代浏览器
3. **麦克风权限**: 首次使用需要授予麦克风访问权限
4. **网络要求**: 需要稳定的网络连接访问OpenAI API

## 性能优化建议

1. **Whisper模型选择**:
   - `tiny`: 最快，准确度较低
   - `base`: 平衡选择（推荐）
   - `small/medium`: 更高准确度，更慢
   - `large`: 最高准确度，需要GPU

2. **缓冲区调整**: 
   - 实时语音的 `buffer_duration` 可根据需要调整
   - 较短的缓冲时间响应更快，但可能截断语音

3. **GPU加速**:
   - 安装CUDA版本的PyTorch可显著提升Whisper性能
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 故障排除

### 问题1: WebSocket连接失败
- 检查防火墙设置
- 确保端口8000未被占用

### 问题2: 语音识别不准确
- 尝试更大的Whisper模型
- 确保麦克风音质良好
- 调整音频能量检测阈值

### 问题3: OpenAI API错误
- 检查API密钥是否正确
- 确认账户余额充足
- 检查网络连接

## 扩展功能建议

1. **多语言支持**: 添加语言切换功能
2. **语音选择**: 集成更多TTS引擎（如Azure、Amazon Polly）
3. **对话导出**: 添加聊天记录导出功能
4. **用户认证**: 添加用户登录和个性化设置
5. **情绪识别**: 集成语音情绪分析
6. **离线模式**: 集成本地LLM模型

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交GitHub Issue或联系项目维护者。