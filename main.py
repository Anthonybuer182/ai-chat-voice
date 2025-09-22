"""
AI语音聊天系统后端
需要安装的依赖：
pip install fastapi uvicorn websockets openai gtts faster-whisper numpy scipy python-multipart
"""

import os
import json
import base64
import asyncio
import tempfile
import time
from typing import Dict, List, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from scipy.io.wavfile import write as write_wav
from gtts import gTTS
from openai import OpenAI, AsyncOpenAI
from faster_whisper import WhisperModel
import logging
from dotenv import load_dotenv
import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置
@dataclass
class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
    WHISPER_MODEL = "base"  # tiny, base, small, medium, large中文
    SAMPLE_RATE = 16000

    
    # 系统提示词模板
    SYSTEM_PROMPTS = {
        "zh": {
            "chat": "你是一个友好的AI助手，请用中文回答。",
            "voice": "你是一个友好的AI助手，请简洁地用中文回答，回答控制在50字以内。"
        },
        "en": {
            "chat": "You are a friendly AI assistant. Please respond in English.",
            "voice": "You are a friendly AI assistant. Please respond concisely in English, keeping your answers under 50 words."
        }
    }
    
config = Config()

# 初始化OpenAI客户端（用于DeepSeek API）
client = AsyncOpenAI(
    api_key=config.DEEPSEEK_API_KEY,
    base_url=config.DEEPSEEK_API_BASE
)

# 初始化Whisper模型
whisper_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

# 创建FastAPI应用
app = FastAPI(title="AI Voice Chat System")

# 配置静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {
            "chat": [],
            "voice": []
        }
    
    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        self.active_connections[channel].append(websocket)
        logger.info(f"Client connected to {channel}. Total connections: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        self.active_connections[channel].remove(websocket)
        logger.info(f"Client disconnected from {channel}. Remaining connections: {len(self.active_connections[channel])}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)

manager = ConnectionManager()

# 聊天历史管理
class ChatHistory:
    def __init__(self, max_history: int = 10):
        self.histories: Dict[str, List[dict]] = {}
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        self.histories[session_id].append({
            "role": role,
            "content": content
        })
        
        # 限制历史长度
        if len(self.histories[session_id]) > self.max_history * 2:
            self.histories[session_id] = self.histories[session_id][-self.max_history:]
    
    def get_history(self, session_id: str) -> List[dict]:
        return self.histories.get(session_id, [])
    
    def clear_history(self, session_id: str):
        if session_id in self.histories:
            del self.histories[session_id]

chat_history = ChatHistory()

# 音频处理工具
class AudioProcessor:
    @staticmethod
    def base64_to_audio(base64_data: str) -> bytes:
        """将base64音频数据转换为二进制"""
        return base64.b64decode(base64_data)
    
    @staticmethod
    def audio_to_base64(audio_data: bytes) -> str:
        """将音频二进制数据转换为base64"""
        return base64.b64encode(audio_data).decode('utf-8')
    
    @staticmethod
    def webm_to_wav(webm_data: bytes) -> str:
        """将WebM音频转换为WAV格式（用于Whisper）"""
        # 这里需要使用ffmpeg或其他工具进行转换
        # 简化处理，直接保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp.write(webm_data)
            return tmp.name
    
    @staticmethod
    def float32_to_wav(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """将float32音频数据转换为WAV格式"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)
            write_wav(tmp.name, sample_rate, audio_int16)
            with open(tmp.name, 'rb') as f:
                wav_data = f.read()
            os.unlink(tmp.name)
            return wav_data

audio_processor = AudioProcessor()

# AI服务
class AIService:
    def __init__(self):
        pass
    
    @staticmethod
    async def get_chat_response(messages: List[dict]) -> str:
        """获取DeepSeek聊天响应"""
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return "抱歉，我遇到了一些问题。请稍后再试。"
    
    @staticmethod
    async def get_chat_response_stream(messages: List[dict]) -> AsyncGenerator[str, None]:
        """获取DeepSeek流式聊天响应"""
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7
            )
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek API stream error: {e}")
            yield "抱歉，我遇到了一些问题。请稍后再试。"
    
    async def text_to_speech_with_engine(self, text: str, engine: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用指定引擎将文本转换为语音"""
        if engine == "gtts":
            return await self._gtts_tts(text, language, max_retries)
        elif engine == "edgetts":
            return await self._edgetts_tts(text, language, max_retries)
        elif engine == "elevenlabs":
            return await self._elevenlabs_tts(text, language, max_retries)
        else:
            logger.warning(f"未知的TTS引擎: {engine}, 使用gTTS作为后备")
            return await self._gtts_tts(text, language, max_retries)

    async def _gtts_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用gTTS进行语音合成"""
        for attempt in range(max_retries):
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                
                # 使用不同的方法避免文件访问冲突
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # 保存到文件
                tts.save(tmp_path)
                
                # 读取文件内容
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                # 删除临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # 忽略删除错误
                
                logger.info(f"gTTS成功 (尝试 {attempt + 1})")
                return audio_data
                
            except Exception as e:
                logger.warning(f"gTTS尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 等待1秒后重试
                else:
                    # 清理可能的临时文件
                    try:
                        if 'tmp_path' in locals():
                            os.unlink(tmp_path)
                    except:
                        pass
        
        logger.error("gTTS所有尝试都失败")
        return b""
    
    async def _edgetts_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用EdgeTTS进行语音合成（需要安装edge-tts）"""
        try:
            # 检查是否安装了edge-tts
            try:
                import edge_tts
            except ImportError:
                logger.warning("edge-tts未安装，使用pip install edge-tts安装")
                return await self._gtts_tts(text, language, max_retries)
            
            # 使用edge-tts
            communicate = edge_tts.Communicate(text, language)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
            
            await communicate.save(tmp_path)
            
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            logger.info("EdgeTTS成功")
            return audio_data
            
        except Exception as e:
                logger.error(f"EdgeTTS失败: {e}")
                return await self._gtts_tts(text, language, max_retries)  # 失败时回退到gTTS
    
    async def _elevenlabs_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用ElevenLabs进行语音合成（需要API密钥）"""
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            logger.warning("ElevenLabs API密钥未设置，使用gTTS作为后备")
            return await self._gtts_tts(text, language, max_retries)
        
        try:
            # 使用ElevenLabs API
            import requests
            
            url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                logger.info("ElevenLabs TTS成功")
                return response.content
            else:
                logger.error(f"ElevenLabs API错误: {response.status_code} - {response.text}")
                return await self._gtts_tts(text, language, max_retries)
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS失败: {e}")
            return await self._gtts_tts(text, language, max_retries)
    
    @staticmethod
    async def _fallback_tts(text: str) -> bytes:
        """备用TTS方案 - 使用简单的在线TTS服务"""
        try:
            # 这里可以使用其他TTS服务，如Microsoft Edge TTS的模拟
            # 由于网络限制，这里提供一个简单的本地生成方案作为fallback
            logger.warning("使用简易TTS fallback，建议安装edge-tts: pip install edge-tts")
            
            # 创建一个简单的音频占位符
            # 在实际应用中，可以安装edge-tts: pip install edge-tts
            # 或者使用其他本地TTS方案
            
            # 返回空的音频数据，前端会显示文本但无语音
            return b""
            
        except Exception as e:
            logger.error(f"Fallback TTS error: {e}")
            return b""
    
    @staticmethod
    async def speech_to_text(audio_file: str) -> str:
        """将语音转换为文本"""
        try:
            segments, info = whisper_model.transcribe(
                audio_file,
                language="zh",
                beam_size=5
            )
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return ""

# 创建AIService实例
ai_service = AIService()

# 路由
@app.get("/")
async def get_index():
    """返回HTML页面"""
    # 这里应该返回上面创建的HTML文件
    # 实际使用时，将HTML保存为单独的文件并通过FileResponse返回
    return HTMLResponse(content=open("index.html", "r", encoding="utf-8").read())

# WebSocket端点 - 文本和语音聊天
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """处理文本和录音聊天"""
    session_id = f"chat_{id(websocket)}"
    await manager.connect(websocket, "chat")
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "text":
                # 处理文本消息
                user_message = data.get("content", "")
                
                # 添加到历史
                chat_history.add_message(session_id, "user", user_message)
                
                # 获取AI响应（流式）
                language = data.get("language", "zh")
                messages = [
                    {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["chat"]},
                    *chat_history.get_history(session_id)
                ]
                
                # 发送流式响应
                full_response = ""
                async for chunk in ai_service.get_chat_response_stream(messages):
                    # 确定状态：如果是第一个块就是start，最后一个块就是end，中间的是continue
                    if not full_response:  # 第一个块
                        status = "start"
                    else:
                        status = "continue"
                    
                    await manager.send_json(websocket, {
                        "type": "text_chunk",
                        "content": chunk,
                        "status": status
                    })
                    full_response += chunk
                
                # 发送结束状态（如果有内容）
                if full_response:
                    await manager.send_json(websocket, {
                        "type": "text_chunk",
                        "content": "",
                        "status": "end"
                    })
                
                # 添加完整响应到历史
                chat_history.add_message(session_id, "assistant", full_response)
                
                # 生成语音
                if full_response:
                    tts_engine = data.get("tts_engine", "gtts")
                    audio_data = await ai_service.text_to_speech_with_engine(full_response, tts_engine, language)
                    if audio_data:
                        await manager.send_json(websocket, {
                            "type": "audio",
                            "audio_data": audio_processor.audio_to_base64(audio_data)
                        })
                
                # 发送完成信号
                await manager.send_json(websocket, {"type": "complete"})
            
            elif message_type == "audio":
                # 处理音频消息
                audio_base64 = data.get("audio_data", "")
                audio_data = audio_processor.base64_to_audio(audio_base64)
                
                # 保存音频文件
                audio_file = audio_processor.webm_to_wav(audio_data)
                
                # 语音转文本
                transcribed_text = await ai_service.speech_to_text(audio_file)
                os.unlink(audio_file)
                
                if transcribed_text:
                    # 添加到历史
                    chat_history.add_message(session_id, "user", transcribed_text)
                    
                    # 获取AI响应
                    language = data.get("language", "zh")
                    messages = [
                        {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["chat"]},
                        *chat_history.get_history(session_id)
                    ]
                    
                    # 发送流式文本响应
                    full_response = ""
                    async for chunk in ai_service.get_chat_response_stream(messages):
                        # 确定状态：如果是第一个块就是start，最后一个块就是end，中间的是continue
                        if not full_response:  # 第一个块
                            status = "start"
                        else:
                            status = "continue"
                        
                        await manager.send_json(websocket, {
                            "type": "text_chunk",
                            "content": chunk,
                            "status": status
                        })
                        full_response += chunk
                    
                    # 发送结束状态（如果有内容）
                    if full_response:
                        await manager.send_json(websocket, {
                            "type": "text_chunk",
                            "content": "",
                            "status": "end"
                        })
                    
                    # 添加到历史
                    chat_history.add_message(session_id, "assistant", full_response)
                    
                    # 生成语音响应
                    tts_engine = data.get("tts_engine", "gtts")
                    audio_response = await ai_service.text_to_speech_with_engine(response_text, tts_engine, language)
                    if audio_response:
                        await manager.send_json(websocket, {
                            "type": "audio",
                            "audio_data": audio_processor.audio_to_base64(audio_response)
                        })
                    
                    # 发送完成信号
                    await manager.send_json(websocket, {"type": "complete"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, "chat")
        chat_history.clear_history(session_id)
        logger.info(f"Chat session {session_id} ended")
    except Exception as e:
        logger.error(f"Chat websocket error: {e}")
        manager.disconnect(websocket, "chat")

# WebSocket端点 - 实时语音对话
@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """处理实时语音对话"""
    session_id = f"voice_{id(websocket)}"
    await manager.connect(websocket, "voice")
    
    # 音频缓冲区
    audio_buffer = []
    buffer_duration = 2.0  # 2秒缓冲
    buffer_size = int(config.SAMPLE_RATE * buffer_duration)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "audio_stream":
                # 处理音频流
                audio_chunk = data.get("audio_data", [])
                audio_buffer.extend(audio_chunk)
                
                # 当缓冲区达到指定大小时处理
                if len(audio_buffer) >= buffer_size:
                    # 转换为numpy数组
                    audio_array = np.array(audio_buffer[:buffer_size], dtype=np.float32)
                    
                    # 检测是否有语音活动（简单的能量检测）
                    energy = np.sqrt(np.mean(audio_array**2))
                    if energy > 0.01:  # 阈值可调整
                        # 转换为WAV
                        wav_data = audio_processor.float32_to_wav(audio_array, config.SAMPLE_RATE)
                        
                        # 保存临时文件
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                            tmp.write(wav_data)
                            audio_file = tmp.name
                        
                        # 语音识别
                        transcribed_text = await ai_service.speech_to_text(audio_file)
                        os.unlink(audio_file)
                        
                        if transcribed_text and len(transcribed_text) > 2:
                            # 发送转录文本
                            await manager.send_json(websocket, {
                                "type": "transcription",
                                "text": transcribed_text
                            })
                            
                            # 添加到历史
                            chat_history.add_message(session_id, "user", transcribed_text)
                            
                            # 获取AI响应
                            language = data.get("language", "zh")
                            messages = [
                                {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["voice"]},
                                *chat_history.get_history(session_id)[-6:]  # 只保留最近的3轮对话
                            ]
                            
                            response_text = await ai_service.get_chat_response(messages)
                            
                            # 发送响应文本
                            await manager.send_json(websocket, {
                                "type": "response",
                                "text": response_text
                            })
                            
                            # 添加到历史
                            chat_history.add_message(session_id, "assistant", response_text)
                            
                            # 生成语音响应
                            tts_engine = data.get("tts_engine", "gtts")
                            audio_response = await ai_service.text_to_speech_with_engine(response_text, tts_engine, language)
                            if audio_response:
                                await manager.send_json(websocket, {
                                    "type": "audio",
                                    "audio_data": audio_processor.audio_to_base64(audio_response)
                                })
                    
                    # 清空缓冲区
                    audio_buffer = audio_buffer[buffer_size:]
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, "voice")
        chat_history.clear_history(session_id)
        logger.info(f"Voice session {session_id} ended")
    except Exception as e:
        logger.error(f"Voice websocket error: {e}")
        manager.disconnect(websocket, "voice")

# 健康检查
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {
            "chat": len(manager.active_connections["chat"]),
            "voice": len(manager.active_connections["voice"])
        }
    }

# 主函数
if __name__ == "__main__":
    import uvicorn
    
    # 确保HTML文件存在
    html_file_path = "index.html"
    if not os.path.exists(html_file_path):
        logger.warning(f"HTML file not found at {html_file_path}. Please create it with the provided HTML content.")
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
