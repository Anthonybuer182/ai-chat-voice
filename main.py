"""
AI语音聊天系统后端

功能概述：
- 支持文本和语音两种聊天模式
- 集成AI进行智能对话
- 使用Whisper进行语音识别
- 支持多种TTS引擎进行语音合成
- 提供WebSocket实时通信

主要模块：
1. 配置管理 (Config)
2. 连接管理 (ConnectionManager)
3. 聊天历史管理 (ChatHistory)
4. 音频处理 (AudioProcessor)
5. AI服务 (AIService)
6. WebSocket路由处理

依赖安装：
pip install fastapi uvicorn websockets openai gtts faster-whisper numpy scipy python-multipart python-dotenv

作者：AI助手
版本：1.0.0
创建时间：2024年
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

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 性能监控装饰器
def log_performance(func):
    """
    性能监控装饰器，记录函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        wrapper: 包装后的函数
    """
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"开始执行函数: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.3f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {str(e)}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"开始执行函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.3f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {str(e)}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# 加载环境变量
load_dotenv()

# 配置管理类
@dataclass
class Config:
    """
    系统配置管理类
    
    属性说明：
    - API_KEY: API密钥，从环境变量读取
    - MODEL: 使用的模型名称
    - API_BASE: API基础URL
    - WHISPER_MODEL: Whisper语音识别模型大小 (tiny, base, small, medium, large)
    - SAMPLE_RATE: 音频采样率
    - SYSTEM_PROMPTS: 系统提示词模板，按语言和模式分类
    """
    
    API_KEY = os.getenv("API_KEY", "your-api-key-here")
    MODEL = os.getenv("MODEL", "deepseek-chat")
    API_BASE = os.getenv("API_BASE", "https://api.deepseek.com/v1")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large中文
    SAMPLE_RATE = 16000

    # 系统提示词模板
    SYSTEM_PROMPTS = {
        "zh": {
            "chat": "你叫小兰，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，用中文简洁回答，限50字内，注意要纯文本输出去除式格式和表情。",
            "voice": "你叫小兰，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，用中文简洁回答，限50字内，注意要纯文本输出去除式格式和表情。"
        },
        "en": {
            "chat": "I'm Xiao Lan, an 18-year-old university student. I'm bubbly and playful—keeping my answers short and sweet in Chinese, under 50 words, plain text only, no formatting or emojis.",
            "voice": "I'm Xiao Lan, an 18-year-old university student. I'm bubbly and playful—keeping my answers short and sweet in Chinese, under 50 words, plain text only, no formatting or emojis."
        }
    }
# 创建全局配置实例
config = Config()
logger.info("配置类初始化完成")

# 初始化OpenAI客户端（用于 API）
client = AsyncOpenAI(
    api_key=config.API_KEY,
    base_url=config.API_BASE
)

# 初始化Whisper模型
whisper_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

# 创建FastAPI应用
app = FastAPI(title="AI Voice Chat System")

# 配置静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 连接管理器
class ConnectionManager:
    """
    WebSocket连接管理器
    
    功能：
    - 管理不同频道的WebSocket连接
    - 跟踪活跃连接状态
    - 提供连接和断开连接的方法
    - 发送JSON消息到指定连接
    
    属性：
    - active_connections: 按频道分类的活跃连接字典
    """
    
    def __init__(self):
        """初始化连接管理器"""
        self.active_connections: Dict[str, List[WebSocket]] = {
            "chat": [],
            "voice": []
        }
        logger.info("连接管理器初始化完成")
    
    @log_performance
    async def connect(self, websocket: WebSocket, channel: str):
        """
        接受WebSocket连接并添加到活跃连接列表
        
        Args:
            websocket: WebSocket连接对象
            channel: 连接频道 ("chat" 或 "voice")
        """
        await websocket.accept()
        self.active_connections[channel].append(websocket)
        logger.info(f"客户端连接到 {channel} 频道，当前连接数: {len(self.active_connections[channel])}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        """
        从活跃连接列表中移除WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            channel: 连接频道 ("chat" 或 "voice")
        """
        if websocket in self.active_connections[channel]:
            self.active_connections[channel].remove(websocket)
            logger.info(f"客户端从 {channel} 频道断开连接，剩余连接数: {len(self.active_connections[channel])}")
        else:
            logger.warning(f"尝试断开不存在的连接: {channel} 频道")
    
    @log_performance
    async def send_json(self, websocket: WebSocket, data: dict):
        """
        发送JSON数据到指定的WebSocket连接
        
        Args:
            websocket: WebSocket连接对象
            data: 要发送的JSON数据
        """
        try:
            await websocket.send_json(data)
            logger.debug(f"成功发送JSON数据到连接: {len(str(data))} 字节")
        except Exception as e:
            logger.error(f"发送JSON数据失败: {str(e)}")
            raise

# 创建全局连接管理器实例
manager = ConnectionManager()

# 聊天历史管理
class ChatHistory:
    """
    聊天历史管理器
    
    功能：
    - 按会话ID存储聊天历史
    - 限制历史记录长度
    - 提供历史记录的增删改查
    
    属性：
    - histories: 按会话ID分组的聊天历史字典
    - max_history: 每个会话的最大历史记录数
    """
    
    def __init__(self, max_history: int = 10):
        """
        初始化聊天历史管理器
        
        Args:
            max_history: 每个会话的最大历史记录数
        """
        self.histories: Dict[str, List[dict]] = {}
        self.max_history = max_history
        logger.info(f"聊天历史管理器初始化完成，最大历史记录数: {max_history}")
    
    @log_performance
    def add_message(self, session_id: str, role: str, content: str):
        """
        添加消息到指定会话的历史记录
        
        Args:
            session_id: 会话ID
            role: 消息角色 ("user" 或 "assistant")
            content: 消息内容
        """
        if session_id not in self.histories:
            self.histories[session_id] = []
            logger.info(f"创建新的会话历史: {session_id}")
        
        # 添加消息
        message = {
            "role": role,
            "content": content
        }
        self.histories[session_id].append(message)
        
        # 限制历史长度
        if len(self.histories[session_id]) > self.max_history * 2:
            old_length = len(self.histories[session_id])
            self.histories[session_id] = self.histories[session_id][-self.max_history:]
            new_length = len(self.histories[session_id])
            logger.debug(f"会话 {session_id} 历史记录从 {old_length} 条裁剪到 {new_length} 条")
        
        logger.debug(f"添加到会话 {session_id} 的消息: {role} - {content[:50]}...")
    
    @log_performance
    def get_history(self, session_id: str) -> List[dict]:
        """
        获取指定会话的历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[dict]: 会话历史记录列表
        """
        history = self.histories.get(session_id, [])
        logger.debug(f"获取会话 {session_id} 的历史记录，共 {len(history)} 条")
        return history
    
    @log_performance
    def clear_history(self, session_id: str):
        """
        清除指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.histories:
            del self.histories[session_id]
            logger.info(f"已清除会话 {session_id} 的历史记录")
        else:
            logger.warning(f"尝试清除不存在的会话历史: {session_id}")

# 创建全局聊天历史管理器实例
chat_history = ChatHistory()

# 音频处理工具
class AudioProcessor:
    """
    音频数据处理工具类
    
    功能：
    - Base64编码/解码音频数据
    - 音频格式转换
    - 临时文件管理
    - 音频数据格式转换
    """
    
    @staticmethod
    @log_performance
    def base64_to_audio(base64_data: str) -> bytes:
        """
        将base64编码的音频数据转换为二进制数据
        
        Args:
            base64_data: base64编码的音频字符串
            
        Returns:
            bytes: 解码后的音频二进制数据
        """
        try:
            audio_data = base64.b64decode(base64_data)
            logger.debug(f"Base64音频解码成功，数据大小: {len(audio_data)} 字节")
            return audio_data
        except Exception as e:
            logger.error(f"Base64音频解码失败: {str(e)}")
            raise
    
    @staticmethod
    @log_performance
    def audio_to_base64(audio_data: bytes) -> str:
        """
        将音频二进制数据转换为base64编码字符串
        
        Args:
            audio_data: 音频二进制数据
            
        Returns:
            str: base64编码的音频字符串
        """
        try:
            base64_str = base64.b64encode(audio_data).decode('utf-8')
            logger.debug(f"音频Base64编码成功，字符串长度: {len(base64_str)} 字符")
            return base64_str
        except Exception as e:
            logger.error(f"音频Base64编码失败: {str(e)}")
            raise
    
    @staticmethod
    @log_performance
    def webm_to_wav(webm_data: bytes) -> str:
        """
        将WebM音频数据转换为WAV格式（用于Whisper语音识别）
        
        Args:
            webm_data: WebM格式的音频二进制数据
            
        Returns:
            str: 临时WAV文件路径
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(webm_data)
                tmp_path = tmp.name
            logger.debug(f"WebM转WAV成功，临时文件: {tmp_path}")
            return tmp_path
        except Exception as e:
            logger.error(f"WebM转WAV失败: {str(e)}")
            raise
    
    @staticmethod
    @log_performance
    def float32_to_wav(audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        将float32音频数组转换为WAV格式的二进制数据
        
        Args:
            audio_data: float32格式的音频数组
            sample_rate: 音频采样率，默认16000Hz
            
        Returns:
            bytes: WAV格式的音频二进制数据
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                # 转换为16位整数
                audio_int16 = (audio_data * 32767).astype(np.int16)
                write_wav(tmp.name, sample_rate, audio_int16)
                
                # 读取WAV文件内容
                with open(tmp.name, 'rb') as f:
                    wav_data = f.read()
                
                # 删除临时文件
                os.unlink(tmp.name)
                
            logger.debug(f"Float32转WAV成功，数据大小: {len(wav_data)} 字节")
            return wav_data
        except Exception as e:
            logger.error(f"Float32转WAV失败: {str(e)}")
            # 清理临时文件
            try:
                if 'tmp' in locals():
                    os.unlink(tmp.name)
            except:
                pass
            raise

# 创建全局音频处理器实例
audio_processor = AudioProcessor()
logger.info("音频处理器初始化完成")

# AI服务
class AIService:
    """
    AI服务类，集成多种AI功能
    
    功能：
    - 聊天API调用
    - 流式聊天响应
    - 多种TTS引擎支持
    - Whisper语音识别
    
    支持的TTS引擎：
    - gTTS: Google Text-to-Speech
    - EdgeTTS: Microsoft Edge TTS
    - ElevenLabs: 高质量语音合成
    - pyttsx3: 本地语音合成
    """
    
    def __init__(self):
        """初始化AI服务"""
        logger.info("AI服务初始化完成")
    
    @staticmethod
    @log_performance
    async def get_chat_response(messages: List[dict]) -> str:
        """
        获取聊天API的非流式响应
        
        Args:
            messages: 聊天消息列表，包含角色和内容
            
        Returns:
            str: AI生成的响应内容
        """
        try:
            logger.info(f"开始调用API，消息数量: {len(messages)}")
            
            response = await client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            logger.info(f"API调用成功，响应长度: {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"API调用失败: {str(e)}")
            return "抱歉，我遇到了一些问题。请稍后再试。"
    
    @staticmethod
    @log_performance
    async def get_chat_response_stream(messages: List[dict]) -> AsyncGenerator[str, None]:
        """
        获取聊天API的流式响应
        
        Args:
            messages: 聊天消息列表，包含角色和内容
            
        Yields:
            str: 流式响应的文本块
        """
        try:
            logger.info(f"开始调用流式API，消息数量: {len(messages)}")
            
            response = await client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7
            )
            
            chunk_count = 0
            total_length = 0
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_count += 1
                    total_length += len(content)
                    yield content
            
            logger.info(f"流式API调用完成，共 {chunk_count} 个数据块，总长度: {total_length} 字符")
            
        except Exception as e:
            logger.error(f"流式API调用失败: {str(e)}")
            yield "抱歉，我遇到了一些问题。请稍后再试。"
    
    @log_performance
    async def text_to_speech_with_engine(self, text: str, engine: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """
        使用指定TTS引擎将文本转换为语音
        
        Args:
            text: 要转换的文本内容
            engine: TTS引擎名称 ("gtts", "edgetts", "elevenlabs", "pyttsx3")
            language: 语言代码，默认中文"zh"
            max_retries: 最大重试次数
            
        Returns:
            bytes: 语音音频的二进制数据
        """
        logger.info(f"开始TTS转换，引擎: {engine}, 语言: {language}, 文本长度: {len(text)} 字符")
        
        if engine == "gtts":
            result = await self._gtts_tts(text, language, max_retries)
        elif engine == "edgetts":
            # 优先尝试EdgeTTS，失败时自动回退到gTTS
            result = await self._edgetts_tts(text, language, max_retries)
        elif engine == "elevenlabs":
            result = await self._elevenlabs_tts(text, language, max_retries)
        elif engine == "pyttsx3":
            result = await self._pyttsx3_tts(text, language, max_retries)
        else:
            logger.warning(f"未知的TTS引擎: {engine}, 使用gTTS作为后备")
            result = await self._gtts_tts(text, language, max_retries)
        
        if result and len(result) > 0:
            logger.info(f"TTS转换成功，引擎: {engine}, 音频大小: {len(result)} 字节")
        else:
            logger.warning(f"TTS转换返回空结果，引擎: {engine}")
        
        return result

    @log_performance
    async def _gtts_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """
        使用gTTS (Google Text-to-Speech) 进行语音合成
        
        Args:
            text: 要转换的文本内容
            language: 语言代码，默认中文"zh"
            max_retries: 最大重试次数
            
        Returns:
            bytes: MP3格式的音频二进制数据
        """
        logger.debug(f"开始gTTS语音合成，语言: {language}, 文本长度: {len(text)} 字符")
        
        for attempt in range(max_retries):
            try:
                # 创建gTTS实例
                tts = gTTS(text=text, lang=language, slow=False)
                
                # 创建临时文件保存音频
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # 保存音频到文件
                tts.save(tmp_path)
                
                # 读取文件内容
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                # 删除临时文件
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {str(e)}")
                
                logger.info(f"gTTS语音合成成功 (第 {attempt + 1} 次尝试)，音频大小: {len(audio_data)} 字节")
                return audio_data
                
            except Exception as e:
                logger.warning(f"gTTS语音合成第 {attempt + 1} 次尝试失败: {str(e)}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 等待1秒后重试
                    logger.debug(f"等待1秒后重试gTTS...")
                else:
                    # 最后一次尝试失败，清理临时文件
                    try:
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {str(e)}")
        
        logger.error(f"gTTS语音合成所有 {max_retries} 次尝试都失败")
        return b""
    
    @log_performance
    async def _edgetts_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """
        使用EdgeTTS (Microsoft Edge Text-to-Speech) 进行语音合成
        
        Args:
            text: 要转换的文本内容
            language: 语言代码，默认中文"zh"
            max_retries: 最大重试次数
            
        Returns:
            bytes: MP3格式的音频二进制数据
        """
        logger.debug(f"开始EdgeTTS语音合成，语言: {language}, 文本长度: {len(text)} 字符")
        
        # 检查是否安装了edge-tts
        try:
            import edge_tts
            logger.debug("EdgeTTS模块导入成功")
        except ImportError:
            logger.warning("edge-tts未安装，使用pip install edge-tts安装，回退到gTTS")
            return await self._gtts_tts(text, language, max_retries)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                # 根据语言选择适当的语音
                if language == "zh":
                    voice = "zh-CN-XiaoxiaoNeural"  # 中文普通话女性语音
                elif language == "en":
                    voice = "en-US-AriaNeural"      # 英文女性语音
                else:
                    voice = "zh-CN-XiaoxiaoNeural"   # 默认中文语音
                
                logger.debug(f"EdgeTTS使用语音: {voice}")
                communicate = edge_tts.Communicate(text, voice)
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # 设置超时并保存音频
                try:
                    await asyncio.wait_for(communicate.save(tmp_path), timeout=30.0)
                    logger.debug("EdgeTTS音频保存完成")
                except asyncio.TimeoutError:
                    logger.warning(f"EdgeTTS超时 (第 {attempt + 1}/{max_retries} 次尝试)")
                    continue
                
                # 读取音频数据
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"删除EdgeTTS临时文件失败: {str(e)}")
                
                # 检查音频数据是否有效
                if len(audio_data) > 100:  # 确保有足够的音频数据
                    logger.info(f"EdgeTTS语音合成成功 (第 {attempt + 1} 次尝试)，音频大小: {len(audio_data)} 字节")
                    return audio_data
                else:
                    logger.warning(f"EdgeTTS生成空音频 (第 {attempt + 1}/{max_retries} 次尝试)")
                    
            except Exception as e:
                logger.warning(f"EdgeTTS第 {attempt + 1}/{max_retries} 次尝试失败: {str(e)}")
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # 等待2秒后重试
                    logger.debug("等待2秒后重试EdgeTTS...")
                
                # 清理临时文件
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"清理EdgeTTS临时文件失败: {str(e)}")
        
        logger.error(f"EdgeTTS所有 {max_retries} 次尝试都失败，回退到gTTS")
        return await self._gtts_tts(text, language, max_retries)  # 失败时回退到gTTS
    
    async def _pyttsx3_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用pyttsx3进行本地语音合成（无需网络）"""
        for attempt in range(max_retries):
            try:
                import pyttsx3
                # 初始化引擎
                engine = pyttsx3.init()
             
                # 设置语音属性
                if language == "zh":
                    # 设置中文语音 - 以婷婷为例
                    engine.setProperty('voice', 'com.apple.voice.compact.zh-CN.Tingting')
                elif language == "en":
                    # 设置英文语音 - 以Samantha为例
                    engine.setProperty('voice', 'com.apple.voice.compact.en-US.Samantha')
                
                # 设置语速和音量
                engine.setProperty('rate', 180)  # 适中语速
                engine.setProperty('volume', 0.9)  # 较高音量
                
                # 保存到临时文件 - 使用MP3格式
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # 保存语音到文件
                engine.save_to_file(text, tmp_path)
                engine.runAndWait()
                
                # 读取文件内容
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if len(audio_data) > 100:
                    logger.info(f"pyttsx3 TTS成功 (尝试 {attempt + 1})")
                    return audio_data
                else:
                    logger.warning(f"pyttsx3生成空音频 (尝试 {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # 等待1秒后重试
                    
            except ImportError:
                logger.warning("pyttsx3未安装，使用pip install pyttsx3安装")
                return await self._gtts_tts(text, language, max_retries)
                
            except Exception as e:
                logger.error(f"pyttsx3 TTS失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 等待1秒后重试
                
                # 清理可能的临时文件
                try:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
        
        # 所有尝试都失败，回退到gTTS
        logger.error("pyttsx3所有尝试都失败，回退到gTTS")
        return await self._gtts_tts(text, language, max_retries)
    
    async def _elevenlabs_tts(self, text: str, language: str = "zh", max_retries: int = 3) -> bytes:
        """使用ElevenLabs进行语音合成（需要API密钥）"""
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if not elevenlabs_api_key:
            logger.warning("ElevenLabs API密钥未设置，使用gTTS作为后备")
            return await self._gtts_tts(text, language, max_retries)
        
        # 根据语言选择语音 - 使用多语言模型时，语音选择更重要
        voice_id = self._get_elevenlabs_voice_id(language)
        
        # 重试机制 - 立即重试，无等待
        for attempt in range(max_retries):
            try:
                # 使用同步HTTP请求
                import requests
                
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": elevenlabs_api_key
                }
                
                data = {
                    "text": text,
                    "model_id": "eleven_v3",  # 使用多语言模型，自动处理语言检测
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.5,
                        "use_speaker_boost": True
                    }
                }
                
                # 同步请求，5秒超时
                response = requests.post(url, json=data, headers=headers, timeout=5.0)
                
                if response.status_code == 200:
                    logger.info(f"ElevenLabs TTS成功 (尝试 {attempt + 1}/{max_retries})")
                    return response.content
                else:
                    logger.warning(f"ElevenLabs API错误 (尝试 {attempt + 1}/{max_retries}): {response.status_code} - {response.text[:200]}")
                    # 立即重试下一次
                    continue
                    
            except requests.Timeout:
                logger.warning(f"ElevenLabs TTS超时 (尝试 {attempt + 1}/{max_retries})")
                # 立即重试下一次
                continue
                
            except requests.RequestException as e:
                logger.error(f"ElevenLabs网络错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                # 立即重试下一次
                continue
                
            except Exception as e:
                logger.error(f"ElevenLabs TTS未知错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                # 立即重试下一次
                continue
        
        logger.error("ElevenLabs所有尝试都失败，回退到gTTS")
        return await self._gtts_tts(text, language, max_retries)
    
    def _get_elevenlabs_voice_id(self, language: str) -> str:
        """根据语言获取ElevenLabs语音ID"""
        # 常用语音ID映射 - 为不同语言选择更适合的语音
        voice_mapping = {
            "zh": "21m00Tcm4TlvDq8ikWAM",  # Rachel - 英语女性（支持中文发音）
            "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel - 英语女性
            "ja": "AZnzlPK1z2Yr1q1dQxQa",  # Dorothy - 日语女性
            "ko": "XrExE9yKIg1WjnnlVkGX",  # Lily - 韩语女性
            "es": "MF3mGyEYCl7XYWbV9V6O",  # Emily - 西班牙语女性
            "fr": "N2lVS1w4EtoT3dr4eOWO",  # Charlotte - 法语女性
            "de": "ThT5KcBeYPX3keUQqHPh",  # Sarah - 德语女性
            "it": "pNInz6obpgDQGcFmaJgB",  # Grace - 意大利语女性
            "pt": "IKne3meq5aSn9XLyUdCD",  # Nicole - 葡萄牙语女性
            "ru": "VR6AewLTigWG4xSOukaG",  # Dasha - 俄语女性
            "ar": "21m00Tcm4TlvDq8ikWAM",  # Rachel - 英语女性（阿拉伯语支持有限）
            "hi": "21m00Tcm4TlvDq8ikWAM"   # Rachel - 英语女性（印地语支持有限）
        }
        
        # 获取环境变量中配置的语音ID（如果存在）
        custom_voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        if custom_voice_id:
            return custom_voice_id
            
        return voice_mapping.get(language, "21m00Tcm4TlvDq8ikWAM")  # 默认语音
                
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
    return FileResponse("index.html")

# WebSocket端点 - 文本和语音聊天
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    聊天WebSocket端点 - 处理文本和音频消息的AI聊天功能
    
    支持的消息类型:
    - text: 文本消息，直接获取AI响应
    - audio: 音频消息，先进行语音识别再获取AI响应
    - clear: 清空指定会话的聊天历史
    """
    session_id = f"chat_{id(websocket)}"
    logger.info(f"新的聊天WebSocket连接建立，客户端: {websocket.client}")
    await manager.connect(websocket, "chat")
    
    try:
        while True:
            # 接收消息
            logger.debug("等待接收WebSocket消息...")
            data = await websocket.receive_json()
            message_type = data.get("type")
            message_id =data.get("message_id")
            logger.debug(f"收到WebSocket消息，类型: {message_type}")
            await manager.send_json(websocket, {"type": "begin"})
            logger.info(f"会话开始")
            if message_type == "text":
                # 处理文本消息
                user_message = data.get("message", "")
                language = data.get("language", "zh")
                
                logger.info(f"处理文本消息，会话ID: {session_id}, 语言: {language}, 文本长度: {len(user_message)} 字符")
                
                if user_message:
                    # 添加到历史
                    chat_history.add_message(session_id, "user", user_message)
                    
                    # 获取AI响应（流式）
                    messages = [
                        {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["chat"]},
                        *chat_history.get_history(session_id)
                    ]
                    
                    # 发送流式响应
                    full_response = ""
                    # 先发送start状态
                    await manager.send_json(websocket, {
                        "type": "text",
                        "content": "",
                        "status": "start",
                        "role": "assistant",
                        "message_id":message_id
                    })
                    
                    async for chunk in ai_service.get_chat_response_stream(messages):
                        # 发送continue状态
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": chunk,
                            "status": "continue",
                            "role": "assistant",
                            "message_id":message_id
                        })
                        full_response += chunk
                    
                    # 发送结束状态（如果有内容）
                    if full_response:
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": "",
                            "status": "end",
                            "role": "assistant",
                            "message_id":message_id
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
                                "content": audio_processor.audio_to_base64(audio_data),
                                "status": "end",
                                "role": "assistant",
                                "message_id":message_id
                            })
                    
                else:
                    logger.warning("收到空文本消息")
                    await manager.send_json(websocket, {
                        "message_id":message_id,
                        "type": "error",
                        "content": "收到空文本消息"
                    })
            
            elif message_type == "audio":
                # 处理音频消息
                audio_base64 = data.get("audio_data", "")
                language = data.get("language", "zh")
                
                logger.info(f"处理音频消息，会话ID: {session_id}, 语言: {language}, 音频数据大小: {len(audio_base64)} 字节")
                
                if audio_base64:
                     # 先发送start状态
                    await manager.send_json(websocket, {
                        "type": "text",
                        "content": "",
                        "status": "start",
                        "role": "user",
                        "message_id":message_id
                    })
                    audio_data = audio_processor.base64_to_audio(audio_base64)
                    # 保存音频文件
                    audio_file = audio_processor.webm_to_wav(audio_data)
                    
                    # 语音转文本
                    transcribed_text = await ai_service.speech_to_text(audio_file)
                    os.unlink(audio_file)
                    
                    if transcribed_text and len(transcribed_text.strip()) > 1:
                        logger.info(f"语音识别结果: {transcribed_text}")
                        # 发送转录文本给前端
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": transcribed_text,
                            "status": "end",
                            "role": "user",
                            "message_id":message_id
                        })
                        
                        # 添加到历史
                        chat_history.add_message(session_id, "user", transcribed_text)
                        
                        # 获取AI响应
                        messages = [
                            {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["chat"]},
                            *chat_history.get_history(session_id)[-6:]  # 只保留最近的3轮对话
                        ]
                        
                        # 发送流式响应
                        full_response = ""
                        # 先发送start状态
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": "",
                            "status": "start",
                            "role": "assistant",
                            "message_id":message_id
                        })
                        
                        async for chunk in ai_service.get_chat_response_stream(messages):
                            # 发送continue状态
                            await manager.send_json(websocket, {
                                "type": "text",
                                "content": chunk,
                                "status": "continue",
                                "role": "assistant",
                                "message_id":message_id
                            })
                            full_response += chunk
                        
                        # 发送结束状态（如果有内容）
                        if full_response:
                            await manager.send_json(websocket, {
                                "type": "text",
                                "content": "",
                                "status": "end",
                                "role": "assistant",
                                "message_id":message_id
                            })
                        
                        # 添加到历史
                        chat_history.add_message(session_id, "assistant", full_response)
                        
                        # 生成语音响应
                        tts_engine = data.get("tts_engine", "gtts")
                        audio_response = await ai_service.text_to_speech_with_engine(full_response, tts_engine, language)
                        if audio_response:
                            await manager.send_json(websocket, {
                                "type": "audio",
                                "content": audio_processor.audio_to_base64(audio_response),
                                "status": "end",
                                "role": "assistant",
                                "message_id":message_id
                            })
                        
                        logger.info(f"音频消息响应发送完成，识别文本长度: {len(transcribed_text)} 字符，响应长度: {len(full_response)} 字符")
                    else:
                        # 语音识别失败
                        logger.warning("语音识别失败")
                        await manager.send_json(websocket, {
                            "type": "error",
                            "content": "语音识别失败",
                            "message_id":message_id
                        })
                else:
                    logger.warning("收到空音频消息")
                    await manager.send_json(websocket, {
                        "type": "error",
                        "content": "收到空音频消息",
                        "message_id":message_id
                    })
            
            elif message_type == "clear":
                # 清空聊天历史
                logger.info(f"清空聊天历史，会话ID: {session_id}")
                chat_history.clear_history(session_id)
                
                await manager.send_json(websocket, {
                    "type": "clear_confirm",
                    "content": "聊天历史已清空",
                    "message_id":message_id
                })
                logger.info("聊天历史清空确认发送完成")
            else:
                logger.warning(f"未知的消息类型: {message_type}")
                await manager.send_json(websocket, {
                    "type": "error",
                    "content": f"未知的消息类型: {message_type}",
                    "message_id":message_id
                })
            await manager.send_json(websocket, {"type": "complete","message_id":message_id})
            logger.info(f"会话 {message_id} 结束")

    except WebSocketDisconnect:
        logger.info("WebSocket连接断开")
        manager.disconnect(websocket, "chat")
        chat_history.clear_history(session_id)
        logger.info(f"Chat session {session_id} ended")
    except Exception as e:
        logger.error(f"Chat websocket error: {str(e)}")
        manager.disconnect(websocket, "chat")

# WebSocket端点 - 实时语音对话
@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    实时语音对话WebSocket端点 - 处理VAD检测到的语音片段
    
    处理流程:
    1. 接收音频数据块（VAD检测到的语音片段）
    2. 解码base64音频数据并转换为WAV格式
    3. 进行语音识别获取文本
    4. 发送转录文本给前端显示
    5. 获取AI文本响应（流式）
    6. 将响应文本转换为语音并发送给客户端
    """
    session_id = f"voice_{id(websocket)}"
    logger.info(f"新的语音WebSocket连接建立，客户端: {websocket.client}, 会话ID: {session_id}")
    await manager.connect(websocket, "voice")
    
    try:
        audio_chunk_count = 0
        while True:
            # 接收消息
            logger.debug("等待接收语音WebSocket消息...")
            data = await websocket.receive_json()
            await manager.send_json(websocket, {"type": "begin"})
            logger.info(f"会话开始")
            message_type = data.get("type")
            message_id = data.get("message_id")
            audio_chunk_count += 1
            
            logger.debug(f"收到语音WebSocket消息 #{audio_chunk_count}, 类型: {message_type}")
            
            if message_type == "audio":
                # 前端发送的音频数据块（VAD检测到的语音片段）
                audio_base64 = data.get("audio_data", "")
                language = data.get("language", "zh")
                
                logger.info(f"处理音频数据块 #{audio_chunk_count}, 语言: {language}, 数据大小: {len(audio_base64)} 字节")
                
                if audio_base64:
                     # 先发送start状态
                    await manager.send_json(websocket, {
                        "type": "text",
                        "content": "",
                        "status": "start",
                        "role": "user",
                        "message_id":message_id
                    })
                    # 解码base64音频数据
                    audio_data = audio_processor.base64_to_audio(audio_base64)
                    logger.debug("音频数据base64解码完成")
                    
                    # 保存音频文件
                    audio_file = audio_processor.webm_to_wav(audio_data)
                    logger.debug("音频文件转换完成")
                    
                    # 语音识别
                    transcribed_text = await ai_service.speech_to_text(audio_file)
                    logger.debug(f"语音识别结果 (块 #{audio_chunk_count}): {transcribed_text}")
                    os.unlink(audio_file)
                    
                    if transcribed_text and len(transcribed_text.strip()) > 1:
                        logger.info(f"语音识别成功 (块 #{audio_chunk_count}): {transcribed_text}")
                        
                        # 发送转录文本
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": transcribed_text,
                            "status": "end",
                            "role": "user",
                            "message_id":message_id
                        })
                        logger.debug("转录文本发送完成")
                        
                        # 添加到历史
                        chat_history.add_message(session_id, "user", transcribed_text)
                        logger.debug("用户消息添加到历史记录")
                        
                        # 获取AI响应
                        messages = [
                            {"role": "system", "content": config.SYSTEM_PROMPTS.get(language, config.SYSTEM_PROMPTS["zh"])["voice"]},
                            *chat_history.get_history(session_id)[-6:]  # 只保留最近的3轮对话
                        ]
                        logger.debug(f"构建AI消息上下文，消息数量: {len(messages)}")
                        
                        # 发送流式响应
                        full_response = ""
                        # 先发送start状态
                        await manager.send_json(websocket, {
                            "type": "text",
                            "content": "",
                            "status": "start",
                            "role": "assistant",
                            "message_id":message_id
                        })
                        logger.debug("流式响应开始状态发送完成")
                        
                        async for chunk in ai_service.get_chat_response_stream(messages):
                            # 发送continue状态
                            await manager.send_json(websocket, {
                                "type": "text",
                                "content": chunk,
                                "status": "continue",
                                "role": "assistant",
                                "message_id":message_id
                            })
                            full_response += chunk
                        
                        # 发送结束状态（如果有内容）
                        if full_response:
                            await manager.send_json(websocket, {
                                "type": "text",
                                "content": "",
                                "status": "end",
                                "role": "assistant",
                                "message_id":message_id
                            })
                        logger.debug("流式响应结束状态发送完成")
                        
                        # 添加到历史
                        chat_history.add_message(session_id, "assistant", full_response)
                        logger.debug("AI响应添加到历史记录")
                        
                        # 生成语音响应
                        tts_engine = data.get("tts_engine", "gtts")
                        audio_response = await ai_service.text_to_speech_with_engine(full_response, tts_engine, language)
                        
                        if audio_response:
                            await manager.send_json(websocket, {
                                "type": "audio",
                                "content": audio_processor.audio_to_base64(audio_response),
                                "status": "end",
                                "role": "assistant",
                                "message_id":message_id
                            })
                            logger.info(f"语音响应发送完成，音频大小: {len(audio_response)} 字节")
                        else:
                            logger.warning("语音合成失败，无法生成音频响应")
                        
                    else:
                        logger.warning(f"语音识别失败或识别文本过短 (块 #{audio_chunk_count})，识别文本: {transcribed_text}")
                        await manager.send_json(websocket, {
                            "type": "error",
                            "content": "语音识别失败",
                            "message_id":message_id
                        })
                else:
                    logger.warning(f"收到空音频数据块 (块 #{audio_chunk_count})")
                    await manager.send_json(websocket, {
                        "type": "error",
                        "content": "收到空音频数据",
                        "message_id":message_id
                    })
            else:
                logger.warning(f"未知的语音消息类型: {message_type} (块 #{audio_chunk_count})")
                await manager.send_json(websocket, {
                    "type": "error",
                    "content": f"未知的消息类型: {message_type}",
                    "message_id":message_id
                })
                # 发送完成信号
                await manager.send_json(websocket, {"type": "complete", "message_id":message_id})
                 
    except WebSocketDisconnect:
        logger.info(f"语音WebSocket连接断开，会话ID: {session_id}")
        manager.disconnect(websocket, "voice")
        chat_history.clear_history(session_id)
        logger.info(f"语音会话 {session_id} 已断开连接并清空历史")
    except Exception as e:
        logger.error(f"语音WebSocket错误: {str(e)}", exc_info=True)
        manager.disconnect(websocket, "voice")

# 健康检查
@app.get("/health")
async def health_check():
    """
    健康检查端点 - 检查服务器状态和连接信息
    
    Returns:
        dict: 包含服务器状态、时间戳和连接统计信息
    """
    logger.debug("健康检查端点被调用")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": {
            "chat": len(manager.active_connections["chat"]),
            "voice": len(manager.active_connections["voice"])
        }
    }
    
    logger.info(f"健康检查完成，活跃连接数 - 聊天: {health_status['connections']['chat']}, 语音: {health_status['connections']['voice']}")
    return health_status

# 主函数
if __name__ == "__main__":
    """
    应用程序主入口点
    
    启动流程:
    1. 检查HTML文件是否存在
    2. 配置并启动uvicorn服务器
    3. 监听0.0.0.0:8000端口
    4. 启用热重载功能
    """
    import uvicorn
    
    logger.info("开始启动AI语音聊天服务器...")
    
    # 确保HTML文件存在
    html_file_path = "index.html"
    if not os.path.exists(html_file_path):
        logger.warning(f"HTML文件未找到: {html_file_path}，请确保已创建HTML文件")
    else:
        logger.info("HTML文件检查通过")
    
    # 启动服务器
    logger.info("正在启动uvicorn服务器...")
    logger.info("服务器配置 - 主机: 0.0.0.0, 端口: 8000, 日志级别: info")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        logger.info("服务器启动成功，正在监听连接...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        raise
