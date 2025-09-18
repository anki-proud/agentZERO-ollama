"""
Ollama API Client for agentZERO
Handles communication with local Ollama instance
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
import aiohttp
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about an Ollama model"""
    name: str
    size: int
    parameters: str
    family: str
    format: str
    modified_at: str

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    stop: Optional[List[str]] = None
    stream: bool = False

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def list_models(self) -> List[ModelInfo]:
        """List all available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model in data.get('models', []):
                            models.append(ModelInfo(
                                name=model['name'],
                                size=model['size'],
                                parameters=model.get('details', {}).get('parameter_size', 'unknown'),
                                family=model.get('details', {}).get('family', 'unknown'),
                                format=model.get('details', {}).get('format', 'unknown'),
                                modified_at=model['modified_at']
                            ))
                        return models
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": model_name}
                async with session.post(
                    f"{self.base_url}/api/pull", 
                    json=payload
                ) as response:
                    if response.status == 200:
                        # Stream the pull progress
                        async for line in response.content:
                            if line:
                                try:
                                    progress = json.loads(line.decode())
                                    if 'status' in progress:
                                        logger.info(f"Pull progress: {progress['status']}")
                                except json.JSONDecodeError:
                                    continue
                        return True
                    else:
                        logger.error(f"Failed to pull model {model_name}: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        config: GenerationConfig = None
    ) -> str:
        """Generate text using specified model"""
        if config is None:
            config = GenerationConfig()
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                }
            }
            
            if config.stop:
                payload["options"]["stop"] = config.stop
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        error_text = await response.text()
                        logger.error(f"Generation failed: {response.status} - {error_text}")
                        return f"Error: Generation failed with status {response.status}"
        except asyncio.TimeoutError:
            logger.error("Generation timed out")
            return "Error: Generation timed out"
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"
    
    async def generate_stream(
        self, 
        model: str, 
        prompt: str, 
        config: GenerationConfig = None
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response"""
        if config is None:
            config = GenerationConfig()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens,
            }
        }
        
        if config.stop:
            payload["options"]["stop"] = config.stop
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    chunk = json.loads(line.decode())
                                    if 'response' in chunk:
                                        yield chunk['response']
                                    if chunk.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Streaming generation failed: {response.status} - {error_text}")
                        yield f"Error: Streaming failed with status {response.status}"
        except Exception as e:
            logger.error(f"Error during streaming generation: {e}")
            yield f"Error: {str(e)}"
    
    async def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        config: GenerationConfig = None
    ) -> str:
        """Chat completion using specified model"""
        if config is None:
            config = GenerationConfig()
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('message', {}).get('content', '')
                    else:
                        error_text = await response.text()
                        logger.error(f"Chat failed: {response.status} - {error_text}")
                        return f"Error: Chat failed with status {response.status}"
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists locally"""
        models = await self.list_models()
        return any(model.name == model_name for model in models)

# Convenience functions for synchronous usage
def sync_generate(model: str, prompt: str, config: GenerationConfig = None) -> str:
    """Synchronous wrapper for generate"""
    async def _generate():
        async with OllamaClient() as client:
            return await client.generate(model, prompt, config)
    
    return asyncio.run(_generate())

def sync_chat(model: str, messages: List[Dict[str, str]], config: GenerationConfig = None) -> str:
    """Synchronous wrapper for chat"""
    async def _chat():
        async with OllamaClient() as client:
            return await client.chat(model, messages, config)
    
    return asyncio.run(_chat())

# Example usage
if __name__ == "__main__":
    async def main():
        async with OllamaClient() as client:
            # Check if service is available
            if not client.is_available():
                print("Ollama service is not available")
                return
            
            # List available models
            models = await client.list_models()
            print(f"Available models: {[m.name for m in models]}")
            
            # Generate text
            if models:
                model_name = models[0].name
                response = await client.generate(
                    model_name, 
                    "Write a short story about a robot learning to paint."
                )
                print(f"Response: {response}")
    
    asyncio.run(main())

