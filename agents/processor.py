"""
Processor Agent for agentZERO
Specialized agent for prompt processing, optimization, and enhancement
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from core.ollama_client import OllamaClient, GenerationConfig
from core.prompt_engine import PromptEngine, PromptMetrics

logger = logging.getLogger(__name__)

class ProcessingType(Enum):
    OPTIMIZATION = "optimization"
    ENHANCEMENT = "enhancement"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXPANSION = "expansion"
    CLARIFICATION = "clarification"
    VALIDATION = "validation"

class PromptQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ProcessingRequest:
    """Request for prompt processing"""
    id: str
    original_prompt: str
    processing_type: ProcessingType
    target_model: Optional[str] = None
    target_audience: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingResult:
    """Result of prompt processing"""
    request_id: str
    original_prompt: str
    processed_prompt: str
    processing_type: ProcessingType
    quality_score: float
    improvements: List[str]
    warnings: List[str]
    metrics: PromptMetrics
    processing_time: float
    model_used: str
    created_at: datetime

class ProcessorAgent:
    """Agent specialized in prompt processing and optimization"""
    
    def __init__(self, model: str = "smollm:135m", ollama_url: str = "http://localhost:11434"):
        self.model = model  # Use fastest model for prompt processing
        self.ollama_client = OllamaClient(ollama_url)
        self.prompt_engine = PromptEngine()
        
        # Processing configuration
        self.max_iterations = 3
        self.quality_threshold = 0.7
        
        # System prompts for different processing types
        self.system_prompts = {
            ProcessingType.OPTIMIZATION: """You are a prompt optimization expert. Your task is to improve prompts to be more effective, clear, and efficient while preserving the original intent. Focus on:
- Clarity and specificity
- Removing ambiguity
- Optimizing for the target model
- Maintaining conciseness
- Improving structure""",
            
            ProcessingType.ENHANCEMENT: """You are a prompt enhancement specialist. Your task is to enrich prompts with additional context, examples, and guidance to improve output quality. Focus on:
- Adding helpful context
- Including relevant examples
- Providing clearer instructions
- Specifying desired format
- Adding quality criteria""",
            
            ProcessingType.TRANSLATION: """You are a prompt translation expert. Your task is to adapt prompts for different contexts, audiences, or technical levels while maintaining effectiveness. Focus on:
- Adjusting language complexity
- Adapting to target audience
- Maintaining core intent
- Cultural sensitivity
- Technical appropriateness""",
            
            ProcessingType.SUMMARIZATION: """You are a prompt summarization expert. Your task is to condense prompts while retaining essential information and effectiveness. Focus on:
- Identifying core requirements
- Removing redundancy
- Maintaining clarity
- Preserving key instructions
- Optimizing length""",
            
            ProcessingType.EXPANSION: """You are a prompt expansion expert. Your task is to elaborate on brief prompts to provide more comprehensive guidance. Focus on:
- Adding necessary details
- Clarifying expectations
- Providing structure
- Including examples
- Specifying criteria""",
            
            ProcessingType.CLARIFICATION: """You are a prompt clarification expert. Your task is to identify and resolve ambiguities in prompts. Focus on:
- Identifying unclear elements
- Resolving ambiguities
- Making implicit requirements explicit
- Improving precision
- Ensuring consistency""",
            
            ProcessingType.VALIDATION: """You are a prompt validation expert. Your task is to analyze prompts for potential issues and suggest improvements. Focus on:
- Identifying problems
- Assessing effectiveness
- Checking completeness
- Evaluating clarity
- Suggesting fixes"""
        }
    
    async def process_prompt(self, request: ProcessingRequest) -> ProcessingResult:
        """Process a prompt according to the specified type"""
        logger.info(f"Processing prompt {request.id} with type {request.processing_type.value}")
        
        start_time = datetime.now()
        
        try:
            # Analyze original prompt
            original_metrics = self.prompt_engine.analyze_prompt(
                request.original_prompt, 
                request.target_model or "smollm:360m"
            )
            
            # Process based on type
            if request.processing_type == ProcessingType.OPTIMIZATION:
                result = await self._optimize_prompt(request)
            elif request.processing_type == ProcessingType.ENHANCEMENT:
                result = await self._enhance_prompt(request)
            elif request.processing_type == ProcessingType.TRANSLATION:
                result = await self._translate_prompt(request)
            elif request.processing_type == ProcessingType.SUMMARIZATION:
                result = await self._summarize_prompt(request)
            elif request.processing_type == ProcessingType.EXPANSION:
                result = await self._expand_prompt(request)
            elif request.processing_type == ProcessingType.CLARIFICATION:
                result = await self._clarify_prompt(request)
            elif request.processing_type == ProcessingType.VALIDATION:
                result = await self._validate_prompt(request)
            else:
                raise ValueError(f"Unknown processing type: {request.processing_type}")
            
            # Analyze processed prompt
            processed_metrics = self.prompt_engine.analyze_prompt(
                result.processed_prompt,
                request.target_model or "smollm:360m"
            )
            
            # Calculate quality score and improvements
            quality_score = self._calculate_quality_score(original_metrics, processed_metrics)
            improvements = self._identify_improvements(original_metrics, processed_metrics)
            warnings = self._identify_warnings(processed_metrics)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return ProcessingResult(
                request_id=request.id,
                original_prompt=request.original_prompt,
                processed_prompt=result.processed_prompt,
                processing_type=request.processing_type,
                quality_score=quality_score,
                improvements=improvements,
                warnings=warnings,
                metrics=processed_metrics,
                processing_time=processing_time,
                model_used=self.model,
                created_at=end_time
            )
            
        except Exception as e:
            logger.error(f"Error processing prompt {request.id}: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Return original prompt with error info
            return ProcessingResult(
                request_id=request.id,
                original_prompt=request.original_prompt,
                processed_prompt=request.original_prompt,
                processing_type=request.processing_type,
                quality_score=0.0,
                improvements=[],
                warnings=[f"Processing failed: {str(e)}"],
                metrics=self.prompt_engine.analyze_prompt(request.original_prompt),
                processing_time=processing_time,
                model_used=self.model,
                created_at=end_time
            )
    
    async def _optimize_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Optimize a prompt for better performance"""
        optimization_prompt = f"""
Original prompt: "{request.original_prompt}"

Target model: {request.target_model or 'smollm:360m'}
Target audience: {request.target_audience or 'general'}

Constraints:
{json.dumps(request.constraints, indent=2) if request.constraints else 'None specified'}

Please optimize this prompt to be more effective. Consider:
1. Clarity and specificity
2. Removing ambiguity and redundancy
3. Optimizing for the target model's capabilities
4. Improving structure and flow
5. Maintaining the original intent

Provide only the optimized prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.OPTIMIZATION, 
            optimization_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _enhance_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Enhance a prompt with additional context and guidance"""
        enhancement_prompt = f"""
Original prompt: "{request.original_prompt}"

Target model: {request.target_model or 'smollm:360m'}
Context: {json.dumps(request.context, indent=2) if request.context else 'None provided'}

Please enhance this prompt by adding:
1. Helpful context and background
2. Clear instructions and expectations
3. Examples where appropriate
4. Desired output format specification
5. Quality criteria or success metrics

The enhanced prompt should be more comprehensive while remaining focused.

Provide only the enhanced prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.ENHANCEMENT, 
            enhancement_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _translate_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Translate/adapt a prompt for different context or audience"""
        translation_prompt = f"""
Original prompt: "{request.original_prompt}"

Target audience: {request.target_audience or 'general'}
Adaptation requirements: {json.dumps(request.constraints, indent=2) if request.constraints else 'None specified'}

Please adapt this prompt for the target audience by:
1. Adjusting language complexity appropriately
2. Using terminology suitable for the audience
3. Maintaining the core intent and requirements
4. Ensuring cultural and contextual appropriateness
5. Optimizing for the target use case

Provide only the adapted prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.TRANSLATION, 
            translation_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _summarize_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Summarize a prompt while retaining essential information"""
        summarization_prompt = f"""
Original prompt: "{request.original_prompt}"

Target length: {request.constraints.get('max_length', 'as concise as possible')}

Please create a concise version of this prompt that:
1. Retains all essential requirements
2. Removes redundancy and unnecessary details
3. Maintains clarity and effectiveness
4. Preserves the core intent
5. Optimizes for brevity without losing meaning

Provide only the summarized prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.SUMMARIZATION, 
            summarization_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _expand_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Expand a brief prompt with more comprehensive guidance"""
        expansion_prompt = f"""
Original prompt: "{request.original_prompt}"

Context: {json.dumps(request.context, indent=2) if request.context else 'None provided'}

Please expand this brief prompt into a more comprehensive version that:
1. Provides detailed instructions and guidance
2. Includes relevant examples or templates
3. Specifies expected output format and quality
4. Adds helpful context and background
5. Clarifies any implicit requirements

Provide only the expanded prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.EXPANSION, 
            expansion_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _clarify_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Clarify ambiguous elements in a prompt"""
        clarification_prompt = f"""
Original prompt: "{request.original_prompt}"

Please analyze this prompt and create a clarified version that:
1. Identifies and resolves ambiguities
2. Makes implicit requirements explicit
3. Improves precision and specificity
4. Ensures consistency throughout
5. Eliminates potential misinterpretations

Focus on making the prompt as clear and unambiguous as possible.

Provide only the clarified prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.CLARIFICATION, 
            clarification_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _validate_prompt(self, request: ProcessingRequest) -> 'ProcessingResult':
        """Validate a prompt and suggest improvements"""
        validation_prompt = f"""
Original prompt: "{request.original_prompt}"

Target model: {request.target_model or 'smollm:360m'}

Please analyze this prompt and create an improved version that addresses any issues you identify:
1. Check for completeness and clarity
2. Identify potential problems or ambiguities
3. Assess effectiveness for the target model
4. Suggest structural improvements
5. Ensure all necessary elements are present

Provide only the improved prompt without additional explanation.
"""
        
        processed_prompt = await self._generate_response(
            ProcessingType.VALIDATION, 
            validation_prompt
        )
        
        return type('Result', (), {'processed_prompt': processed_prompt.strip()})()
    
    async def _generate_response(self, processing_type: ProcessingType, prompt: str) -> str:
        """Generate response using the appropriate system prompt"""
        system_prompt = self.system_prompts[processing_type]
        
        config = GenerationConfig(
            temperature=0.3,  # Lower temperature for more consistent processing
            max_tokens=1024,
            top_p=0.9
        )
        
        messages = self.prompt_engine.create_chat_messages(system_prompt, prompt)
        
        async with self.ollama_client as client:
            response = await client.chat(self.model, messages, config)
        
        return response
    
    def _calculate_quality_score(self, original: PromptMetrics, processed: PromptMetrics) -> float:
        """Calculate quality improvement score"""
        # Factors that contribute to quality
        readability_improvement = processed.readability_score - original.readability_score
        complexity_optimization = abs(0.5 - original.complexity_score) - abs(0.5 - processed.complexity_score)
        
        # Model compatibility improvement
        avg_original_compatibility = sum(original.model_compatibility.values()) / len(original.model_compatibility)
        avg_processed_compatibility = sum(processed.model_compatibility.values()) / len(processed.model_compatibility)
        compatibility_improvement = avg_processed_compatibility - avg_original_compatibility
        
        # Combine factors (weighted)
        quality_score = (
            readability_improvement * 0.3 +
            complexity_optimization * 0.3 +
            compatibility_improvement * 0.4
        )
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, 0.5 + quality_score))
    
    def _identify_improvements(self, original: PromptMetrics, processed: PromptMetrics) -> List[str]:
        """Identify specific improvements made"""
        improvements = []
        
        if processed.readability_score > original.readability_score + 0.1:
            improvements.append("Improved readability and clarity")
        
        if processed.token_count < original.token_count * 0.8:
            improvements.append("Reduced token count for efficiency")
        elif processed.token_count > original.token_count * 1.2:
            improvements.append("Added comprehensive details and context")
        
        # Check model compatibility improvements
        for model in processed.model_compatibility:
            if processed.model_compatibility[model] > original.model_compatibility.get(model, 0) + 0.1:
                improvements.append(f"Improved compatibility with {model}")
        
        if abs(processed.complexity_score - 0.5) < abs(original.complexity_score - 0.5):
            improvements.append("Optimized complexity level")
        
        return improvements
    
    def _identify_warnings(self, metrics: PromptMetrics) -> List[str]:
        """Identify potential issues with the processed prompt"""
        warnings = []
        
        if metrics.token_count > 1000:
            warnings.append("Prompt may be too long for smaller models")
        
        if metrics.complexity_score > 0.8:
            warnings.append("High complexity may challenge smaller models")
        
        if metrics.readability_score < 0.3:
            warnings.append("Low readability may cause confusion")
        
        # Check model compatibility
        for model, score in metrics.model_compatibility.items():
            if score < 0.5:
                warnings.append(f"Low compatibility with {model}")
        
        return warnings
    
    async def batch_process(self, requests: List[ProcessingRequest]) -> List[ProcessingResult]:
        """Process multiple prompts in batch"""
        logger.info(f"Processing batch of {len(requests)} prompts")
        
        # Process in parallel with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent processing
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.process_prompt(request)
        
        tasks = [process_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing request {requests[i].id}: {result}")
                # Create error result
                error_result = ProcessingResult(
                    request_id=requests[i].id,
                    original_prompt=requests[i].original_prompt,
                    processed_prompt=requests[i].original_prompt,
                    processing_type=requests[i].processing_type,
                    quality_score=0.0,
                    improvements=[],
                    warnings=[f"Processing failed: {str(result)}"],
                    metrics=self.prompt_engine.analyze_prompt(requests[i].original_prompt),
                    processing_time=0.0,
                    model_used=self.model,
                    created_at=datetime.now()
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results

# Example usage
if __name__ == "__main__":
    async def test_processor():
        processor = ProcessorAgent()
        
        # Test different processing types
        test_prompts = [
            ("Write code", ProcessingType.ENHANCEMENT),
            ("Explain quantum computing in great detail with mathematical formulations and comprehensive analysis", ProcessingType.SUMMARIZATION),
            ("Make a thing that does stuff", ProcessingType.CLARIFICATION),
            ("Create a simple web page", ProcessingType.OPTIMIZATION)
        ]
        
        for prompt, proc_type in test_prompts:
            request = ProcessingRequest(
                id=f"test_{proc_type.value}",
                original_prompt=prompt,
                processing_type=proc_type,
                target_model="smollm:360m"
            )
            
            result = await processor.process_prompt(request)
            
            print(f"\n--- {proc_type.value.upper()} ---")
            print(f"Original: {result.original_prompt}")
            print(f"Processed: {result.processed_prompt}")
            print(f"Quality Score: {result.quality_score:.2f}")
            print(f"Improvements: {result.improvements}")
            print(f"Warnings: {result.warnings}")
    
    asyncio.run(test_processor())

