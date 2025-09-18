"""
Prompt Processing Engine for agentZERO
Handles prompt optimization, templating, and model-specific adaptations
"""

import re
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    name: str
    description: str
    system_prompt: str
    user_template: str
    variables: List[str] = field(default_factory=list)
    model_specific: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_tokens: int = 2048
    temperature: float = 0.7

@dataclass
class OptimizationRule:
    """Rule for prompt optimization"""
    name: str
    pattern: str
    replacement: str
    model_filter: Optional[List[str]] = None
    description: str = ""

@dataclass
class PromptMetrics:
    """Metrics for prompt performance"""
    token_count: int
    estimated_cost: float
    complexity_score: float
    readability_score: float
    model_compatibility: Dict[str, float]

class PromptEngine:
    """Engine for processing and optimizing prompts"""
    
    def __init__(self, templates_dir: str = "prompts", models_dir: str = "models"):
        self.templates_dir = Path(templates_dir)
        self.models_dir = Path(models_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.optimization_rules: List[OptimizationRule] = []
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Load templates and rules
        self._load_templates()
        self._load_optimization_rules()
        self._load_model_configs()
        
        # Model-specific token limits and characteristics
        self.model_specs = {
            "smollm:135m": {
                "max_tokens": 1024,
                "context_window": 2048,
                "strengths": ["speed", "efficiency"],
                "weaknesses": ["complex_reasoning", "long_context"],
                "optimal_prompt_length": 200,
                "token_cost": 0.001
            },
            "smollm:360m": {
                "max_tokens": 1536,
                "context_window": 4096,
                "strengths": ["balanced", "general_tasks"],
                "weaknesses": ["specialized_knowledge"],
                "optimal_prompt_length": 400,
                "token_cost": 0.002
            },
            "smollm:1.7b": {
                "max_tokens": 2048,
                "context_window": 8192,
                "strengths": ["reasoning", "complex_tasks", "context"],
                "weaknesses": ["speed"],
                "optimal_prompt_length": 800,
                "token_cost": 0.005
            }
        }
    
    def _load_templates(self):
        """Load prompt templates from files"""
        template_files = list(self.templates_dir.rglob("*.yaml"))
        
        for template_file in template_files:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'template' in data:
                    template_data = data['template']
                    template = PromptTemplate(
                        name=template_data['name'],
                        description=template_data.get('description', ''),
                        system_prompt=template_data.get('system_prompt', ''),
                        user_template=template_data.get('user_template', ''),
                        variables=template_data.get('variables', []),
                        model_specific=template_data.get('model_specific', {}),
                        max_tokens=template_data.get('max_tokens', 2048),
                        temperature=template_data.get('temperature', 0.7)
                    )
                    self.templates[template.name] = template
                    logger.info(f"Loaded template: {template.name}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
    
    def _load_optimization_rules(self):
        """Load prompt optimization rules"""
        rules_file = self.templates_dir / "optimization" / "rules.yaml"
        
        if rules_file.exists():
            try:
                with open(rules_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                for rule_data in data.get('rules', []):
                    rule = OptimizationRule(
                        name=rule_data['name'],
                        pattern=rule_data['pattern'],
                        replacement=rule_data['replacement'],
                        model_filter=rule_data.get('model_filter'),
                        description=rule_data.get('description', '')
                    )
                    self.optimization_rules.append(rule)
                    logger.info(f"Loaded optimization rule: {rule.name}")
            except Exception as e:
                logger.error(f"Error loading optimization rules: {e}")
    
    def _load_model_configs(self):
        """Load model-specific configurations"""
        config_files = list(self.models_dir.rglob("*.yaml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if 'model' in data:
                    model_name = data['model']['name']
                    self.model_configs[model_name] = data
                    logger.info(f"Loaded model config: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model config {config_file}: {e}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())
    
    def render_template(
        self, 
        template_name: str, 
        variables: Dict[str, Any],
        model: str = "smollm:360m"
    ) -> Tuple[str, str]:
        """Render a template with variables"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Use model-specific overrides if available
        system_prompt = template.system_prompt
        user_template = template.user_template
        
        if model in template.model_specific:
            model_overrides = template.model_specific[model]
            system_prompt = model_overrides.get('system_prompt', system_prompt)
            user_template = model_overrides.get('user_template', user_template)
        
        # Render templates
        system_rendered = Template(system_prompt).render(**variables)
        user_rendered = Template(user_template).render(**variables)
        
        return system_rendered, user_rendered
    
    def optimize_prompt(
        self, 
        prompt: str, 
        model: str = "smollm:360m",
        target_length: Optional[int] = None
    ) -> str:
        """Optimize a prompt for a specific model"""
        optimized = prompt
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            if rule.model_filter and model not in rule.model_filter:
                continue
            
            optimized = re.sub(rule.pattern, rule.replacement, optimized, flags=re.IGNORECASE)
        
        # Model-specific optimizations
        if model in self.model_specs:
            spec = self.model_specs[model]
            optimal_length = target_length or spec["optimal_prompt_length"]
            
            # Truncate if too long
            if len(optimized) > optimal_length * 2:
                optimized = self._smart_truncate(optimized, optimal_length)
            
            # Add model-specific instructions
            if "speed" in spec["strengths"]:
                optimized = f"Be concise and direct. {optimized}"
            elif "reasoning" in spec["strengths"]:
                optimized = f"Think step by step. {optimized}"
        
        return optimized
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Intelligently truncate text while preserving meaning"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') <= max_length:
                truncated += sentence + '. '
            else:
                break
        
        if not truncated:
            # Fallback to character truncation
            truncated = text[:max_length-3] + "..."
        
        return truncated.strip()
    
    def analyze_prompt(self, prompt: str, model: str = "smollm:360m") -> PromptMetrics:
        """Analyze prompt characteristics and performance metrics"""
        # Estimate token count (rough approximation)
        token_count = len(prompt.split()) * 1.3  # Average tokens per word
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(prompt)
        
        # Calculate readability score
        readability_score = self._calculate_readability(prompt)
        
        # Estimate cost
        model_spec = self.model_specs.get(model, self.model_specs["smollm:360m"])
        estimated_cost = token_count * model_spec["token_cost"]
        
        # Model compatibility
        model_compatibility = self._calculate_model_compatibility(prompt)
        
        return PromptMetrics(
            token_count=int(token_count),
            estimated_cost=estimated_cost,
            complexity_score=complexity_score,
            readability_score=readability_score,
            model_compatibility=model_compatibility
        )
    
    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity score (0-1)"""
        factors = {
            'length': min(len(prompt) / 1000, 1.0) * 0.3,
            'vocabulary': len(set(prompt.lower().split())) / len(prompt.split()) * 0.3,
            'punctuation': prompt.count(',') + prompt.count(';') + prompt.count(':') * 0.1,
            'questions': prompt.count('?') * 0.1,
            'instructions': len(re.findall(r'\b(please|should|must|need to|have to)\b', prompt.lower())) * 0.1
        }
        
        return min(sum(factors.values()), 1.0)
    
    def _calculate_readability(self, prompt: str) -> float:
        """Calculate readability score (0-1, higher is more readable)"""
        words = prompt.split()
        sentences = prompt.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (inverted complexity)
        readability = 1.0 - min((avg_words_per_sentence / 20 + avg_chars_per_word / 10) / 2, 1.0)
        
        return max(readability, 0.0)
    
    def _calculate_model_compatibility(self, prompt: str) -> Dict[str, float]:
        """Calculate compatibility scores for different models"""
        compatibility = {}
        
        for model, spec in self.model_specs.items():
            score = 1.0
            
            # Length penalty
            optimal_length = spec["optimal_prompt_length"]
            length_ratio = len(prompt) / optimal_length
            if length_ratio > 1.5:
                score *= 0.7
            elif length_ratio < 0.5:
                score *= 0.9
            
            # Complexity penalty for smaller models
            complexity = self._calculate_complexity(prompt)
            if "complex_reasoning" in spec["weaknesses"] and complexity > 0.7:
                score *= 0.6
            
            # Context window check
            estimated_tokens = len(prompt.split()) * 1.3
            if estimated_tokens > spec["context_window"] * 0.8:
                score *= 0.5
            
            compatibility[model] = max(score, 0.1)
        
        return compatibility
    
    def suggest_model(self, prompt: str) -> str:
        """Suggest the best model for a given prompt"""
        compatibility = self._calculate_model_compatibility(prompt)
        
        # Weight by model capabilities
        weighted_scores = {}
        for model, score in compatibility.items():
            spec = self.model_specs[model]
            
            # Prefer faster models for simple tasks
            complexity = self._calculate_complexity(prompt)
            if complexity < 0.3 and "speed" in spec["strengths"]:
                score *= 1.2
            elif complexity > 0.7 and "reasoning" in spec["strengths"]:
                score *= 1.3
            
            weighted_scores[model] = score
        
        return max(weighted_scores, key=weighted_scores.get)
    
    def create_chat_messages(
        self, 
        system_prompt: str, 
        user_prompt: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Create properly formatted chat messages"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for prompt/model combination"""
        content = f"{prompt}|{model}|{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

# Example usage and testing
if __name__ == "__main__":
    engine = PromptEngine()
    
    # Test prompt optimization
    test_prompt = "Please explain quantum computing in detail with examples and applications"
    
    for model in ["smollm:135m", "smollm:360m", "smollm:1.7b"]:
        optimized = engine.optimize_prompt(test_prompt, model)
        metrics = engine.analyze_prompt(optimized, model)
        
        print(f"\n--- {model} ---")
        print(f"Optimized: {optimized}")
        print(f"Tokens: {metrics.token_count}")
        print(f"Complexity: {metrics.complexity_score:.2f}")
        print(f"Readability: {metrics.readability_score:.2f}")
        print(f"Cost: ${metrics.estimated_cost:.4f}")
    
    # Test model suggestion
    suggested = engine.suggest_model(test_prompt)
    print(f"\nSuggested model: {suggested}")

