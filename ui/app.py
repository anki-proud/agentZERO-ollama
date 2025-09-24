"""
Main Streamlit application for agentZERO-ollama
Provides web interface for interacting with the AI agents
"""

import streamlit as st
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ollama_client import OllamaClient, GenerationConfig
from core.prompt_engine import PromptEngine
from agents.planner import PlannerAgent, TaskPlan
from agents.executor import ExecutorAgent, TaskExecution
from agents.processor import ProcessorAgent, ProcessingRequest, ProcessingType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="agentZERO-ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cyberpunk theme
st.markdown("""
<style>
    .main {
        background-color: #0a0a0a;
        color: #00ff41;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    .stSidebar {
        background-color: #16213e;
        border-right: 2px solid #00ff41;
    }
    
    .stButton > button {
        background-color: #0f3460;
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    
    .stButton > button:hover {
        background-color: #00ff41;
        color: #0a0a0a;
        box-shadow: 0 0 10px #00ff41;
    }
    
    .stTextInput > div > div > input {
        background-color: #1a1a2e;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Courier New', monospace;
    }
    
    .stSelectbox > div > div > select {
        background-color: #1a1a2e;
        color: #00ff41;
        border: 1px solid #00ff41;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #1a1a2e;
        color: #00ff41;
        border: 1px solid #00ff41;
        font-family: 'Courier New', monospace;
    }
    
    .agent-status {
        padding: 10px;
        border: 1px solid #00ff41;
        border-radius: 5px;
        background-color: #1a1a2e;
        margin: 10px 0;
    }
    
    .model-info {
        background-color: #0f3460;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ff41;
        margin: 10px 0;
    }
    
    .execution-step {
        background-color: #16213e;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #00ff41;
        border-radius: 3px;
    }
    
    h1, h2, h3 {
        color: #00ff41;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 5px #00ff41;
    }
    
    .metric-card {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #00ff41;
        text-align: center;
    }
    
    .success {
        color: #00ff41;
    }
    
    .warning {
        color: #ffaa00;
    }
    
    .error {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ollama_client' not in st.session_state:
    st.session_state.ollama_client = OllamaClient()
if 'prompt_engine' not in st.session_state:
    st.session_state.prompt_engine = PromptEngine()
if 'planner' not in st.session_state:
    st.session_state.planner = PlannerAgent()
if 'executor' not in st.session_state:
    st.session_state.executor = ExecutorAgent()
if 'processor' not in st.session_state:
    st.session_state.processor = ProcessorAgent()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_execution' not in st.session_state:
    st.session_state.current_execution = None

def check_ollama_connection():
    """Check if Ollama is available"""
    try:
        return st.session_state.ollama_client.is_available()
    except:
        return False

async def get_available_models():
    """Get list of available Ollama models"""
    try:
        async with st.session_state.ollama_client as client:
            models = await client.list_models()
            return [model.name for model in models]
    except:
        return []

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🤖 agentZERO-ollama</h1>
        <p style='color: #00ff41; font-family: Courier New;'>
            Agentic AI powered by local Ollama models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Control Panel")
        
        # Connection status
        if check_ollama_connection():
            st.markdown('<div class="success">✅ Ollama Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error">❌ Ollama Disconnected</div>', unsafe_allow_html=True)
            st.error("Please ensure Ollama is running on localhost:11434")
            return
        
        # Model selection
        st.markdown("### 🧠 Model Selection")
        available_models = asyncio.run(get_available_models())
        
        if not available_models:
            st.warning("No models found. Please pull SmolLM models:")
            st.code("""
ollama pull smollm:135m
ollama pull smollm:360m  
ollama pull smollm:1.7b
            """)
            return
        
        selected_model = st.selectbox(
            "Choose Model:",
            available_models,
            index=0 if available_models else None
        )
        
        # Model info
        if selected_model:
            show_model_info(selected_model)
        
        # Agent mode selection
        st.markdown("### 🎯 Agent Mode")
        agent_mode = st.selectbox(
            "Select Mode:",
            ["Chat", "Task Planning", "Prompt Processing", "Batch Processing"]
        )
        
        # Settings
        st.markdown("### ⚙️ Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 100, 2048, 1024, 100)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.session_state.current_execution = None
            st.rerun()
    
    # Main content area
    if agent_mode == "Chat":
        show_chat_interface(selected_model, temperature, max_tokens)
    elif agent_mode == "Task Planning":
        show_task_planning_interface(selected_model)
    elif agent_mode == "Prompt Processing":
        show_prompt_processing_interface()
    elif agent_mode == "Batch Processing":
        show_batch_processing_interface()

def show_model_info(model_name: str):
    """Display information about the selected model"""
    model_specs = {
        "smollm:135m": {
            "size": "92MB",
            "params": "135M",
            "speed": "~200ms",
            "ram": "1GB",
            "strengths": ["Speed", "Efficiency"],
            "use_cases": ["Quick Q&A", "Simple tasks"]
        },
        "smollm:360m": {
            "size": "229MB", 
            "params": "360M",
            "speed": "~500ms",
            "ram": "2GB",
            "strengths": ["Balance", "Versatility"],
            "use_cases": ["General tasks", "Content creation"]
        },
        "smollm:1.7b": {
            "size": "991MB",
            "params": "1.7B", 
            "speed": "~1.2s",
            "ram": "4GB",
            "strengths": ["Reasoning", "Complexity"],
            "use_cases": ["Analysis", "Planning"]
        }
    }
    
    if model_name in model_specs:
        spec = model_specs[model_name]
        st.markdown(f"""
        <div class="model-info">
            <h4>{model_name}</h4>
            <p><strong>Size:</strong> {spec['size']}</p>
            <p><strong>Parameters:</strong> {spec['params']}</p>
            <p><strong>Speed:</strong> {spec['speed']}</p>
            <p><strong>RAM:</strong> {spec['ram']}</p>
            <p><strong>Strengths:</strong> {', '.join(spec['strengths'])}</p>
            <p><strong>Best for:</strong> {', '.join(spec['use_cases'])}</p>
        </div>
        """, unsafe_allow_html=True)

def show_chat_interface(model: str, temperature: float, max_tokens: int):
    """Display chat interface"""
    st.markdown("## 💬 Chat Interface")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 10px 0;">
                    <div style="background-color: #0f3460; padding: 10px; border-radius: 10px; display: inline-block; max-width: 70%;">
                        <strong>You:</strong> {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: left; margin: 10px 0;">
                    <div style="background-color: #1a1a2e; padding: 10px; border-radius: 10px; display: inline-block; max-width: 70%; border-left: 3px solid #00ff41;">
                        <strong>agentZERO:</strong> {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Enter your message:", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", key="send_chat")
    with col2:
        if st.button("🔄 Regenerate Last", key="regenerate"):
            if st.session_state.chat_history:
                # Remove last assistant response and regenerate
                if st.session_state.chat_history[-1]["role"] == "assistant":
                    st.session_state.chat_history.pop()
                    if st.session_state.chat_history:
                        last_user_message = st.session_state.chat_history[-1]["content"]
                        asyncio.run(process_chat_message(last_user_message, model, temperature, max_tokens))
                        st.rerun()
    
    if send_button and user_input:
        asyncio.run(process_chat_message(user_input, model, temperature, max_tokens))
        st.rerun()

async def process_chat_message(message: str, model: str, temperature: float, max_tokens: int):
    """Process a chat message"""
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    try:
        # Generate response
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Use chat history for context
        messages = []
        for msg in st.session_state.chat_history[-10:]:  # Last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        async with st.session_state.ollama_client as client:
            response = await client.chat(model, messages, config)
        
        # Add assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

def show_task_planning_interface(model: str):
    """Display task planning interface"""
    st.markdown("## 📋 Task Planning & Execution")
    
    # Task input
    task_description = st.text_area(
        "Describe your task:",
        placeholder="e.g., Create a simple web application that displays weather information",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Create Plan", key="create_plan"):
            if task_description:
                with st.spinner("Creating plan..."):
                    plan = asyncio.run(st.session_state.planner.create_plan(task_description))
                    st.session_state.current_plan = plan
                    st.success("Plan created successfully!")
                    st.rerun()
    
    with col2:
        if st.button("▶️ Execute Plan", key="execute_plan"):
            if hasattr(st.session_state, 'current_plan'):
                with st.spinner("Executing plan..."):
                    execution = asyncio.run(st.session_state.executor.execute_plan(st.session_state.current_plan))
                    st.session_state.current_execution = execution
                    st.success("Plan executed!")
                    st.rerun()
    
    # Display current plan
    if hasattr(st.session_state, 'current_plan'):
        plan = st.session_state.current_plan
        st.markdown("### 📊 Current Plan")
        
        st.markdown(f"""
        <div class="agent-status">
            <h4>Task: {plan.original_request}</h4>
            <p><strong>Complexity:</strong> {plan.complexity.value}</p>
            <p><strong>Estimated Time:</strong> {plan.estimated_total_time} seconds</p>
            <p><strong>Steps:</strong> {len(plan.steps)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show steps
        for step in plan.steps:
            st.markdown(f"""
            <div class="execution-step">
                <h5>Step {step.id}: {step.description}</h5>
                <p><strong>Type:</strong> {step.type.value}</p>
                <p><strong>Time:</strong> {step.estimated_time}s</p>
                <p><strong>Model:</strong> {step.required_model or 'Auto-select'}</p>
                {f"<p><strong>Dependencies:</strong> {step.dependencies}</p>" if step.dependencies else ""}
            </div>
            """, unsafe_allow_html=True)
    
    # Display execution results
    if st.session_state.current_execution:
        execution = st.session_state.current_execution
        st.markdown("### 🎯 Execution Results")
        
        summary = st.session_state.executor.get_execution_summary(execution)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{summary['completed_steps']}</h4>
                <p>Completed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{summary['failed_steps']}</h4>
                <p>Failed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{summary['success_rate']:.1%}</h4>
                <p>Success Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{summary['total_execution_time']:.1f}s</h4>
                <p>Total Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show step results
        for step_id, result in execution.results.items():
            status_color = "success" if result.status.value == "completed" else "error"
            st.markdown(f"""
            <div class="execution-step">
                <h5 class="{status_color}">Step {step_id}: {result.status.value.upper()}</h5>
                <p><strong>Model:</strong> {result.model_used}</p>
                <p><strong>Time:</strong> {result.execution_time:.2f}s</p>
                <p><strong>Tokens:</strong> {result.tokens_used}</p>
                <details>
                    <summary>Output</summary>
                    <pre>{result.output[:500]}{'...' if len(result.output) > 500 else ''}</pre>
                </details>
                {f"<p class='error'><strong>Error:</strong> {result.error_message}</p>" if result.error_message else ""}
            </div>
            """, unsafe_allow_html=True)

def show_prompt_processing_interface():
    """Display prompt processing interface"""
    st.markdown("## 🔧 Prompt Processing")
    
    # Input prompt
    original_prompt = st.text_area(
        "Enter prompt to process:",
        placeholder="Enter your prompt here...",
        height=150
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        processing_type = st.selectbox(
            "Processing Type:",
            [pt.value for pt in ProcessingType]
        )
    
    with col2:
        target_model = st.selectbox(
            "Target Model:",
            ["smollm:135m", "smollm:360m", "smollm:1.7b"]
        )
    
    if st.button("🔄 Process Prompt", key="process_prompt"):
        if original_prompt:
            with st.spinner("Processing prompt..."):
                request = ProcessingRequest(
                    id=str(uuid.uuid4()),
                    original_prompt=original_prompt,
                    processing_type=ProcessingType(processing_type),
                    target_model=target_model
                )
                
                result = asyncio.run(st.session_state.processor.process_prompt(request))
                st.session_state.processing_result = result
                st.rerun()
    
    # Display results
    if hasattr(st.session_state, 'processing_result'):
        result = st.session_state.processing_result
        
        st.markdown("### 📊 Processing Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{result.quality_score:.2f}</h4>
                <p>Quality Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{result.metrics.token_count}</h4>
                <p>Tokens</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{result.processing_time:.2f}s</h4>
                <p>Process Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show original vs processed
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Prompt")
            st.text_area("", value=result.original_prompt, height=200, disabled=True, key="original")
        
        with col2:
            st.markdown("#### Processed Prompt")
            st.text_area("", value=result.processed_prompt, height=200, disabled=True, key="processed")
        
        # Show improvements and warnings
        if result.improvements:
            st.markdown("#### ✅ Improvements")
            for improvement in result.improvements:
                st.markdown(f"- {improvement}")
        
        if result.warnings:
            st.markdown("#### ⚠️ Warnings")
            for warning in result.warnings:
                st.markdown(f"- {warning}")

def show_batch_processing_interface():
    """Display batch processing interface"""
    st.markdown("## 📦 Batch Processing")
    
    st.info("Upload a JSON file with multiple prompts for batch processing")
    
    uploaded_file = st.file_uploader("Choose JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            batch_data = json.load(uploaded_file)
            st.success(f"Loaded {len(batch_data)} prompts")
            
            if st.button("🚀 Process Batch"):
                with st.spinner("Processing batch..."):
                    requests = []
                    for i, item in enumerate(batch_data):
                        request = ProcessingRequest(
                            id=f"batch_{i}",
                            original_prompt=item.get("prompt", ""),
                            processing_type=ProcessingType(item.get("type", "optimization")),
                            target_model=item.get("model", "smollm:360m")
                        )
                        requests.append(request)
                    
                    results = asyncio.run(st.session_state.processor.batch_process(requests))
                    st.session_state.batch_results = results
                    st.success("Batch processing completed!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Display batch results
    if hasattr(st.session_state, 'batch_results'):
        results = st.session_state.batch_results
        
        st.markdown("### 📊 Batch Results")
        
        # Summary metrics
        total_results = len(results)
        avg_quality = sum(r.quality_score for r in results) / total_results
        total_time = sum(r.processing_time for r in results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{total_results}</h4>
                <p>Total Processed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{avg_quality:.2f}</h4>
                <p>Avg Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{total_time:.1f}s</h4>
                <p>Total Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Results table
        st.markdown("#### Results Details")
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1} - Quality: {result.quality_score:.2f}"):
                st.text_area("Original", result.original_prompt, height=100, disabled=True, key=f"batch_orig_{i}")
                st.text_area("Processed", result.processed_prompt, height=100, disabled=True, key=f"batch_proc_{i}")

if __name__ == "__main__":
    main()

