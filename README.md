# agentZERO-ollama

🤖 **En agentic Ollama-basert AI-agent fokusert på prompt-prosessering**

AgentZERO-ollama er en lightweight, lokal AI-agent som bruker små Ollama-modeller (under 500MB) for effektiv prompt-prosessering og agentic oppgaver. Systemet er designet for å kjøre på beskjedne maskinvareressurser mens det leverer kraftig AI-funksjonalitet.

## 🎯 Hovedfunksjoner

- **3 Optimaliserte Modeller**: SmolLM 135M (92MB), SmolLM 360M (229MB), og SmolLM 1.7B (991MB)
- **Prompt Engineering**: Avansert prompt-prosessering og optimalisering
- **Agentic Workflow**: Multi-step oppgaveløsning med intelligent planlegging
- **Lokal Kjøring**: Ingen data sendes til eksterne servere
- **Lightweight**: Fungerer på standard laptops og desktop-maskiner
- **Cyberpunk UI**: Moderne, terminal-inspirert brukergrensesnitt

## 🏗️ Arkitektur

```
agentZERO-ollama/
├── agents/                 # Agent-definisjonsfiler
│   ├── planner.py         # Planleggingsagent
│   ├── executor.py        # Utførelsesagent
│   └── processor.py       # Prompt-prosesseringsagent
├── models/                # Modellkonfigurasjon
│   ├── smollm_135m.yaml   # SmolLM 135M konfigurasjon
│   ├── smollm_360m.yaml   # SmolLM 360M konfigurasjon
│   └── smollm_1.7b.yaml   # SmolLM 1.7B konfigurasjon
├── prompts/               # Prompt-templates
│   ├── system/            # System prompts
│   ├── task/              # Oppgave-spesifikke prompts
│   └── optimization/      # Prompt-optimaliseringsregler
├── ui/                    # Brukergrensesnitt
│   ├── app.py            # Hovedapplikasjon
│   ├── static/           # CSS, JS, bilder
│   └── templates/        # HTML-templates
├── core/                  # Kjernefunksjonalitet
│   ├── ollama_client.py  # Ollama API-klient
│   ├── prompt_engine.py  # Prompt-prosessering
│   └── agent_manager.py  # Agent-koordinering
└── docker/               # Docker-konfigurasjon
    ├── Dockerfile
    └── docker-compose.yml
```

## 🚀 Rask Start

### Forutsetninger

- Python 3.9+
- Docker (valgfritt)
- Ollama installert lokalt

### Installasjon

1. **Klon repositoriet:**
```bash
git clone https://github.com/yourusername/agentZERO-ollama.git
cd agentZERO-ollama
```

2. **Installer avhengigheter:**
```bash
pip install -r requirements.txt
```

3. **Last ned Ollama-modeller:**
```bash
ollama pull smollm:135m
ollama pull smollm:360m
ollama pull smollm:1.7b
```

4. **Start applikasjonen:**
```bash
python ui/app.py
```

5. **Åpne nettleseren:**
Gå til `http://localhost:8080`

## 🎮 Bruk

### Grunnleggende Chat
```python
from core.agent_manager import AgentManager

agent = AgentManager(model="smollm:135m")
response = agent.process("Skriv en kort historie om en robot")
print(response)
```

### Avansert Prompt-prosessering
```python
from core.prompt_engine import PromptEngine

engine = PromptEngine()
optimized_prompt = engine.optimize(
    original="Forklar kvantedatabehandling",
    target_model="smollm:360m",
    complexity="intermediate"
)
```

### Multi-Agent Workflow
```python
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent

planner = PlannerAgent(model="smollm:1.7b")
executor = ExecutorAgent(model="smollm:360m")

plan = planner.create_plan("Lag en enkel nettside")
result = executor.execute_plan(plan)
```

## 🧠 Modell-spesifikasjoner

| Modell | Størrelse | Parametere | Bruksområde | RAM-krav |
|--------|-----------|------------|-------------|----------|
| SmolLM 135M | 92MB | 135M | Rask respons, enkle oppgaver | 1GB |
| SmolLM 360M | 229MB | 360M | Balansert ytelse/hastighet | 2GB |
| SmolLM 1.7B | 991MB | 1.7B | Komplekse oppgaver, resonnering | 4GB |

## 🎨 Prompt-prosessering Funksjoner

### Automatisk Optimalisering
- **Modell-spesifikk tilpasning**: Prompts optimaliseres for hver modells styrker
- **Kontekst-komprimering**: Reduserer token-bruk uten å miste informasjon
- **Template-system**: Gjenbrukbare prompt-templates for vanlige oppgaver

### Intelligent Routing
- **Oppgave-analyse**: Automatisk valg av beste modell for oppgaven
- **Load balancing**: Fordeler oppgaver basert på modell-tilgjengelighet
- **Fallback-system**: Automatisk nedgradering ved ressursmangel

### Prompt Engineering Tools
- **A/B Testing**: Sammenlign prompt-varianter
- **Ytelsesmålinger**: Spor responstid og kvalitet
- **Iterativ forbedring**: Lær fra tidligere interaksjoner

## 🔧 Konfigurasjon

### Modell-konfigurasjon (models/smollm_135m.yaml)
```yaml
model:
  name: "smollm:135m"
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  
prompts:
  system: "Du er en hjelpsom AI-assistent som gir korte, presise svar."
  max_length: 512
  
performance:
  timeout: 30
  retry_attempts: 3
```

### Agent-konfigurasjon
```yaml
agents:
  planner:
    model: "smollm:1.7b"
    role: "strategic_planning"
    max_steps: 10
    
  executor:
    model: "smollm:360m"
    role: "task_execution"
    parallel_tasks: 3
    
  processor:
    model: "smollm:135m"
    role: "prompt_optimization"
    cache_enabled: true
```

## 🐳 Docker Deployment

```bash
# Bygg og start med Docker Compose
docker-compose up --build

# Eller bygg manuelt
docker build -t agentzero-ollama .
docker run -p 8080:8080 -v ollama_models:/root/.ollama agentzero-ollama
```

## 📊 Ytelse og Benchmarks

### Responstider (gjennomsnitt)
- **SmolLM 135M**: ~200ms
- **SmolLM 360M**: ~500ms  
- **SmolLM 1.7B**: ~1.2s

### Minnebruk
- **Basis system**: ~500MB RAM
- **Med alle modeller**: ~2.5GB RAM
- **Disk space**: ~1.5GB total

## 🛠️ Utvikling

### Legg til ny agent
```python
from core.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, model="smollm:360m"):
        super().__init__(model)
        self.role = "custom_task"
    
    def process_task(self, task):
        prompt = self.build_prompt(task)
        return self.generate_response(prompt)
```

### Lag custom prompt template
```yaml
# prompts/custom/my_template.yaml
template:
  name: "code_generation"
  description: "Genererer kode basert på beskrivelse"
  
system_prompt: |
  Du er en ekspert programmerer. Generer ren, kommentert kode.
  
user_template: |
  Oppgave: {task}
  Språk: {language}
  Kompleksitet: {complexity}
  
  Generer kode som løser denne oppgaven.
```

## 🤝 Bidrag

Vi ønsker bidrag! Se [CONTRIBUTING.md](CONTRIBUTING.md) for retningslinjer.

### Utviklingsoppsett
```bash
# Klon repo
git clone https://github.com/yourusername/agentZERO-ollama.git
cd agentZERO-ollama

# Opprett virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# eller
venv\Scripts\activate     # Windows

# Installer dev-avhengigheter
pip install -r requirements-dev.txt

# Kjør tester
pytest tests/
```

## 📝 Lisens

MIT License - se [LICENSE](LICENSE) for detaljer.

## 🙏 Takk til

- [Ollama](https://ollama.ai/) for den fantastiske lokale LLM-plattformen
- [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M) for de effektive små modellene
- Agent Zero community for inspirasjon

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/agentZERO-ollama/issues)
- 💬 **Diskusjoner**: [GitHub Discussions](https://github.com/yourusername/agentZERO-ollama/discussions)
- 📧 **Email**: support@agentzero-ollama.com

---

**AgentZERO-ollama** - Kraftig AI, minimal fotavtrykk. 🚀

