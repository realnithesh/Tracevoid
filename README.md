# TRACEVOID

Local claim-analysis workstation using Ollama.

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- The DeepSeek R1 8B model downloaded locally

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull deepseek-r1:8b
python server.py
```

Open http://127.0.0.1:5000 after the server starts. Ollama must remain running
locally at `http://127.0.0.1:11434` by default.

The model analyzes only the claim and source material you provide; it does not
search the web or independently verify facts. Local CPU inference can be slow.
You can set `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, and
`OLLAMA_NUM_PREDICT` in `.env` to change the local endpoint, model, request
timeout, or maximum response length.