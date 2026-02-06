# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARES (Agentic Research and Evaluation Suite) is an RL-first framework for training and evaluating code agents. It implements an async version of DeepMind's dm_env specification, treating LLM requests as observations and LLM responses as actions within a standard RL loop. Published on PyPI as `martian-ares`.

## Development Commands

### Setup
```bash
uv sync --all-groups       # Install all dependencies including dev tools
uv sync                    # Install only main dependencies
uv sync --group dev        # Install dev tools (ruff, pyright, pytest)
uv sync --group examples   # Install example dependencies
```

### Testing
```bash
uv run pytest                              # Run all unit tests (src/ only)
uv run pytest src/ares/config_test.py      # Run a specific test file
uv run pytest -k "test_pattern_name"       # Run tests matching a pattern
uv run pytest -n auto                      # Run tests in parallel (pytest-xdist)
```

Unit tests follow the `*_test.py` naming pattern (preferred) or `test_*.py` and are colocated with source files in `src/`. Integration tests live under `integration_tests/` and must be run manually (they require live API keys).

### Code Quality
```bash
uv run ruff check          # Lint
uv run ruff check --fix    # Auto-fix lint issues
uv run ruff format         # Format code
uv run pyright             # Type checking (basic mode, Python 3.12)
```

Ruff config: line length 120, target `py313`, Google-style docstrings. See `ruff.toml` for full rule set.

### Running Examples
```bash
uv run -m examples.01_sequential_eval_with_local_llm   # Local Docker + local LLM
uv run -m examples.02_sequential_eval_with_api          # Sequential eval with API
uv run -m examples.03_parallel_eval_with_api            # Parallel eval with API
```

### ARES Proxy (Go)
```bash
cd ares-proxy && make build   # Build the Go proxy binary
cd ares-proxy && make test    # Run Go tests
cd ares-proxy && make run     # Run proxy (port 8080, 15 min timeout)
```

### Before Pushing
```bash
uv run pytest && uv run ruff format && uv run ruff check && uv run pyright
```

## High-Level Architecture

### Core Abstraction: The RL Loop

1. **Environment** emits an **Observation** (LLM request with task context)
2. **Agent** receives observation and returns an **Action** (LLM response with code/commands)
3. **Environment** processes action, executes commands in container, returns next observation
4. Loop continues until episode terminates (success, step limit of 250, or explicit submission)
5. **Reward** is computed at episode end (read from `/reward.txt` or `/reward.json` in container)

### Registry and Presets System

The primary API for creating environments:

```python
import ares
env = ares.make("sbv-mswea")           # SWE-bench Verified with mini-swe-agent
env = ares.make("sbv-mswea:0")         # Single task at index 0
env = ares.make("sbv-mswea:0:10")      # Tasks 0-9 (slice)
env = ares.make("sbv-mswea@2/8")       # Shard 2 of 8
ares.info()                             # List all presets
```

Task selector syntax: `:N` (single index), `:start:end` (slice), `@shard/total` (sharding).

Presets are auto-registered on import from Harbor datasets. Each dataset gets both `mswea` (mini-swe-agent) and `terminus2` variants. The default container factory is `DockerContainer` (not Daytona).

Key files: `src/ares/registry.py` (registry mechanism), `src/ares/presets.py` (default preset registrations).

### Key Components

#### Environments (`src/ares/environments/`)
- `Environment` protocol defines the dm_env interface (`reset()`, `step()`, `close()`)
- `CodeEnvironment` (`code_env.py`) - Concrete implementation for Harbor-compatible datasets (including SWE-bench)
- `TimeStep` namedtuple for observations, rewards, and episode signals
- `create_container()` helper, `Janitor` class for emergency container cleanup
- Harbor integration: `load_harbor_dataset()`, `list_harbor_datasets()`

#### Code Agents (`src/ares/code_agents/`)
- `CodeAgent` protocol: `async def run(self, task: str) -> None`
- `MiniSWECodeAgent` (`mini_swe_agent.py`) - Wraps mini-swe-agent library, parses bash commands from markdown
- `Terminus2Agent` (`terminus2/terminus2_agent.py`) - Uses tmux for persistent terminal sessions, supports XML and JSON parsing

#### Containers (`src/ares/containers/`)
- `Container` protocol with `start()`, `exec_run()`, `upload_files/download_files`, `stop_and_remove()`
- `DockerContainer` (`docker.py`) - Local Docker containers (default)
- `DaytonaContainer` (`daytona.py`) - Cloud containers via Daytona API
- Both register cleanup with `atexit` for emergency shutdown

#### LLM Clients (`src/ares/llms/`)
- `LLMRequest`/`LLMResponse` dataclasses, `LLMClient` protocol
- **Queue-Mediated LLM Client** (`queue_mediated_client.py`) - The most critical pattern. Intercepts LLM calls from agents using `asyncio.Queue`, exposing them as RL observations. This is what allows agents to be written naturally while the environment controls the RL loop.
- `ChatCompletionCompatibleLLMClient` (`chat_completions_compatible.py`) - OpenAI-compatible API client (Martian API default)
- Cost tracking via `accounting.py`

#### ARES Proxy (`ares-proxy/`)
Go-based HTTP proxy that intercepts OpenAI-compatible chat completion requests. Three endpoints:
- `POST /v1/chat/completions` - Queues request, blocks until response
- `GET /poll` - Retrieves pending requests for RL controller
- `POST /respond` - Sends response back to waiting client

This is the network-level equivalent of `QueueMediatedLLMClient` - enables RL control of LLM interactions when agents run in separate processes/containers.

#### Supporting Modules
- `experiment_tracking/` - `StatTracker` protocol with `NullStatTracker`, `LoggingStatTracker`, `TensorboardStatTracker`
- `config.py` - Pydantic Settings-based configuration from `.env`
- `testing/` - `MockContainer` and `MockLLMClient` for unit tests
- `contrib/` - Experimental: `llama_cpp.py` (local CPU inference), `eval_visualizer.py` (Textual TUI)

## Key Design Patterns

- **Protocol-Oriented Design**: `typing.Protocol` for structural subtyping. Key protocols: `Environment`, `CodeAgent`, `Container`, `LLMClient`, `ContainerFactory`, `CodeAgentFactory`, `StatTracker`.
- **Factory Pattern**: Environments receive factories (not concrete instances) for dependency injection.
- **Context Manager Lifecycle**: All major resources use `async with` for guaranteed cleanup.
- **Dataclass Immutability**: Most dataclasses use `frozen=True` for async safety.
- **Queue-Mediated Communication**: Async queues bridge linear agent code with the RL environment.
- **YAGNI**: Prefer concrete implementations over premature abstractions.

## Code Conventions

### Imports (Google Style)
- **Always import modules, not classes or functions**
- **External consumers**: `import ares` or `from ares import llms` -> `llms.LLMRequest`
- **Internal code**: `from ares.llms import request` -> `request.LLMRequest`
- **Avoid**: `from ares.llms.request import LLMRequest`
- Force single-line imports (except `typing` and `collections.abc`)

### Naming
- Private methods: `_method_name`; loggers: `_LOGGER`; constants: `UPPER_CASE`
- Full type annotations throughout; generic types used extensively

### Comments
- WHY over WHAT; HOW only when genuinely complex

## Environment Variables

Create `.env` from `.env.example`:
- `DAYTONA_API_KEY`, `DAYTONA_API_URL` - Required for Daytona containers
- `CHAT_COMPLETION_API_KEY` - Required for LLM inference (Martian API)
- `CHAT_COMPLETION_BASE_URL` - Override default API endpoint (optional)

## CI/CD

Three GitHub Actions workflows:
- `ruff.yml` - Linting (`ruff check`) and formatting (`ruff format --check`)
- `unit-tests.yml` - `pytest` on `src/`
- `go-tests.yml` - Go tests for ares-proxy (triggers only on `ares-proxy/**` changes)
