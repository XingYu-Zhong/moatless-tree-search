# Technology Stack

## Build System & Package Management
- **Poetry**: Primary dependency management and build system
- **Python**: 3.10-3.13 supported versions
- **Package**: Distributed as `moatless-tree-search` on PyPI

## Core Dependencies
- **Pydantic**: Data validation and settings management (v2.8+)
- **LiteLLM**: Multi-provider LLM integration (v1.51+)
- **Instructor**: Structured LLM outputs (<=1.6.3)
- **NetworkX**: Graph operations for tree structures
- **Tree-sitter**: Code parsing (Python, Java support)
- **GitPython**: Git repository operations
- **Unidiff**: Patch/diff handling

## LLM Providers
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet/Haiku)
- Custom API endpoints via base URL configuration
- DeepSeek, Hugging Face support

## Vector Search & Embeddings
- **LlamaIndex**: Vector indexing framework
- **FAISS**: Similarity search backend
- **Voyage AI**: Embedding provider for semantic search
- **OpenAI Embeddings**: Alternative embedding option

## Testing & Development
- **Pytest**: Test framework with custom markers
- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Coverage**: Test coverage reporting

## Visualization & UI
- **Streamlit**: Web interface for trajectory visualization
- **Plotly**: Interactive charts and graphs
- **Matplotlib**: Static plotting
- **PyGraphviz**: Graph visualization

## Common Commands

### Development Setup
```bash
# Install with Poetry
poetry install --with dev

# Apple Silicon workaround for graphviz
brew install graphviz
export CFLAGS="-I $(brew --prefix graphviz)/include"
export LDFLAGS="-L $(brew --prefix graphviz)/lib"
poetry install --with dev
```

### Running Tests
```bash
# Basic test run
pytest

# With LLM integration tests
pytest --run-llm-integration

# With vector index tests
pytest --run-with-index

# Custom directories
pytest --index-store-dir=/custom/path --repo-dir=/custom/repos
```

### Evaluation & Benchmarking
```bash
# Run evaluation
moatless-evaluate \
    --model "gpt-4o-mini" \
    --repo_base_dir /tmp/repos \
    --eval_dir "./evaluations" \
    --eval_name mts \
    --temp 0.7 \
    --num_workers 1 \
    --use_testbed \
    --feedback \
    --max_iterations 100 \
    --max_expansions 5
```

### Streamlit Interface
```bash
# Launch with trajectory file
moatless-streamlit path/to/trajectory.json

# Launch interactive UI
moatless-streamlit
```

## Environment Configuration
Required environment variables (see `.env.example`):
- `INDEX_STORE_DIR`: Vector index storage
- `REPO_DIR`: Repository clones location
- LLM API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
- `VOYAGE_API_KEY`: For embeddings
- `TESTBED_API_KEY`, `TESTBED_BASE_URL`: For testing integration