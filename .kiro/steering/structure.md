# Project Structure

## Root Directory
- `pyproject.toml`: Poetry configuration and dependencies
- `pytest.ini`: Test configuration with logging settings
- `run.sh`: Evaluation script with model configurations
- `.env.example`: Environment variable template
- `datasets/`: Pre-built evaluation datasets (JSON format)
- `assets/`: Documentation images and diagrams

## Core Package (`moatless/`)

### Agent System
- `agent/`: Core agent implementations
  - `agent.py`: Base ActionAgent class with completion integration
  - `code_agent.py`: Specialized coding agent
  - `code_prompts.py`: System prompts for coding tasks
  - `settings.py`: Agent configuration schemas

### Actions Framework
- `actions/`: Modular action system
  - `action.py`: Base Action class and registry
  - `*_action.py`: Specific actions (FindClass, ViewCode, StringReplace, etc.)
  - `model.py`: Pydantic models for action arguments and observations
  - Mixins: `code_action_value_mixin.py`, `code_modification_mixin.py`

### Search & Tree Operations
- `search_tree.py`: MCTS implementation and tree management
- `node.py`: Tree node representations and state management
- `selector/`: Node selection strategies (BestFirstSelector, etc.)
- `value_function/`: Node evaluation and scoring

### Code Understanding
- `index/`: Vector indexing and semantic search
  - `code_index.py`: Main indexing interface
  - `embed_model.py`: Embedding model abstractions
  - `simple_faiss.py`: FAISS-based similarity search
- `codeblocks/`: Code parsing and structure analysis
  - `parser/`: Tree-sitter based parsers for Python/Java
  - `queries/`: Tree-sitter query definitions (.scm files)

### Repository Management
- `repository/`: Git repository operations
  - `repository.py`: Main repository interface
  - `file.py`: File content and metadata handling
  - `git.py`: Git operations and diff management

### Completion & LLM Integration
- `completion/`: LLM provider abstractions
  - `completion.py`: Base completion interface
  - `anthropic.py`: Anthropic-specific implementation
  - `model.py`: Response models and schemas

### Feedback & Evaluation
- `feedback/`: Agent feedback systems
- `discriminator.py`: Multi-agent discrimination
- `benchmark/`: Evaluation framework
  - `run_evaluation.py`: Main evaluation runner
  - `swebench/`: SWE-Bench specific utilities
  - `evaluation_*.py`: Configuration and factory classes

### Utilities
- `utils/`: Helper functions
  - `file.py`: File operations
  - `parse.py`: Text parsing utilities
  - `tokenizer.py`: Token counting and management
- `runtime/`: Execution environments
- `workspace.py`: Workspace management
- `file_context.py`: File context tracking

### Visualization
- `streamlit/`: Web interface
  - `app.py`: Main Streamlit application
  - `tree_visualization.py`: Tree rendering
  - `cli.py`: Command-line interface

## Testing (`tests/`)
- Mirrors main package structure
- `conftest.py`: Pytest configuration with custom fixtures
- `integration_test.py`: End-to-end testing
- Custom markers: `llm_integration`, `with_index`
- Test data in `tests/*/data/` subdirectories

## Naming Conventions
- **Classes**: PascalCase (e.g., `ActionAgent`, `SearchTree`)
- **Files**: snake_case matching class names
- **Actions**: Descriptive names (e.g., `FindFunction`, `StringReplace`)
- **Modules**: Organized by functionality, not implementation details
- **Tests**: `test_*.py` with descriptive test function names

## Architecture Patterns
- **Plugin System**: Actions are registered and discoverable
- **Pydantic Models**: Extensive use for validation and serialization
- **Composition**: Agents compose multiple actions and utilities
- **Factory Pattern**: Evaluation and completion model creation
- **Observer Pattern**: Message history and logging integration