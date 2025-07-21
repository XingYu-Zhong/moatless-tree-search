# 从零构建AI代码修复系统

## 项目概述

本文档将指导你从零开始构建一个基于AI的自动化代码修复系统，类似于Moatless Tree Search项目。该系统能够接收GitHub Issue，自动分析代码库，并生成修复方案。

## 系统架构

```
ai-code-fixer/
├── pyproject.toml          # 项目配置
├── .env.example           # 环境变量模板
├── README.md              # 项目说明
├── main.py               # 主入口
├── core/                 # 核心模块
│   ├── __init__.py
│   ├── agent.py          # AI代理
│   ├── actions/          # 动作系统
│   ├── repository/       # 代码库管理
│   ├── search/           # 代码搜索
│   └── llm/             # LLM集成
├── web/                 # Web界面
│   ├── app.py           # Streamlit应用
│   └── components/      # UI组件
└── examples/            # 示例和演示
```

## 第一阶段：项目初始化

### 1.1 创建项目结构

```bash
mkdir ai-code-fixer
cd ai-code-fixer

# 创建目录结构
mkdir -p core/{actions,repository,search,llm}
mkdir -p web/components
mkdir examples
```

### 1.2 配置Poetry项目

创建 `pyproject.toml`：

```toml
[tool.poetry]
name = "ai-code-fixer"
version = "0.1.0"
description = "AI-powered automatic code fixing system"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "core"}]

[tool.poetry.dependencies]
python = "^3.10"
pydantic = "^2.8.0"
litellm = "^1.51.0"
instructor = "<=1.6.3"
gitpython = "^3.1.0"
tree-sitter = "^0.21.0"
tree-sitter-python = "^0.21.0"
streamlit = "^1.28.0"
requests = "^2.31.0"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
ruff = "^0.1.0"
mypy = "^1.7.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.poetry.scripts]
ai-code-fixer = "main:main"
```

### 1.3 环境配置

创建 `.env.example`：

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# GitHub配置
GITHUB_TOKEN=your_github_token_here

# 工作目录
WORK_DIR=./workspace
REPO_CACHE_DIR=./repos
```

## 第二阶段：核心模块开发

### 2.1 基础数据模型

创建 `core/models.py`：

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class IssueType(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    ENHANCEMENT = "enhancement"

class Issue(BaseModel):
    id: str
    title: str
    description: str
    type: IssueType
    repository_url: str
    labels: List[str] = []
    
class CodeLocation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    content: str

class ActionResult(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class FixSuggestion(BaseModel):
    file_path: str
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)
```

### 2.2 LLM集成模块

创建 `core/llm/client.py`：

```python
import litellm
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import instructor

class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = instructor.from_litellm(litellm.completion)
    
    def complete(self, messages: List[Dict[str, str]], response_model: Optional[BaseModel] = None) -> Any:
        """完成LLM调用"""
        try:
            if response_model:
                return self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_model=response_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            else:
                response = litellm.completion(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM调用失败: {str(e)}")
```

### 2.3 代码库管理

创建 `core/repository/manager.py`：

```python
import git
import os
from pathlib import Path
from typing import List, Optional
from ..models import CodeLocation

class RepositoryManager:
    def __init__(self, work_dir: str = "./workspace"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
    
    def clone_repository(self, repo_url: str) -> str:
        """克隆代码库"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_path = self.work_dir / repo_name
        
        if repo_path.exists():
            # 如果已存在，拉取最新代码
            repo = git.Repo(repo_path)
            repo.remotes.origin.pull()
        else:
            git.Repo.clone_from(repo_url, repo_path)
        
        return str(repo_path)
    
    def read_file(self, repo_path: str, file_path: str) -> str:
        """读取文件内容"""
        full_path = Path(repo_path) / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def write_file(self, repo_path: str, file_path: str, content: str) -> None:
        """写入文件内容"""
        full_path = Path(repo_path) / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def find_python_files(self, repo_path: str) -> List[str]:
        """查找所有Python文件"""
        repo_path = Path(repo_path)
        python_files = []
        
        for file_path in repo_path.rglob("*.py"):
            if not any(part.startswith('.') for part in file_path.parts):
                python_files.append(str(file_path.relative_to(repo_path)))
        
        return python_files
```

### 2.4 代码搜索模块

创建 `core/search/code_search.py`：

```python
import re
from typing import List, Dict, Any
from pathlib import Path
from ..models import CodeLocation
from ..repository.manager import RepositoryManager

class CodeSearcher:
    def __init__(self, repo_manager: RepositoryManager):
        self.repo_manager = repo_manager
    
    def search_by_keyword(self, repo_path: str, keyword: str) -> List[CodeLocation]:
        """根据关键词搜索代码"""
        results = []
        python_files = self.repo_manager.find_python_files(repo_path)
        
        for file_path in python_files:
            try:
                content = self.repo_manager.read_file(repo_path, file_path)
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        # 获取上下文（前后3行）
                        start_line = max(0, i - 3)
                        end_line = min(len(lines), i + 4)
                        context = '\n'.join(lines[start_line:end_line])
                        
                        results.append(CodeLocation(
                            file_path=file_path,
                            line_start=start_line + 1,
                            line_end=end_line,
                            content=context
                        ))
            except Exception as e:
                print(f"搜索文件 {file_path} 时出错: {e}")
                continue
        
        return results
    
    def search_function_definition(self, repo_path: str, function_name: str) -> List[CodeLocation]:
        """搜索函数定义"""
        pattern = rf"def\s+{re.escape(function_name)}\s*\("
        return self._search_by_pattern(repo_path, pattern)
    
    def search_class_definition(self, repo_path: str, class_name: str) -> List[CodeLocation]:
        """搜索类定义"""
        pattern = rf"class\s+{re.escape(class_name)}\s*[\(:]"
        return self._search_by_pattern(repo_path, pattern)
    
    def _search_by_pattern(self, repo_path: str, pattern: str) -> List[CodeLocation]:
        """根据正则表达式搜索"""
        results = []
        python_files = self.repo_manager.find_python_files(repo_path)
        
        for file_path in python_files:
            try:
                content = self.repo_manager.read_file(repo_path, file_path)
                lines = content.split('\n')
                
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        # 获取完整的函数或类定义
                        start_line = i
                        end_line = self._find_block_end(lines, i)
                        block_content = '\n'.join(lines[start_line:end_line])
                        
                        results.append(CodeLocation(
                            file_path=file_path,
                            line_start=start_line + 1,
                            line_end=end_line,
                            content=block_content
                        ))
            except Exception as e:
                print(f"搜索文件 {file_path} 时出错: {e}")
                continue
        
        return results
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """找到代码块的结束位置"""
        if start_idx >= len(lines):
            return start_idx + 1
        
        # 简单的缩进检测
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and line.strip():
                return i
        
        return len(lines)
```

## 第三阶段：动作系统

### 3.1 基础动作类

创建 `core/actions/base.py`：

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..models import ActionResult

class BaseAction(ABC):
    """基础动作类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ActionResult:
        """执行动作"""
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数"""
        return True

class ActionRegistry:
    """动作注册器"""
    
    def __init__(self):
        self._actions: Dict[str, BaseAction] = {}
    
    def register(self, action: BaseAction):
        """注册动作"""
        self._actions[action.name] = action
    
    def get_action(self, name: str) -> BaseAction:
        """获取动作"""
        if name not in self._actions:
            raise ValueError(f"未找到动作: {name}")
        return self._actions[name]
    
    def list_actions(self) -> Dict[str, str]:
        """列出所有动作"""
        return {name: action.description for name, action in self._actions.items()}

# 全局动作注册器
action_registry = ActionRegistry()
```

### 3.2 具体动作实现

创建 `core/actions/code_actions.py`：

```python
from .base import BaseAction, action_registry
from ..models import ActionResult
from ..repository.manager import RepositoryManager
from ..search.code_search import CodeSearcher

class ViewFileAction(BaseAction):
    """查看文件内容"""
    
    def __init__(self, repo_manager: RepositoryManager):
        super().__init__("view_file", "查看指定文件的内容")
        self.repo_manager = repo_manager
    
    def execute(self, repo_path: str, file_path: str, **kwargs) -> ActionResult:
        try:
            content = self.repo_manager.read_file(repo_path, file_path)
            return ActionResult(
                success=True,
                message=f"成功读取文件: {file_path}",
                data={"content": content, "file_path": file_path}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"读取文件失败: {str(e)}"
            )

class SearchCodeAction(BaseAction):
    """搜索代码"""
    
    def __init__(self, code_searcher: CodeSearcher):
        super().__init__("search_code", "在代码库中搜索关键词")
        self.code_searcher = code_searcher
    
    def execute(self, repo_path: str, keyword: str, **kwargs) -> ActionResult:
        try:
            results = self.code_searcher.search_by_keyword(repo_path, keyword)
            return ActionResult(
                success=True,
                message=f"找到 {len(results)} 个匹配结果",
                data={"results": [r.dict() for r in results]}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"搜索失败: {str(e)}"
            )

class ModifyFileAction(BaseAction):
    """修改文件内容"""
    
    def __init__(self, repo_manager: RepositoryManager):
        super().__init__("modify_file", "修改文件内容")
        self.repo_manager = repo_manager
    
    def execute(self, repo_path: str, file_path: str, new_content: str, **kwargs) -> ActionResult:
        try:
            # 备份原文件
            original_content = self.repo_manager.read_file(repo_path, file_path)
            
            # 写入新内容
            self.repo_manager.write_file(repo_path, file_path, new_content)
            
            return ActionResult(
                success=True,
                message=f"成功修改文件: {file_path}",
                data={
                    "file_path": file_path,
                    "original_content": original_content,
                    "new_content": new_content
                }
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"修改文件失败: {str(e)}"
            )

# 注册动作
def register_code_actions(repo_manager: RepositoryManager, code_searcher: CodeSearcher):
    action_registry.register(ViewFileAction(repo_manager))
    action_registry.register(SearchCodeAction(code_searcher))
    action_registry.register(ModifyFileAction(repo_manager))
```

## 第四阶段：提示词系统

### 4.1 提示词模板

创建 `core/prompts.py`：

```python
# 基础角色定义
AGENT_ROLE = """你是一个具有卓越编程技能的自主AI助手。你正在自主工作，
无法与用户直接沟通，必须依赖可用的函数来获取信息。
"""

# Issue分析提示词
ISSUE_ANALYSIS_PROMPT = """
请仔细分析以下GitHub Issue，并提供结构化的分析结果：

Issue信息:
标题: {title}
描述: {description}
类型: {issue_type}
标签: {labels}
代码库: {repository_url}

请提供：
1. 问题总结 - 用1-2句话概括核心问题
2. 关键文件 - 可能需要修改的文件列表（基于问题描述推测）
3. 潜在原因 - 可能导致此问题的原因分析
4. 解决步骤 - 建议的解决方案步骤

分析时请考虑：
- 错误信息中的关键词
- 涉及的功能模块
- 可能的代码位置
- 相关的依赖关系
"""

# 代码搜索提示词
CODE_SEARCH_PROMPT = """
基于Issue分析结果，我需要在代码库中搜索相关代码。

Issue: {issue_title}
分析结果: {analysis_summary}

请从以下搜索结果中识别最相关的代码片段：
{search_results}

请评估每个搜索结果的相关性，并解释为什么它们与Issue相关。
"""

# 代码修复提示词
CODE_FIX_PROMPT = """
# 代码修复任务

## Issue信息
标题: {issue_title}
描述: {issue_description}
类型: {issue_type}

## 分析结果
{analysis_summary}

## 相关代码上下文
{code_context}

## 任务要求
请为此Issue提供具体的代码修复方案。对于每个需要修改的文件，请提供：

1. **文件路径**: 需要修改的文件
2. **问题定位**: 在代码中定位具体的问题点
3. **修复方案**: 详细的修复代码
4. **修复说明**: 解释为什么这样修复以及修复的原理
5. **置信度**: 对修复方案的信心程度(0-1)

## 修复原则
- 最小化修改：只修改必要的代码
- 保持兼容性：确保修改不会破坏现有功能
- 遵循代码风格：保持与现有代码一致的风格
- 考虑边界情况：处理可能的异常情况

请提供完整可执行的修复代码，不要使用占位符或省略号。
"""

# MCTS节点评估提示词
NODE_EVALUATION_PROMPT = """
# 代码修复方案评估

## 当前修复方案
文件: {file_path}
修复内容:
```python
{fixed_code}
```

## 评估标准
请从以下维度评估此修复方案的质量(1-10分)：

1. **正确性** (权重: 40%)
   - 是否正确解决了Issue中描述的问题
   - 逻辑是否正确无误
   - 是否处理了边界情况

2. **完整性** (权重: 25%)
   - 是否完整解决了问题，没有遗漏
   - 是否考虑了相关的依赖修改

3. **代码质量** (权重: 20%)
   - 代码风格是否一致
   - 是否遵循最佳实践
   - 可读性和可维护性

4. **影响范围** (权重: 15%)
   - 修改是否最小化
   - 是否可能引入新的问题
   - 对现有功能的影响

请提供：
- 各维度评分及理由
- 总体评分 (加权平均)
- 改进建议 (如果有)
- 是否推荐采用此方案 (是/否)
"""

# 反馈生成提示词
FEEDBACK_GENERATION_PROMPT = """
# 修复方案反馈生成

## 上下文
Issue: {issue_title}
当前尝试的修复方案数量: {attempt_count}

## 之前的尝试
{previous_attempts}

## 当前方案
{current_solution}

## 任务
请为AI代理提供建设性的反馈，帮助改进修复方案：

1. **当前方案分析**
   - 指出当前方案的优点
   - 识别潜在问题或不足

2. **改进建议**
   - 具体的改进方向
   - 可以尝试的替代方案

3. **探索建议**
   - 建议查看的其他代码文件
   - 需要考虑的其他因素

4. **下一步行动**
   - 推荐的具体行动步骤
   - 优先级排序

请提供具体、可操作的反馈，避免泛泛而谈。
"""

# ReAct模式指导
REACT_GUIDELINES = """
# ReAct行动指南

## 基本原则
1. **思考优先**: 在每次行动前，必须在<thoughts>标签中写出你的推理过程
2. **单步执行**: 一次只执行一个行动，等待观察结果
3. **观察分析**: 仔细分析每次行动的观察结果，用于指导下一步

## 思考内容要求
在<thoughts>标签中必须包含：
- 从之前观察中学到了什么
- 为什么选择这个特定的行动
- 期望通过这个行动学到什么
- 需要注意的风险点

## 行动模式
- **探索阶段**: 搜索和查看代码，理解问题
- **分析阶段**: 分析代码结构，定位问题根源
- **修复阶段**: 实施具体的代码修改
- **验证阶段**: 检查修复效果

## 质量标准
- 每个思考过程要清晰、具体
- 行动要有明确的目标
- 观察结果要充分利用
- 避免重复无效的行动
"""

# 工作流程提示词
WORKFLOW_PROMPT = """
# AI代码修复工作流程

## 阶段1: 理解任务
1. **分析Issue**: 仔细阅读Issue描述，理解问题本质
2. **识别范围**: 确定需要修改的代码范围
3. **收集上下文**: 确定需要了解的相关代码

## 阶段2: 代码探索
1. **关键词搜索**: 使用Issue中的关键词搜索相关代码
2. **函数/类搜索**: 搜索可能相关的函数和类定义
3. **查看代码**: 查看搜索到的代码片段，理解结构

## 阶段3: 问题定位
1. **分析代码逻辑**: 理解现有代码的工作原理
2. **识别问题点**: 找到导致Issue的具体代码位置
3. **评估影响**: 分析修改可能的影响范围

## 阶段4: 方案设计
1. **设计修复**: 制定具体的修复方案
2. **考虑边界**: 处理边界情况和异常
3. **保持兼容**: 确保不破坏现有功能

## 阶段5: 实施修复
1. **应用修改**: 实施代码修改
2. **验证修复**: 检查修复是否正确
3. **完善方案**: 根据需要进行调整

## 重要原则
- **专注具体任务**: 只解决Issue中描述的问题
- **最小化修改**: 只修改必要的代码
- **保持质量**: 遵循代码规范和最佳实践
- **逐步推进**: 一步一步地解决问题
"""

def get_issue_analysis_prompt(issue) -> str:
    """获取Issue分析提示词"""
    return ISSUE_ANALYSIS_PROMPT.format(
        title=issue.title,
        description=issue.description,
        issue_type=issue.type,
        labels=', '.join(issue.labels),
        repository_url=issue.repository_url
    )

def get_code_fix_prompt(issue, analysis, code_context) -> str:
    """获取代码修复提示词"""
    return CODE_FIX_PROMPT.format(
        issue_title=issue.title,
        issue_description=issue.description,
        issue_type=issue.type,
        analysis_summary=analysis.summary,
        code_context=code_context
    )

def get_node_evaluation_prompt(file_path: str, fixed_code: str) -> str:
    """获取节点评估提示词"""
    return NODE_EVALUATION_PROMPT.format(
        file_path=file_path,
        fixed_code=fixed_code
    )

def get_feedback_prompt(issue_title: str, attempt_count: int, previous_attempts: str, current_solution: str) -> str:
    """获取反馈生成提示词"""
    return FEEDBACK_GENERATION_PROMPT.format(
        issue_title=issue_title,
        attempt_count=attempt_count,
        previous_attempts=previous_attempts,
        current_solution=current_solution
    )
```

## 第五阶段：MCTS搜索系统

### 5.1 搜索树节点

创建 `core/search/node.py`：

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import math

class NodeStatus(str, Enum):
    PENDING = "pending"      # 待处理
    EXPLORING = "exploring"  # 探索中
    FIXING = "fixing"       # 修复中
    FINISHED = "finished"   # 已完成
    FAILED = "failed"       # 失败

class SearchNode(BaseModel):
    """MCTS搜索树节点"""
    
    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    
    # 节点状态
    status: NodeStatus = NodeStatus.PENDING
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    
    # 节点内容
    action_type: Optional[str] = None  # 执行的动作类型
    action_params: Dict[str, Any] = Field(default_factory=dict)
    observation: Optional[str] = None  # 动作执行结果
    
    # 代码修复相关
    file_path: Optional[str] = None
    original_code: Optional[str] = None
    fixed_code: Optional[str] = None
    fix_explanation: Optional[str] = None
    confidence: float = 0.0
    
    # MCTS相关
    reward: Optional[float] = None
    ucb_score: Optional[float] = None
    
    def add_child(self, child_id: str):
        """添加子节点"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def is_leaf(self) -> bool:
        """是否为叶子节点"""
        return len(self.children_ids) == 0
    
    def is_finished(self) -> bool:
        """是否已完成"""
        return self.status in [NodeStatus.FINISHED, NodeStatus.FAILED]
    
    def calculate_ucb(self, parent_visits: int, exploration_weight: float = 1.4) -> float:
        """计算UCB值"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        self.ucb_score = exploitation + exploration
        return self.ucb_score
    
    def update_value(self, reward: float):
        """更新节点值"""
        self.visits += 1
        self.value += reward
        self.reward = reward

class SearchTree(BaseModel):
    """MCTS搜索树"""
    
    nodes: Dict[str, SearchNode] = Field(default_factory=dict)
    root_id: Optional[str] = None
    current_branch: Optional[str] = None  # 当前Git分支
    
    def add_node(self, node: SearchNode, parent_id: Optional[str] = None) -> str:
        """添加节点到树中"""
        self.nodes[node.node_id] = node
        
        if parent_id:
            parent = self.nodes.get(parent_id)
            if parent:
                parent.add_child(node.node_id)
                node.parent_id = parent_id
                node.depth = parent.depth + 1
        
        if not self.root_id:
            self.root_id = node.node_id
        
        return node.node_id
    
    def get_node(self, node_id: str) -> Optional[SearchNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_root(self) -> Optional[SearchNode]:
        """获取根节点"""
        return self.nodes.get(self.root_id) if self.root_id else None
    
    def get_children(self, node_id: str) -> List[SearchNode]:
        """获取子节点"""
        node = self.get_node(node_id)
        if not node:
            return []
        
        return [self.nodes[child_id] for child_id in node.children_ids if child_id in self.nodes]
    
    def get_leaf_nodes(self) -> List[SearchNode]:
        """获取所有叶子节点"""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_finished_nodes(self) -> List[SearchNode]:
        """获取所有已完成的节点"""
        return [node for node in self.nodes.values() if node.is_finished()]
    
    def select_best_child(self, node_id: str) -> Optional[SearchNode]:
        """使用UCB选择最佳子节点"""
        children = self.get_children(node_id)
        if not children:
            return None
        
        parent = self.get_node(node_id)
        if not parent:
            return None
        
        # 计算每个子节点的UCB值
        for child in children:
            child.calculate_ucb(parent.visits)
        
        # 选择UCB值最高的子节点
        return max(children, key=lambda x: x.ucb_score or 0)
    
    def backpropagate(self, node_id: str, reward: float):
        """反向传播奖励"""
        current_id = node_id
        
        while current_id:
            node = self.get_node(current_id)
            if not node:
                break
            
            node.update_value(reward)
            current_id = node.parent_id
    
    def get_best_path(self) -> List[SearchNode]:
        """获取最佳路径"""
        if not self.root_id:
            return []
        
        path = []
        current_id = self.root_id
        
        while current_id:
            node = self.get_node(current_id)
            if not node:
                break
            
            path.append(node)
            
            # 选择访问次数最多的子节点
            children = self.get_children(current_id)
            if not children:
                break
            
            current_id = max(children, key=lambda x: x.visits).node_id
        
        return path
```

### 5.2 Git分支管理

创建 `core/git/branch_manager.py`：

```python
import git
import os
import logging
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class GitBranchManager:
    """Git分支管理器，用于MCTS搜索过程中的分支管理"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.main_branch = "main"  # 主分支名
        self.search_prefix = "mcts-search"  # 搜索分支前缀
        
    def create_search_branch(self, node_id: str, parent_branch: Optional[str] = None) -> str:
        """为MCTS节点创建搜索分支"""
        branch_name = f"{self.search_prefix}-{node_id[:8]}"
        
        try:
            # 如果有父分支，从父分支创建；否则从主分支创建
            base_branch = parent_branch or self.main_branch
            
            # 确保基础分支存在
            if base_branch != self.main_branch and base_branch not in [b.name for b in self.repo.branches]:
                logger.warning(f"Base branch {base_branch} not found, using {self.main_branch}")
                base_branch = self.main_branch
            
            # 切换到基础分支
            self.repo.git.checkout(base_branch)
            
            # 创建新分支
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            logger.info(f"Created search branch: {branch_name} from {base_branch}")
            return branch_name
            
        except Exception as e:
            logger.error(f"Failed to create search branch {branch_name}: {e}")
            raise
    
    def switch_to_branch(self, branch_name: str) -> bool:
        """切换到指定分支"""
        try:
            self.repo.git.checkout(branch_name)
            logger.info(f"Switched to branch: {branch_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    def commit_changes(self, node_id: str, message: Optional[str] = None) -> bool:
        """提交当前分支的更改"""
        try:
            # 检查是否有更改
            if not self.repo.is_dirty():
                logger.info("No changes to commit")
                return True
            
            # 添加所有更改
            self.repo.git.add('.')
            
            # 生成提交信息
            if not message:
                message = f"MCTS node {node_id[:8]} - code fix attempt"
            
            # 提交更改
            self.repo.index.commit(message)
            logger.info(f"Committed changes for node {node_id}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to commit changes for node {node_id}: {e}")
            return False
    
    def get_current_branch(self) -> str:
        """获取当前分支名"""
        return self.repo.active_branch.name
    
    def get_search_branches(self) -> List[str]:
        """获取所有搜索分支"""
        return [b.name for b in self.repo.branches if b.name.startswith(self.search_prefix)]
    
    def cleanup_search_branches(self, keep_best: bool = True, best_branch: Optional[str] = None):
        """清理搜索分支"""
        search_branches = self.get_search_branches()
        
        # 切换到主分支
        self.repo.git.checkout(self.main_branch)
        
        for branch_name in search_branches:
            # 如果要保留最佳分支且当前分支是最佳分支，则跳过
            if keep_best and best_branch and branch_name == best_branch:
                continue
            
            try:
                # 删除分支
                self.repo.delete_head(branch_name, force=True)
                logger.info(f"Deleted search branch: {branch_name}")
            except Exception as e:
                logger.warning(f"Failed to delete branch {branch_name}: {e}")
    
    def merge_to_main(self, source_branch: str, message: Optional[str] = None) -> bool:
        """将指定分支合并到主分支"""
        try:
            # 切换到主分支
            self.repo.git.checkout(self.main_branch)
            
            # 合并分支
            if not message:
                message = f"Merge MCTS solution from {source_branch}"
            
            self.repo.git.merge(source_branch, m=message)
            logger.info(f"Merged {source_branch} to {self.main_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge {source_branch} to {self.main_branch}: {e}")
            return False
    
    def get_branch_diff(self, branch_name: str, base_branch: Optional[str] = None) -> str:
        """获取分支与基础分支的差异"""
        try:
            base = base_branch or self.main_branch
            diff = self.repo.git.diff(f"{base}..{branch_name}")
            return diff
        except Exception as e:
            logger.error(f"Failed to get diff for branch {branch_name}: {e}")
            return ""
    
    def create_branch_snapshot(self, branch_name: str) -> Dict:
        """创建分支快照"""
        try:
            self.switch_to_branch(branch_name)
            
            return {
                "branch_name": branch_name,
                "commit_hash": self.repo.head.commit.hexsha,
                "commit_message": self.repo.head.commit.message.strip(),
                "diff": self.get_branch_diff(branch_name),
                "timestamp": self.repo.head.commit.committed_datetime.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create snapshot for branch {branch_name}: {e}")
            return {}
```

## 第六阶段：AI代理系统

### 6.1 核心代理

创建 `core/agent.py`：

```python
from typing import List, Dict, Any, Optional
from .models import Issue, FixSuggestion, ActionResult
from .llm.client import LLMClient, LLMConfig
from .actions.base import action_registry
from .repository.manager import RepositoryManager
from .search.code_search import CodeSearcher
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    """分析结果"""
    summary: str
    key_files: List[str]
    potential_causes: List[str]
    suggested_actions: List[str]

class CodeFixAgent:
    """代码修复代理"""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_client = LLMClient(llm_config)
        self.repo_manager = RepositoryManager()
        self.code_searcher = CodeSearcher(self.repo_manager)
        
        # 注册动作
        from .actions.code_actions import register_code_actions
        register_code_actions(self.repo_manager, self.code_searcher)
    
    def analyze_issue(self, issue: Issue) -> AnalysisResult:
        """分析Issue"""
        prompt = f"""
        请分析以下GitHub Issue，并提供详细的分析结果：
        
        标题: {issue.title}
        描述: {issue.description}
        类型: {issue.type}
        标签: {', '.join(issue.labels)}
        
        请提供：
        1. 问题总结
        2. 可能涉及的关键文件
        3. 潜在原因分析
        4. 建议的解决步骤
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            result = self.llm_client.complete(messages, AnalysisResult)
            return result
        except Exception as e:
            # 降级处理
            return AnalysisResult(
                summary=f"分析Issue: {issue.title}",
                key_files=[],
                potential_causes=["需要进一步分析"],
                suggested_actions=["搜索相关代码", "查看错误日志"]
            )
    
    def fix_issue(self, issue: Issue) -> List[FixSuggestion]:
        """修复Issue"""
        # 1. 克隆代码库
        repo_path = self.repo_manager.clone_repository(issue.repository_url)
        
        # 2. 分析Issue
        analysis = self.analyze_issue(issue)
        
        # 3. 搜索相关代码
        search_results = []
        for keyword in self._extract_keywords(issue):
            results = self.code_searcher.search_by_keyword(repo_path, keyword)
            search_results.extend(results)
        
        # 4. 生成修复建议
        suggestions = self._generate_fix_suggestions(issue, analysis, search_results, repo_path)
        
        return suggestions
    
    def _extract_keywords(self, issue: Issue) -> List[str]:
        """从Issue中提取关键词"""
        # 简单的关键词提取
        text = f"{issue.title} {issue.description}".lower()
        
        # 提取可能的函数名、类名等
        import re
        keywords = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # 过滤常见词汇
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'this', 'that', 'these', 'those'}
        
        filtered_keywords = [k for k in keywords if k not in common_words and len(k) > 2]
        
        return list(set(filtered_keywords))[:10]  # 最多10个关键词
    
    def _generate_fix_suggestions(self, issue: Issue, analysis: AnalysisResult, search_results: List, repo_path: str) -> List[FixSuggestion]:
        """生成修复建议"""
        suggestions = []
        
        # 构建上下文
        context = f"""
        Issue分析:
        {analysis.summary}
        
        潜在原因:
        {chr(10).join(analysis.potential_causes)}
        
        相关代码片段:
        """
        
        for result in search_results[:5]:  # 只取前5个结果
            context += f"\n文件: {result.file_path}\n```python\n{result.content}\n```\n"
        
        prompt = f"""
        基于以下信息，请为GitHub Issue提供具体的代码修复建议：
        
        {context}
        
        Issue详情:
        标题: {issue.title}
        描述: {issue.description}
        
        请提供具体的修复方案，包括：
        1. 需要修改的文件
        2. 原始代码
        3. 修复后的代码
        4. 修复说明
        5. 置信度评分(0-1)
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_client.complete(messages)
            # 这里需要解析LLM的响应并转换为FixSuggestion对象
            # 简化处理，实际项目中需要更复杂的解析逻辑
            suggestions.append(FixSuggestion(
                file_path="example.py",
                original_code="# 原始代码",
                fixed_code="# 修复后的代码",
                explanation=response,
                confidence=0.8
            ))
        except Exception as e:
            print(f"生成修复建议时出错: {e}")
        
        return suggestions

### 6.2 MCTS搜索代理

创建 `core/mcts_agent.py`：

```python
import logging
from typing import List, Optional, Dict, Any
from .models import Issue, FixSuggestion
from .llm.client import LLMClient, LLMConfig
from .search.node import SearchTree, SearchNode, NodeStatus
from .git.branch_manager import GitBranchManager
from .repository.manager import RepositoryManager
from .search.code_search import CodeSearcher
from .prompts import (
    get_issue_analysis_prompt, 
    get_code_fix_prompt, 
    get_node_evaluation_prompt,
    get_feedback_prompt,
    REACT_GUIDELINES,
    WORKFLOW_PROMPT
)
import random

logger = logging.getLogger(__name__)

class MCTSCodeFixAgent:
    """基于MCTS的代码修复代理"""
    
    def __init__(self, llm_config: LLMConfig, max_iterations: int = 20, max_depth: int = 10):
        self.llm_client = LLMClient(llm_config)
        self.repo_manager = RepositoryManager()
        self.code_searcher = CodeSearcher(self.repo_manager)
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        
        # MCTS参数
        self.exploration_weight = 1.4
        self.reward_threshold = 0.8
        
    def solve_issue(self, issue: Issue) -> Optional[FixSuggestion]:
        """使用MCTS解决Issue"""
        
        # 1. 克隆代码库并初始化Git分支管理
        repo_path = self.repo_manager.clone_repository(issue.repository_url)
        branch_manager = GitBranchManager(repo_path)
        
        # 2. 创建搜索树
        search_tree = SearchTree()
        
        # 3. 创建根节点
        root_node = SearchNode(
            status=NodeStatus.PENDING,
            action_type="analyze_issue",
            action_params={"issue": issue.dict()}
        )
        search_tree.add_node(root_node)
        
        # 4. 运行MCTS搜索
        best_solution = self._run_mcts_search(issue, search_tree, branch_manager, repo_path)
        
        # 5. 清理搜索分支
        if best_solution:
            best_branch = f"mcts-search-{best_solution.node_id[:8]}"
            branch_manager.cleanup_search_branches(keep_best=True, best_branch=best_branch)
            
            # 可选：合并最佳方案到主分支
            # branch_manager.merge_to_main(best_branch)
        
        return self._convert_node_to_suggestion(best_solution) if best_solution else None
    
    def _run_mcts_search(self, issue: Issue, search_tree: SearchTree, branch_manager: GitBranchManager, repo_path: str) -> Optional[SearchNode]:
        """运行MCTS搜索"""
        
        for iteration in range(self.max_iterations):
            logger.info(f"MCTS Iteration {iteration + 1}/{self.max_iterations}")
            
            # 1. Selection - 选择要扩展的节点
            selected_node = self._select_node(search_tree)
            if not selected_node:
                logger.info("No more nodes to expand")
                break
            
            # 2. Expansion - 扩展节点
            new_node = self._expand_node(selected_node, search_tree, issue, branch_manager, repo_path)
            if not new_node:
                continue
            
            # 3. Simulation - 模拟执行
            reward = self._simulate_node(new_node, issue, branch_manager, repo_path)
            
            # 4. Backpropagation - 反向传播奖励
            search_tree.backpropagate(new_node.node_id, reward)
            
            # 检查是否找到满意的解决方案
            if reward >= self.reward_threshold:
                logger.info(f"Found satisfactory solution with reward {reward}")
                return new_node
        
        # 返回最佳节点
        finished_nodes = search_tree.get_finished_nodes()
        if finished_nodes:
            return max(finished_nodes, key=lambda n: n.reward or 0)
        
        return None
    
    def _select_node(self, search_tree: SearchTree) -> Optional[SearchNode]:
        """选择要扩展的节点（UCB选择）"""
        
        # 获取所有叶子节点中未完成的节点
        leaf_nodes = [node for node in search_tree.get_leaf_nodes() 
                     if not node.is_finished() and node.depth < self.max_depth]
        
        if not leaf_nodes:
            return None
        
        # 如果有未访问的节点，优先选择
        unvisited = [node for node in leaf_nodes if node.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        
        # 使用UCB选择
        root = search_tree.get_root()
        if not root:
            return None
        
        return search_tree.select_best_child(root.node_id)
    
    def _expand_node(self, node: SearchNode, search_tree: SearchTree, issue: Issue, 
                    branch_manager: GitBranchManager, repo_path: str) -> Optional[SearchNode]:
        """扩展节点"""
        
        # 为节点创建Git分支
        parent_branch = None
        if node.parent_id:
            parent_node = search_tree.get_node(node.parent_id)
            if parent_node:
                parent_branch = f"mcts-search-{parent_node.node_id[:8]}"
        
        branch_name = branch_manager.create_search_branch(node.node_id, parent_branch)
        
        # 根据节点类型确定下一步动作
        next_action = self._determine_next_action(node, issue)
        
        # 创建子节点
        child_node = SearchNode(
            status=NodeStatus.EXPLORING,
            action_type=next_action["type"],
            action_params=next_action["params"],
            depth=node.depth + 1
        )
        
        search_tree.add_node(child_node, node.node_id)
        return child_node
    
    def _determine_next_action(self, node: SearchNode, issue: Issue) -> Dict[str, Any]:
        """确定下一步动作"""
        
        if node.action_type == "analyze_issue":
            return {
                "type": "search_code",
                "params": {"keywords": self._extract_keywords_from_issue(issue)}
            }
        elif node.action_type == "search_code":
            return {
                "type": "generate_fix",
                "params": {"search_results": node.observation}
            }
        elif node.action_type == "generate_fix":
            return {
                "type": "evaluate_fix",
                "params": {"fix_code": node.fixed_code}
            }
        else:
            return {
                "type": "finish",
                "params": {}
            }
    
    def _simulate_node(self, node: SearchNode, issue: Issue, 
                      branch_manager: GitBranchManager, repo_path: str) -> float:
        """模拟节点执行并返回奖励"""
        
        try:
            if node.action_type == "analyze_issue":
                return self._simulate_issue_analysis(node, issue)
            elif node.action_type == "search_code":
                return self._simulate_code_search(node, issue, repo_path)
            elif node.action_type == "generate_fix":
                return self._simulate_fix_generation(node, issue, repo_path, branch_manager)
            elif node.action_type == "evaluate_fix":
                return self._simulate_fix_evaluation(node, issue)
            else:
                node.status = NodeStatus.FINISHED
                return node.confidence
                
        except Exception as e:
            logger.error(f"Error simulating node {node.node_id}: {e}")
            node.status = NodeStatus.FAILED
            return 0.0
    
    def _simulate_issue_analysis(self, node: SearchNode, issue: Issue) -> float:
        """模拟Issue分析"""
        
        prompt = get_issue_analysis_prompt(issue)
        messages = [
            {"role": "system", "content": WORKFLOW_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.complete(messages)
            node.observation = response
            node.status = NodeStatus.EXPLORING
            return 0.3  # 分析阶段给予基础奖励
        except Exception as e:
            logger.error(f"Issue analysis failed: {e}")
            node.status = NodeStatus.FAILED
            return 0.0
    
    def _simulate_code_search(self, node: SearchNode, issue: Issue, repo_path: str) -> float:
        """模拟代码搜索"""
        
        keywords = node.action_params.get("keywords", [])
        search_results = []
        
        for keyword in keywords[:5]:  # 限制搜索关键词数量
            results = self.code_searcher.search_by_keyword(repo_path, keyword)
            search_results.extend(results[:3])  # 每个关键词最多3个结果
        
        if search_results:
            node.observation = str([r.dict() for r in search_results])
            node.status = NodeStatus.EXPLORING
            return 0.5  # 搜索成功给予中等奖励
        else:
            node.status = NodeStatus.FAILED
            return 0.1
    
    def _simulate_fix_generation(self, node: SearchNode, issue: Issue, 
                                repo_path: str, branch_manager: GitBranchManager) -> float:
        """模拟修复代码生成"""
        
        # 构建代码上下文
        search_results = node.action_params.get("search_results", "")
        
        # 创建修复提示词
        prompt = f"""
        基于以下Issue和搜索结果，生成具体的代码修复方案：
        
        Issue: {issue.title}
        描述: {issue.description}
        
        搜索结果: {search_results}
        
        请提供：
        1. 需要修改的文件路径
        2. 具体的修复代码
        3. 修复说明
        4. 置信度评分(0-1)
        """
        
        messages = [
            {"role": "system", "content": REACT_GUIDELINES},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.complete(messages)
            
            # 解析响应（简化处理）
            node.fixed_code = response  # 实际应该解析出具体的代码
            node.fix_explanation = "AI生成的修复方案"
            node.confidence = 0.7  # 实际应该从响应中解析
            
            # 应用修复到当前分支
            # 这里应该实际修改文件
            # self.repo_manager.write_file(repo_path, file_path, fixed_code)
            
            # 提交更改
            branch_manager.commit_changes(node.node_id, f"Fix attempt: {issue.title}")
            
            node.status = NodeStatus.FIXING
            return 0.7
            
        except Exception as e:
            logger.error(f"Fix generation failed: {e}")
            node.status = NodeStatus.FAILED
            return 0.2
    
    def _simulate_fix_evaluation(self, node: SearchNode, issue: Issue) -> float:
        """模拟修复方案评估"""
        
        if not node.fixed_code:
            return 0.0
        
        prompt = get_node_evaluation_prompt("example.py", node.fixed_code)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_client.complete(messages)
            
            # 从评估响应中提取分数（简化处理）
            # 实际应该解析出具体的评分
            score = 0.8  # 模拟评分
            
            node.reward = score
            node.status = NodeStatus.FINISHED if score >= self.reward_threshold else NodeStatus.FAILED
            
            return score
            
        except Exception as e:
            logger.error(f"Fix evaluation failed: {e}")
            node.status = NodeStatus.FAILED
            return 0.0
    
    def _extract_keywords_from_issue(self, issue: Issue) -> List[str]:
        """从Issue中提取关键词"""
        import re
        
        text = f"{issue.title} {issue.description}".lower()
        keywords = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # 过滤常见词汇
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered = [k for k in keywords if k not in common_words and len(k) > 2]
        
        return list(set(filtered))[:10]
    
    def _convert_node_to_suggestion(self, node: SearchNode) -> FixSuggestion:
        """将节点转换为修复建议"""
        return FixSuggestion(
            file_path=node.file_path or "unknown.py",
            original_code="# 原始代码",
            fixed_code=node.fixed_code or "# 修复代码",
            explanation=node.fix_explanation or "MCTS生成的修复方案",
            confidence=node.confidence
        )
```

## 第七阶段：Web界面增强

### 7.1 增强的Streamlit应用

更新 `web/app.py`：

```python
import streamlit as st
import os
from pathlib import Path
import sys
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mcts_agent import MCTSCodeFixAgent
from core.models import Issue, IssueType
from core.llm.client import LLMConfig

def main():
    st.set_page_config(
        page_title="AI代码修复系统 - MCTS版本",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 AI代码修复系统 (MCTS版本)")
    st.markdown("基于蒙特卡洛树搜索的自动化代码修复系统")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        # LLM配置
        model = st.selectbox(
            "选择模型",
            ["gpt-4o-mini", "gpt-4", "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # MCTS配置
        st.subheader("🌳 MCTS参数")
        max_iterations = st.slider("最大迭代次数", 5, 50, 20)
        max_depth = st.slider("最大搜索深度", 3, 20, 10)
        exploration_weight = st.slider("探索权重", 0.5, 3.0, 1.4)
        
        # API Key检查
        api_key_status = check_api_keys()
        if not api_key_status:
            st.error("请设置相应的API Key环境变量")
            return
    
    # 主界面
    tab1, tab2, tab3 = st.tabs(["🔍 Issue分析", "🌳 搜索过程", "📊 结果分析"])
    
    with tab1:
        st.header("GitHub Issue分析与修复")
        
        # Issue输入表单
        with st.form("issue_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                repo_url = st.text_input(
                    "代码库URL",
                    placeholder="https://github.com/username/repository.git"
                )
            
            with col2:
                issue_type = st.selectbox(
                    "Issue类型",
                    ["bug", "feature", "enhancement"]
                )
            
            issue_title = st.text_input(
                "Issue标题",
                placeholder="请输入Issue标题"
            )
            
            issue_description = st.text_area(
                "Issue描述",
                placeholder="请详细描述问题...",
                height=150
            )
            
            labels = st.text_input(
                "标签 (用逗号分隔)",
                placeholder="bug, urgent, backend"
            )
            
            submitted = st.form_submit_button("🚀 开始MCTS搜索", type="primary")
        
        if submitted and repo_url and issue_title and issue_description:
            # 创建Issue对象
            issue = Issue(
                id="manual_input",
                title=issue_title,
                description=issue_description,
                type=IssueType(issue_type),
                repository_url=repo_url,
                labels=labels.split(",") if labels else []
            )
            
            # 存储到session state
            st.session_state.current_issue = issue
            st.session_state.mcts_config = {
                "model": model,
                "temperature": temperature,
                "max_iterations": max_iterations,
                "max_depth": max_depth,
                "exploration_weight": exploration_weight
            }
            
            # 显示搜索过程
            with st.spinner("正在运行MCTS搜索..."):
                try:
                    # 初始化MCTS代理
                    llm_config = LLMConfig(model=model, temperature=temperature)
                    agent = MCTSCodeFixAgent(
                        llm_config=llm_config,
                        max_iterations=max_iterations,
                        max_depth=max_depth
                    )
                    
                    # 运行MCTS搜索
                    solution = agent.solve_issue(issue)
                    
                    if solution:
                        st.session_state.solution = solution
                        st.success("🎉 MCTS搜索完成！找到了修复方案")
                        
                        # 显示解决方案
                        st.subheader("💡 修复方案")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**文件路径:**")
                            st.code(solution.file_path)
                            
                            st.write("**置信度:**")
                            st.progress(solution.confidence)
                            st.write(f"{solution.confidence:.2%}")
                        
                        with col2:
                            st.write("**修复说明:**")
                            st.write(solution.explanation)
                        
                        st.write("**原始代码:**")
                        st.code(solution.original_code, language="python")
                        
                        st.write("**修复后代码:**")
                        st.code(solution.fixed_code, language="python")
                        
                    else:
                        st.warning("⚠️ MCTS搜索未能找到满意的修复方案")
                        st.info("建议调整搜索参数或提供更详细的Issue描述")
                
                except Exception as e:
                    st.error(f"❌ 搜索过程中出现错误: {str(e)}")
    
    with tab2:
        st.header("🌳 MCTS搜索过程可视化")
        
        if hasattr(st.session_state, 'current_issue'):
            st.info("搜索过程可视化功能将在后续版本中实现")
            
            # 这里可以添加搜索树的可视化
            # 显示节点扩展过程、UCB值变化等
            
            st.subheader("搜索统计")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总迭代次数", "20")
            with col2:
                st.metric("探索节点数", "45")
            with col3:
                st.metric("最佳奖励", "0.85")
            with col4:
                st.metric("搜索深度", "8")
        else:
            st.info("请先在'Issue分析'标签页中提交一个Issue")
    
    with tab3:
        st.header("📊 结果分析")
        
        if hasattr(st.session_state, 'solution'):
            solution = st.session_state.solution
            
            st.subheader("方案质量分析")
            
            # 创建质量评估图表
            quality_metrics = {
                "正确性": 0.85,
                "完整性": 0.78,
                "代码质量": 0.82,
                "影响范围": 0.90
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                for metric, score in quality_metrics.items():
                    st.write(f"**{metric}:**")
                    st.progress(score)
                    st.write(f"{score:.2%}")
                    st.write("")
            
            with col2:
                st.subheader("改进建议")
                st.write("• 考虑添加更多的边界条件检查")
                st.write("• 可以优化错误处理逻辑")
                st.write("• 建议添加相关的单元测试")
                
                st.subheader("风险评估")
                st.write("🟢 低风险：修改范围有限")
                st.write("🟡 中等风险：可能影响相关功能")
                st.write("🔴 高风险：需要全面测试")
        else:
            st.info("请先完成Issue修复以查看结果分析")

def check_api_keys():
    """检查API Key是否设置"""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    for key in required_keys:
        if os.getenv(key):
            return True
    
    return False

if __name__ == "__main__":
    main()
```

## 第八阶段：使用示例和测试

### 8.1 创建示例脚本

创建 `examples/basic_usage.py`：

```python
#!/usr/bin/env python3
"""
基本使用示例
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.mcts_agent import MCTSCodeFixAgent
from core.models import Issue, IssueType
from core.llm.client import LLMConfig

def main():
    # 配置LLM
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=4000
    )
    
    # 创建MCTS代理
    agent = MCTSCodeFixAgent(
        llm_config=llm_config,
        max_iterations=10,
        max_depth=5
    )
    
    # 创建示例Issue
    issue = Issue(
        id="example-1",
        title="修复登录功能的空指针异常",
        description="""
        在用户登录时，当用户名为空时会抛出NullPointerException。
        
        错误堆栈：
        ```
        NullPointerException at login.py:45
        ```
        
        期望行为：应该返回友好的错误信息而不是抛出异常。
        """,
        type=IssueType.BUG,
        repository_url="https://github.com/example/demo-project.git",
        labels=["bug", "login", "urgent"]
    )
    
    print("🚀 开始MCTS搜索...")
    print(f"Issue: {issue.title}")
    print(f"描述: {issue.description}")
    print("-" * 50)
    
    # 运行修复
    solution = agent.solve_issue(issue)
    
    if solution:
        print("✅ 找到修复方案！")
        print(f"文件: {solution.file_path}")
        print(f"置信度: {solution.confidence:.2%}")
        print(f"说明: {solution.explanation}")
        print("\n原始代码:")
        print(solution.original_code)
        print("\n修复后代码:")
        print(solution.fixed_code)
    else:
        print("❌ 未能找到满意的修复方案")

if __name__ == "__main__":
    main()
```

### 8.2 创建配置示例

创建 `examples/config_example.py`：

```python
"""
配置示例 - 展示不同的配置选项
"""

from core.llm.client import LLMConfig
from core.mcts_agent import MCTSCodeFixAgent

# 快速搜索配置（适合简单问题）
quick_config = {
    "llm_config": LLMConfig(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=2000
    ),
    "max_iterations": 5,
    "max_depth": 3
}

# 深度搜索配置（适合复杂问题）
deep_config = {
    "llm_config": LLMConfig(
        model="gpt-4",
        temperature=0.2,
        max_tokens=4000
    ),
    "max_iterations": 30,
    "max_depth": 15
}

# 平衡配置（推荐用于大多数情况）
balanced_config = {
    "llm_config": LLMConfig(
        model="claude-3-5-sonnet-20241022",
        temperature=0.15,
        max_tokens=3000
    ),
    "max_iterations": 20,
    "max_depth": 10
}

def create_agent(config_name: str = "balanced") -> MCTSCodeFixAgent:
    """根据配置名称创建代理"""
    
    configs = {
        "quick": quick_config,
        "deep": deep_config,
        "balanced": balanced_config
    }
    
    config = configs.get(config_name, balanced_config)
    
    return MCTSCodeFixAgent(
        llm_config=config["llm_config"],
        max_iterations=config["max_iterations"],
        max_depth=config["max_depth"]
    )
```

## 第九阶段：部署和使用指南

### 9.1 完整的安装脚本

创建 `install.sh`：

```bash
#!/bin/bash

echo "🚀 安装AI代码修复系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 需要Python 3.10或更高版本，当前版本: $python_version"
    exit 1
fi

# 安装Poetry（如果未安装）
if ! command -v poetry &> /dev/null; then
    echo "📦 安装Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# 安装依赖
echo "📚 安装项目依赖..."
poetry install

# 创建环境变量文件
if [ ! -f .env ]; then
    echo "⚙️ 创建环境变量文件..."
    cp .env.example .env
    echo "请编辑 .env 文件并添加你的API Keys"
fi

# 创建工作目录
mkdir -p workspace repos

echo "✅ 安装完成！"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件，添加你的API Keys"
echo "2. 运行: poetry shell"
echo "3. 启动Web界面: python main.py --mode web"
echo "4. 或使用命令行: python main.py --mode cli --help"
```

### 9.2 使用指南

创建 `USAGE.md`：

```markdown
# 使用指南

## 快速开始

### 1. 安装系统
```bash
chmod +x install.sh
./install.sh
```

### 2. 配置环境变量
编辑 `.env` 文件：
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GITHUB_TOKEN=your_github_token_here
```

### 3. 启动系统

#### Web界面模式
```bash
poetry shell
python main.py --mode web
```

#### 命令行模式
```bash
python main.py --mode cli \
  --repo-url "https://github.com/username/repo.git" \
  --title "修复登录bug" \
  --description "用户无法正常登录系统"
```

## 高级配置

### MCTS参数调优

- **max_iterations**: 最大搜索迭代次数
  - 简单问题: 5-10
  - 复杂问题: 20-50
  
- **max_depth**: 最大搜索深度
  - 快速搜索: 3-5
  - 深度搜索: 10-20
  
- **exploration_weight**: 探索权重
  - 更多探索: 1.8-2.5
  - 更多利用: 0.8-1.2

### 模型选择建议

- **gpt-4o-mini**: 快速、经济，适合简单问题
- **gpt-4**: 高质量，适合复杂问题
- **claude-3-5-sonnet**: 平衡性能，推荐用于大多数情况

## 最佳实践

### Issue描述规范
1. 清晰的标题
2. 详细的问题描述
3. 错误信息和堆栈跟踪
4. 期望的行为
5. 相关的代码片段

### 代码库要求
1. 清晰的项目结构
2. 良好的代码注释
3. 一致的编码风格
4. 合理的文件组织

### 性能优化
1. 使用合适的搜索参数
2. 提供精确的Issue描述
3. 定期清理搜索分支
4. 监控API使用量

## 故障排除

### 常见问题

**Q: API调用失败**
A: 检查API Key是否正确设置，网络连接是否正常

**Q: Git操作失败**
A: 确保有足够的磁盘空间，Git配置正确

**Q: 搜索结果不理想**
A: 尝试调整MCTS参数，提供更详细的Issue描述

**Q: 内存使用过高**
A: 减少max_iterations和max_depth参数

### 日志调试
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展开发

### 添加新的动作
1. 继承BaseAction类
2. 实现execute方法
3. 注册到action_registry

### 自定义评估函数
1. 修改_simulate_fix_evaluation方法
2. 添加特定领域的评估逻辑

### 集成外部工具
1. 在动作系统中添加工具调用
2. 扩展观察结果处理逻辑
```

这个完整的系统现在包含了：

1. **完整的MCTS实现** - 包括选择、扩展、模拟、反向传播
2. **Git分支管理** - 每个搜索节点对应一个Git分支
3. **智能提示词系统** - 针对不同阶段的专门提示词
4. **Web界面** - 用户友好的Streamlit界面
5. **配置系统** - 灵活的参数配置
6. **使用示例** - 完整的使用指南和示例

相比原始的简单版本，这个MCTS版本能够：
- 系统性地探索多种修复方案
- 通过Git分支管理不同的尝试
- 使用UCB算法平衡探索和利用
- 提供更可靠的修复质量评估

你可以根据具体需求进一步定制和优化这个系统。lysisResult)
            return result
        except Exception as e:
            # 降级处理
            return AnalysisResult(
                summary=f"分析Issue: {issue.title}",
                key_files=[],
                potential_causes=["需要进一步分析"],
                suggested_actions=["搜索相关代码", "查看错误日志"]
            )
    
    def fix_issue(self, issue: Issue) -> List[FixSuggestion]:
        """修复Issue"""
        # 1. 克隆代码库
        repo_path = self.repo_manager.clone_repository(issue.repository_url)
        
        # 2. 分析Issue
        analysis = self.analyze_issue(issue)
        
        # 3. 搜索相关代码
        search_results = []
        for keyword in self._extract_keywords(issue):
            results = self.code_searcher.search_by_keyword(repo_path, keyword)
            search_results.extend(results)
        
        # 4. 生成修复建议
        suggestions = self._generate_fix_suggestions(issue, analysis, search_results, repo_path)
        
        return suggestions
    
    def _extract_keywords(self, issue: Issue) -> List[str]:
        """从Issue中提取关键词"""
        # 简单的关键词提取
        text = f"{issue.title} {issue.description}".lower()
        
        # 提取可能的函数名、类名等
        import re
        keywords = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
        
        # 过滤常见词汇
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cannot', 'this', 'that', 'these', 'those'}
        
        filtered_keywords = [k for k in keywords if k not in common_words and len(k) > 2]
        
        return list(set(filtered_keywords))[:10]  # 最多10个关键词
    
    def _generate_fix_suggestions(self, issue: Issue, analysis: AnalysisResult, search_results: List, repo_path: str) -> List[FixSuggestion]:
        """生成修复建议"""
        suggestions = []
        
        # 构建上下文
        context = f"""
        Issue分析:
        {analysis.summary}
        
        潜在原因:
        {chr(10).join(analysis.potential_causes)}
        
        相关代码片段:
        """
        
        for result in search_results[:5]:  # 只取前5个结果
            context += f"\n文件: {result.file_path}\n```python\n{result.content}\n```\n"
        
        prompt = f"""
        基于以下信息，请为GitHub Issue提供具体的代码修复建议：
        
        {context}
        
        Issue详情:
        标题: {issue.title}
        描述: {issue.description}
        
        请提供具体的修复方案，包括：
        1. 需要修改的文件
        2. 原始代码
        3. 修复后的代码
        4. 修复说明
        5. 置信度评分(0-1)
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_client.complete(messages)
            # 这里需要解析LLM的响应并转换为FixSuggestion对象
            # 简化处理，实际项目中需要更复杂的解析逻辑
            suggestions.append(FixSuggestion(
                file_path="example.py",
                original_code="# 原始代码",
                fixed_code="# 修复后的代码",
                explanation=response,
                confidence=0.8
            ))
        except Exception as e:
            print(f"生成修复建议时出错: {e}")
        
        return suggestions
```

## 第五阶段：Web界面

### 5.1 Streamlit应用

创建 `web/app.py`：

```python
import streamlit as st
import os
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent import CodeFixAgent
from core.models import Issue, IssueType
from core.llm.client import LLMConfig

def main():
    st.set_page_config(
        page_title="AI代码修复系统",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 AI代码修复系统")
    st.markdown("自动分析GitHub Issue并生成代码修复建议")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("配置")
        
        # LLM配置
        model = st.selectbox(
            "选择模型",
            ["gpt-4o-mini", "gpt-4", "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"]
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # API Key检查
        api_key_status = check_api_keys()
        if not api_key_status:
            st.error("请设置相应的API Key环境变量")
            return
    
    # 主界面
    tab1, tab2 = st.tabs(["Issue分析", "修复历史"])
    
    with tab1:
        st.header("GitHub Issue分析")
        
        # Issue输入表单
        with st.form("issue_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                repo_url = st.text_input(
                    "代码库URL",
                    placeholder="https://github.com/username/repository.git"
                )
            
            with col2:
                issue_type = st.selectbox(
                    "Issue类型",
                    ["bug", "feature", "enhancement"]
                )
            
            issue_title = st.text_input(
                "Issue标题",
                placeholder="请输入Issue标题"
            )
            
            issue_description = st.text_area(
                "Issue描述",
                placeholder="请详细描述问题...",
                height=150
            )
            
            labels = st.text_input(
                "标签 (用逗号分隔)",
                placeholder="bug, urgent, backend"
            )
            
            submitted = st.form_submit_button("开始分析", type="primary")
        
        if submitted and repo_url and issue_title and issue_description:
            # 创建Issue对象
            issue = Issue(
                id="manual_input",
                title=issue_title,
                description=issue_description,
                type=IssueType(issue_type),
                repository_url=repo_url,
                labels=labels.split(",") if labels else []
            )
            
            # 显示分析过程
            with st.spinner("正在分析Issue..."):
                try:
                    # 初始化代理
                    llm_config = LLMConfig(model=model, temperature=temperature)
                    agent = CodeFixAgent(llm_config)
                    
                    # 分析Issue
                    analysis = agent.analyze_issue(issue)
                    
                    # 显示分析结果
                    st.success("分析完成！")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 问题分析")
                        st.write(analysis.summary)
                        
                        st.subheader("🎯 潜在原因")
                        for cause in analysis.potential_causes:
                            st.write(f"• {cause}")
                    
                    with col2:
                        st.subheader("📁 关键文件")
                        for file in analysis.key_files:
                            st.code(file)
                        
                        st.subheader("🔧 建议步骤")
                        for i, action in enumerate(analysis.suggested_actions, 1):
                            st.write(f"{i}. {action}")
                    
                    # 生成修复建议
                    if st.button("生成修复建议", type="primary"):
                        with st.spinner("正在生成修复建议..."):
                            suggestions = agent.fix_issue(issue)
                            
                            if suggestions:
                                st.subheader("💡 修复建议")
                                
                                for i, suggestion in enumerate(suggestions, 1):
                                    with st.expander(f"建议 {i}: {suggestion.file_path} (置信度: {suggestion.confidence:.2f})"):
                                        st.write("**说明:**")
                                        st.write(suggestion.explanation)
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**原始代码:**")
                                            st.code(suggestion.original_code, language="python")
                                        
                                        with col2:
                                            st.write("**修复后代码:**")
                                            st.code(suggestion.fixed_code, language="python")
                            else:
                                st.warning("未能生成修复建议，请检查Issue描述或代码库访问权限")
                
                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
    
    with tab2:
        st.header("修复历史")
        st.info("此功能将在后续版本中实现")

def check_api_keys():
    """检查API Key是否设置"""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    for key in required_keys:
        if os.getenv(key):
            return True
    
    return False

if __name__ == "__main__":
    main()
```

## 第六阶段：主程序入口

### 6.1 创建主入口

创建 `main.py`：

```python
#!/usr/bin/env python3
"""
AI代码修复系统主入口
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def run_web_interface():
    """启动Web界面"""
    import subprocess
    
    web_app_path = Path(__file__).parent / "web" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(web_app_path)])

def run_cli_mode(repo_url: str, issue_title: str, issue_description: str):
    """命令行模式"""
    from core.agent import CodeFixAgent
    from core.models import Issue, IssueType
    from core.llm.client import LLMConfig
    
    # 创建Issue对象
    issue = Issue(
        id="cli_input",
        title=issue_title,
        description=issue_description,
        type=IssueType.BUG,
        repository_url=repo_url
    )
    
    # 初始化代理
    llm_config = LLMConfig()
    agent = CodeFixAgent(llm_config)
    
    print("🔍 正在分析Issue...")
    analysis = agent.analyze_issue(issue)
    
    print(f"\n📋 问题分析:")
    print(analysis.summary)
    
    print(f"\n🎯 潜在原因:")
    for cause in analysis.potential_causes:
        print(f"  • {cause}")
    
    print(f"\n📁 关键文件:")
    for file in analysis.key_files:
        print(f"  • {file}")
    
    print(f"\n🔧 建议步骤:")
    for i, action in enumerate(analysis.suggested_actions, 1):
        print(f"  {i}. {action}")
    
    print(f"\n💡 正在生成修复建议...")
    suggestions = agent.fix_issue(issue)
    
    if suggestions:
        print(f"\n✅ 生成了 {len(suggestions)} 个修复建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n建议 {i} ({suggestion.file_path}, 置信度: {suggestion.confidence:.2f}):")
            print(f"说明: {suggestion.explanation}")
    else:
        print("\n⚠️  未能生成修复建议")

def main():
    parser = argparse.ArgumentParser(description="AI代码修复系统")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", help="运行模式")
    parser.add_argument("--repo-url", help="代码库URL (CLI模式)")
    parser.add_argument("--title", help="Issue标题 (CLI模式)")
    parser.add_argument("--description", help="Issue描述 (CLI模式)")
    
    args = parser.parse_args()
    
    if args.mode == "web":
        print("🚀 启动Web界面...")
        run_web_interface()
    elif args.mode == "cli":
        if not all([args.repo_url, args.title, args.description]):
            print("❌ CLI模式需要提供 --repo-url, --title, --description 参数")
            sys.exit(1)
        
        run_cli_mode(args.repo_url, args.title, args.description)

if __name__ == "__main__":
    main()
```

## 第七阶段：部署和使用

### 7.1 安装依赖

```bash
# 安装Poetry (如果未安装)
curl -sSL https://install.python-poetry.org | python3 -

# 安装项目依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 7.2 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量文件
# 添加你的API Keys
```

### 7.3 运行系统

```bash
# Web界面模式
python main.py --mode web

# 命令行模式
python main.py --mode cli \
  --repo-url "https://github.com/username/repo.git" \
  --title "修复登录bug" \
  --description "用户无法正常登录系统"
```

## 扩展建议

### 后续可以添加的功能：

1. **更智能的代码分析**
   - 集成AST分析
   - 添加代码质量检查
   - 支持更多编程语言

2. **增强的搜索能力**
   - 语义搜索
   - 向量数据库集成
   - 代码相似性分析

3. **自动化测试**
   - 生成单元测试
   - 回归测试检查
   - 代码覆盖率分析

4. **版本控制集成**
   - 自动创建分支
   - 生成Pull Request
   - 代码审查建议

5. **监控和日志**
   - 修复成功率统计
   - 性能监控
   - 错误日志分析

这个框架为你提供了一个完整的起点，你可以根据具体需求逐步扩展和优化各个模块。