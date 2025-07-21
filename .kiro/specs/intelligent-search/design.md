# 智能搜索环节设计文档

## 概述

智能搜索环节是Moatless Tree Search的核心组件，实现了基于蒙特卡洛树搜索（MCTS）的代码修复解决方案探索算法。该设计结合了UCT（Upper Confidence bounds applied to Trees）算法、多智能体协作和价值函数评估，能够系统性地探索代码修复的解决方案空间。

## 架构设计

### 核心组件架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SearchTree                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Node管理      │  │   选择策略      │  │   扩展控制      │  │
│  │   - 树结构      │  │   - UCT算法     │  │   - 动作生成    │  │
│  │   - 状态跟踪    │  │   - 奖励计算    │  │   - 子节点创建  │  │
│  │   - 访问计数    │  │   - 探索平衡    │  │   - 深度控制    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌─────────▼────────┐    ┌────────▼────────┐
│   Selector     │    │  ValueFunction   │    │  ActionAgent    │
│  - BestFirst   │    │  - 奖励评估      │    │  - 动作执行     │
│  - Softmax     │    │  - 代码质量      │    │  - LLM调用      │
│  - LLM选择     │    │  - 测试结果      │    │  - 结果解析     │
└────────────────┘    └──────────────────┘    └─────────────────┘
```

### 数据流设计

```
输入问题 → 创建根节点 → 选择节点 → 生成动作 → 执行动作 → 评估奖励 → 更新树 → 选择下一节点
    ↑                                                                              │
    └──────────────────────── 迭代循环直到满足终止条件 ←─────────────────────────────┘
```

## 组件详细设计

### 1. SearchTree 类设计

**核心属性：**
```python
class SearchTree(BaseModel):
    root: Node                    # 根节点
    selector: Selector           # 节点选择器
    agent: ActionAgent          # 动作执行智能体
    value_function: ValueFunction # 价值函数
    discriminator: Discriminator  # 判别器
    
    # 搜索控制参数
    max_expansions: int = 1      # 每个状态最大扩展次数
    max_iterations: int = 10     # 最大迭代次数
    max_depth: int = None        # 最大搜索深度
    reward_threshold: float = None # 奖励阈值
```

**核心方法：**
- `run_search()`: 主搜索循环
- `select_node()`: 选择待扩展节点
- `expand_node()`: 扩展节点
- `update_tree()`: 更新树状态

### 2. 节点选择策略（Selector）

#### UCT算法实现

**UCT分数计算公式：**
```
UCT(node) = exploitation + exploration + bonuses - penalties

其中：
- exploitation = reward_weight * node_reward
- exploration = exploration_weight * sqrt(ln(parent_visits) / node_visits)
- bonuses = depth_bonus + high_value_bonus + diversity_bonus
- penalties = duplicate_penalty + finished_trajectory_penalty
```

**关键参数：**
```python
class Selector(BaseModel):
    exploitation_weight: float = 1.0      # 利用权重
    exploration_weight: float = 1.0       # 探索权重
    depth_weight: float = 0.8            # 深度权重
    high_value_threshold: float = 50.0    # 高价值阈值
    diversity_weight: float = 100.0       # 多样性权重
```

#### 选择策略类型

1. **BestFirstSelector**: 选择UCT分数最高的节点
2. **SoftmaxSelector**: 基于概率分布选择节点
3. **LLMSelector**: 使用LLM进行智能选择
4. **FeedbackSelector**: 基于反馈的选择策略

### 3. 价值函数设计

#### 评估维度

**代码质量评估：**
- 语法正确性（0-25分）
- 逻辑合理性（0-25分）
- 代码风格（0-15分）
- 性能影响（0-10分）

**功能完整性评估：**
- 需求满足度（0-30分）
- 边界情况处理（0-20分）
- 错误处理（0-15分）

**测试结果评估：**
- 测试通过率（0-40分）
- 新增测试覆盖（0-20分）
- 回归测试（0-15分）

#### 奖励计算实现

```python
class ValueFunction:
    def calculate_reward(self, node: Node) -> Reward:
        # 基础代码质量评分
        code_quality_score = self._evaluate_code_quality(node)
        
        # 测试结果评分
        test_score = self._evaluate_test_results(node)
        
        # 功能完整性评分
        functionality_score = self._evaluate_functionality(node)
        
        # 综合评分
        total_score = (
            code_quality_score * 0.4 +
            test_score * 0.4 +
            functionality_score * 0.2
        )
        
        return Reward(value=total_score, explanation=explanation)
```

### 4. 动作执行系统

#### 动作类型设计

**查找动作：**
- `FindClass`: 查找类定义
- `FindFunction`: 查找函数定义
- `FindCodeSnippet`: 查找代码片段
- `SemanticSearch`: 语义搜索

**查看动作：**
- `ViewCode`: 查看代码内容

**修改动作：**
- `StringReplace`: 字符串替换
- `CreateFile`: 创建新文件
- `AppendString`: 追加内容

**测试动作：**
- `RunTests`: 运行测试

**控制动作：**
- `Finish`: 完成任务
- `Reject`: 拒绝当前方案

#### 动作执行流程

```python
def execute_action(self, node: Node, action: Action) -> Node:
    # 1. 验证动作有效性
    if not self._validate_action(action):
        return self._create_error_node(node, "Invalid action")
    
    # 2. 执行动作
    try:
        observation = action.execute(node.file_context)
    except Exception as e:
        return self._create_error_node(node, str(e))
    
    # 3. 创建子节点
    child_node = Node(
        parent=node,
        action=action,
        observation=observation
    )
    
    # 4. 更新文件上下文
    child_node.file_context = self._update_file_context(
        node.file_context, action, observation
    )
    
    return child_node
```

### 5. 多智能体协作设计

#### 判别器（Discriminator）

**功能：** 从多个候选解决方案中选择最佳方案

**实现：**
```python
class AgentDiscriminator:
    def __init__(self, n_agents: int = 5, n_rounds: int = 3):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
    
    def select_best_trajectory(self, trajectories: List[Node]) -> Node:
        # 多轮投票选择最佳轨迹
        scores = {}
        for round in range(self.n_rounds):
            for agent_id in range(self.n_agents):
                votes = self._agent_vote(trajectories, agent_id)
                for traj, score in votes.items():
                    scores[traj] = scores.get(traj, 0) + score
        
        return max(scores.keys(), key=lambda x: scores[x])
```

#### 反馈生成器（FeedbackGenerator）

**功能：** 为后续搜索提供指导性反馈

**反馈类型：**
- 路径建议：推荐探索方向
- 错误纠正：指出常见错误
- 优化建议：提供改进方案

### 6. 搜索控制机制

#### 终止条件

1. **迭代次数限制：** `max_iterations`
2. **成本限制：** `max_cost`（基于token使用量）
3. **奖励阈值：** `reward_threshold`
4. **完成节点数量：** `min_finished_nodes`, `max_finished_nodes`
5. **深度限制：** `max_depth`

#### 搜索优化策略

**剪枝策略：**
- 低奖励节点剪枝
- 重复路径剪枝
- 深度限制剪枝

**缓存机制：**
- 相似度计算缓存
- 动作结果缓存
- 价值函数评估缓存

## 数据模型

### Node 数据结构

```python
class Node:
    node_id: int                    # 节点唯一标识
    parent: Optional[Node]          # 父节点
    children: List[Node]            # 子节点列表
    action: Optional[Action]        # 执行的动作
    observation: Optional[Observation] # 动作执行结果
    reward: Optional[Reward]        # 节点奖励
    visits: int                     # 访问次数
    file_context: FileContext       # 文件上下文
    metadata: Dict[str, Any]        # 元数据
```

### UCTScore 数据结构

```python
@dataclass
class UCTScore:
    final_score: float = 0.0                    # 最终分数
    exploitation: float = 0.0                   # 利用项
    exploration: float = 0.0                    # 探索项
    depth_bonus: float = 0.0                    # 深度奖励
    high_value_leaf_bonus: float = 0.0          # 高价值叶子奖励
    diversity_bonus: float = 0.0                # 多样性奖励
    duplicate_child_penalty: float = 0.0        # 重复子节点惩罚
    finished_trajectory_penalty: float = 0.0    # 完成轨迹惩罚
```

## 错误处理

### 异常类型

1. **RuntimeError**: 运行时错误
2. **RejectError**: 动作被拒绝
3. **ValidationError**: 输入验证错误
4. **TimeoutError**: 超时错误

### 错误恢复策略

1. **重试机制**: 对临时性错误进行重试
2. **降级策略**: 使用简化的备选方案
3. **错误传播**: 将错误信息传递给上层处理
4. **状态回滚**: 恢复到上一个稳定状态

## 测试策略

### 单元测试

- 各组件独立功能测试
- 边界条件测试
- 异常情况测试

### 集成测试

- 组件间协作测试
- 端到端流程测试
- 性能基准测试

### 评估指标

- 搜索效率：平均迭代次数
- 解决方案质量：奖励分数分布
- 资源消耗：时间和内存使用
- 成功率：问题解决比例