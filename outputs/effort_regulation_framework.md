# Executive Brain: Effort Regulation Framework (Model-Agnostic)

## Document Overview

**Purpose**: Theoretical framework for sophisticated effort regulation in LLM-based systems
**Version**: 2.1 (model-agnostic, no cost/time analysis)
**Date**: 2025-11-08
**Key Principle**: Dynamically allocate computational effort based on task complexity and context

---

## 1. Core Concept: Effort as a Continuous Variable

### 1.1 Definition

**Effort** = The amount of computational resources (time, thinking steps, iterations) allocated to solving a task.

**Effort Spectrum**:
```
0.0 ────────────────────────────────────────► 1.0
│                   │                      │
Minimal            Balanced             Maximum
(Fast, lower       (Standard            (Slow, highest
 accuracy)          reasoning)            accuracy)
```

**Key Insight**: Not all tasks require maximum effort. A simple factual query ("What's the Phoenix project status?") should use minimal effort, while a complex decision ("Analyze Q3 spending and recommend 10% budget cuts") requires maximum effort.

### 1.2 Effort Allocation Goals

1. **Optimize Quality/Speed Trade-off**: Allocate just enough effort to meet quality thresholds
2. **Resource Efficiency**: Don't waste computation on simple tasks
3. **Adaptive**: Increase effort if initial attempt fails quality gates
4. **Learning**: Improve allocation over time based on outcomes

---

## 2. Task Complexity Analysis: Six Dimensions

### 2.1 Framework

**Complexity Score** = Weighted combination of six independent dimensions:

```python
def compute_intrinsic_complexity(task: str) -> float:
    """
    Return complexity score 0.0 (trivial) to 1.0 (extremely complex).
    """
    dimensions = {
        "reasoning_depth": analyze_reasoning_depth(task),      # Weight: 0.25
        "knowledge_breadth": analyze_knowledge_breadth(task),  # Weight: 0.15
        "tool_orchestration": analyze_tool_complexity(task),   # Weight: 0.20
        "ambiguity": analyze_ambiguity(task),                  # Weight: 0.15
        "constraints": analyze_constraints(task),              # Weight: 0.15
        "novelty": analyze_novelty(task)                       # Weight: 0.10
    }

    weights = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]
    complexity = sum(
        dimensions[dim] * weight
        for dim, weight in zip(dimensions, weights)
    )

    return complexity
```

### 2.2 Dimension 1: Reasoning Depth

**Question**: How many reasoning steps are required?

**Scale**:
- **0.0-0.2**: Single-step lookup (e.g., "What is the capital of France?")
- **0.2-0.4**: 2-3 step inference (e.g., "Who reports to the CTO?")
- **0.4-0.6**: Multi-step reasoning (e.g., "Why did revenue drop in Q3?")
- **0.6-0.8**: Deep analysis requiring synthesis (e.g., "Compare Q3 performance across departments and identify trends")
- **0.8-1.0**: Complex proofs or novel problem solving (e.g., "Design an algorithm to optimize...")

**Detection Heuristics**:
```python
def analyze_reasoning_depth(task: str) -> float:
    # Pattern matching
    single_step_patterns = [
        r"what is", r"who is", r"when did", r"define", r"status of"
    ]
    multi_step_patterns = [
        r"analyze.*and.*recommend", r"compare.*and.*decide",
        r"why.*because", r"prove", r"optimize"
    ]
    deep_patterns = [
        r"synthesize.*from.*multiple", r"derive.*from.*first principles",
        r"design.*system", r"simulate.*outcomes"
    ]

    if any(re.search(p, task.lower()) for p in deep_patterns):
        return 0.9
    elif any(re.search(p, task.lower()) for p in multi_step_patterns):
        return 0.6
    elif any(re.search(p, task.lower()) for p in single_step_patterns):
        return 0.2
    else:
        # Use lightweight LLM to classify
        return llm_classify_depth(task)
```

### 2.3 Dimension 2: Knowledge Breadth

**Question**: How many knowledge domains are involved?

**Scale**:
- **0.0-0.3**: Single narrow domain (e.g., "What's Project X budget?")
- **0.3-0.5**: Two related domains (e.g., "How does Project X budget impact Q3 forecast?")
- **0.5-0.7**: Multiple domains (e.g., "Analyze Project X's financial, technical, and HR impacts")
- **0.7-1.0**: Cross-domain synthesis (e.g., "Recommend strategy considering financial, market, technical, and regulatory factors")

**Detection**:
```python
def analyze_knowledge_breadth(task: str) -> float:
    # Extract entities and concepts
    entities = extract_entities(task)  # NER

    # Query knowledge graph for domains
    domains = set()
    for entity in entities:
        entity_domains = knowledge_graph.get_domains(entity)
        domains.update(entity_domains)

    # Map domain count to score
    domain_count = len(domains)
    if domain_count == 1:
        return 0.2
    elif domain_count == 2:
        return 0.4
    elif domain_count >= 3:
        return 0.7
    else:
        return 0.5  # Default
```

### 2.4 Dimension 3: Tool Orchestration Complexity

**Question**: How many tools need to be coordinated?

**Scale**:
- **0.0**: No tools needed (pure reasoning)
- **0.0-0.3**: 1-2 simple tools (e.g., database query + email)
- **0.3-0.6**: 3-5 tools with minimal dependencies
- **0.6-0.8**: 6-10 tools with sequential dependencies
- **0.8-1.0**: 10+ tools with complex orchestration (parallel + sequential)

**Detection**:
```python
def analyze_tool_complexity(task: str) -> float:
    # Predict which tools will be needed
    likely_tools = predict_required_tools(task)  # Use LLM or rules

    tool_count = len(likely_tools)
    dependencies = analyze_tool_dependencies(likely_tools)

    # Base complexity from count
    base = min(tool_count / 10, 1.0)

    # Add dependency complexity
    dependency_factor = len(dependencies) / 20

    return min(base + dependency_factor, 1.0)

def predict_required_tools(task: str) -> List[str]:
    """
    Use lightweight LLM to predict needed tools.
    """
    prompt = f"""
    Task: {task}
    Available tools: database, email, calendar, web_search, file_ops, crm, erp

    Which tools will likely be needed? Return JSON list.
    """
    result = fast_llm.generate(prompt, max_tokens=100)
    return json.loads(result)
```

### 2.5 Dimension 4: Ambiguity

**Question**: How clear is the task specification?

**Scale**:
- **0.0**: Crystal clear, unambiguous (e.g., "Retrieve Q3 budget for Marketing department")
- **0.3**: Minor ambiguity (e.g., "Get budget for Marketing" - which period?)
- **0.6**: Moderate ambiguity (e.g., "How are we doing on budget?" - which department? which metric?)
- **1.0**: Highly ambiguous (e.g., "Make things better" - what things? how?)

**Detection**:
```python
def analyze_ambiguity(task: str) -> float:
    ambiguity_markers = [
        r"\bor\b",           # "Should we do X or Y?"
        r"maybe", r"possibly", r"unclear", r"not sure",
        r"depends on", r"various", r"several options",
        r"things", r"stuff", r"it"  # Vague pronouns
    ]

    marker_count = sum(
        1 for marker in ambiguity_markers
        if re.search(marker, task.lower())
    )

    # Short vague questions are highly ambiguous
    if "?" in task and len(task.split()) < 10:
        marker_count += 1

    # Missing key information (heuristic: no entities mentioned)
    entities = extract_entities(task)
    if len(entities) == 0:
        marker_count += 1

    return min(marker_count / 5, 1.0)
```

### 2.6 Dimension 5: Constraints

**Question**: How many constraints must be satisfied?

**Scale**:
- **0.0**: No constraints (open-ended)
- **0.3**: 1-2 soft constraints (e.g., "preferably by Friday")
- **0.6**: 3-4 hard constraints (e.g., "must comply with GDPR, within $10K budget")
- **1.0**: Many complex constraints (e.g., "optimize cost while maintaining quality, complying with regulations, meeting deadline, and satisfying stakeholders")

**Detection**:
```python
def analyze_constraints(task: str) -> float:
    constraint_markers = [
        r"without", r"must not", r"cannot", r"while maintaining",
        r"subject to", r"within.*budget", r"by.*deadline",
        r"compliant with", r"constrained by"
    ]

    constraint_count = sum(
        1 for marker in constraint_markers
        if re.search(marker, task.lower())
    )

    return min(constraint_count / 5, 1.0)
```

### 2.7 Dimension 6: Novelty

**Question**: Has a similar task been done before?

**Scale**:
- **0.0-0.2**: Identical or nearly identical task seen recently (cache hit)
- **0.2-0.5**: Similar task pattern exists (can adapt prior solution)
- **0.5-0.8**: Somewhat novel (some familiar elements, some new)
- **0.8-1.0**: Completely novel (no prior examples)

**Detection**:
```python
def analyze_novelty(task: str) -> float:
    # Vector search for similar past tasks
    similar_tasks = knowledge_graph.vector_search(
        embedding=embed(task),
        node_label="Decision",
        top_k=5
    )

    if not similar_tasks:
        return 0.8  # No prior examples

    best_similarity = similar_tasks[0]["similarity"]

    if best_similarity > 0.95:
        return 0.1  # Nearly identical
    elif best_similarity > 0.8:
        return 0.3  # Very similar
    elif best_similarity > 0.6:
        return 0.5  # Somewhat similar
    else:
        return 0.8  # Novel
```

### 2.8 Example: Complete Complexity Analysis

```python
task = """
Analyze Q3 spending across all departments, identify cost reduction opportunities
that won't impact revenue, simulate 10% budget cut scenarios, and recommend which
departments should absorb cuts while maintaining compliance with labor laws.
"""

complexity_breakdown = {
    "reasoning_depth": 0.9,        # Deep analysis + simulation
    "knowledge_breadth": 0.8,      # Finance + HR + Legal + Operations
    "tool_orchestration": 0.7,     # Database, financial model, simulation, policy check
    "ambiguity": 0.3,              # Relatively clear objective
    "constraints": 0.8,            # Revenue constraint, legal compliance, 10% target
    "novelty": 0.6                 # Similar budget analysis done before, but novel constraints
}

weights = [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]

intrinsic_complexity = sum(
    score * weight
    for score, weight in zip(complexity_breakdown.values(), weights)
)
# Result: 0.73 (HIGH COMPLEXITY)
```

---

## 3. Context-Aware Effort Modulation

### 3.1 Contextual Factors

**Intrinsic complexity** (from task analysis) is only half the story. Context modulates effort:

| Factor | Impact on Effort | Rationale |
|--------|------------------|-----------|
| **Urgency** | High urgency → **Lower** effort | Trade accuracy for speed |
| **Risk** | High risk → **Higher** effort | Accuracy critical for important decisions |
| **Resource Budget** | Low remaining budget → **Lower** effort | Conserve resources |
| **User Preference** | User prefers speed → **Lower** effort | Honor user preference |
| **Time of Day** | Off-hours → **Lower** effort (optional) | Non-urgent tasks processed async |

### 3.2 Context Scoring

```python
def analyze_context(task: str, user: User, system_state: SystemState) -> Dict[str, float]:
    """
    Analyze contextual factors that modulate effort.
    Return scores 0.0-1.0 for each factor.
    """
    return {
        "urgency": assess_urgency(task, user),
        "risk": calculate_risk_score(task, user),  # From governance layer
        "budget_remaining": get_budget_utilization(system_state),
        "user_preference": user.preferences.get("effort_preference", 0.5),
        "time_sensitivity": assess_time_of_day(task)
    }

def assess_urgency(task: str, user: User) -> float:
    """Score 0.0 (can wait) to 1.0 (urgent)"""
    urgency_keywords = {
        "critical": ["urgent", "asap", "immediately", "emergency", "critical"],
        "high": ["today", "this morning", "by EOD"],
        "medium": ["this week", "soon"],
        "low": ["when you can", "no rush", "batch", "eventually"]
    }

    task_lower = task.lower()
    if any(kw in task_lower for kw in urgency_keywords["critical"]):
        return 1.0
    elif any(kw in task_lower for kw in urgency_keywords["high"]):
        return 0.7
    elif any(kw in task_lower for kw in urgency_keywords["medium"]):
        return 0.5
    else:
        return 0.2
```

### 3.3 Final Effort Calculation

```python
def compute_final_effort(
    intrinsic_complexity: float,
    context: Dict[str, float]
) -> float:
    """
    Combine intrinsic complexity with context to get final effort allocation.
    """
    # Base effort from task complexity
    base_effort = intrinsic_complexity

    # Contextual modulation
    urgency_multiplier = 1.0 - (context["urgency"] * 0.3)  # Max 30% reduction
    risk_multiplier = 1.0 + (context["risk"] * 0.5)  # Max 50% increase
    budget_multiplier = context["budget_remaining"]  # Direct scaling
    preference_multiplier = 0.7 + (context["user_preference"] * 0.6)  # Range: 0.7-1.3

    # Composite multiplier
    total_multiplier = (
        urgency_multiplier *
        risk_multiplier *
        budget_multiplier *
        preference_multiplier
    )

    # Final effort (clamped to [0.0, 1.0])
    final_effort = min(base_effort * total_multiplier, 1.0)

    return final_effort
```

**Example**:
```python
intrinsic_complexity = 0.73  # From budget analysis example

context = {
    "urgency": 0.3,              # No immediate deadline
    "risk": 0.6,                 # Medium-high risk (financial decision)
    "budget_remaining": 0.8,     # 80% of daily budget left
    "user_preference": 0.7,      # User prefers accuracy
    "time_sensitivity": 0.5      # Business hours (neutral)
}

# Multipliers:
# urgency:     1.0 - (0.3 * 0.3) = 0.91
# risk:        1.0 + (0.6 * 0.5) = 1.3
# budget:      0.8
# preference:  0.7 + (0.7 * 0.6) = 1.12
# total:       0.91 * 1.3 * 0.8 * 1.12 = 1.06

final_effort = min(0.73 * 1.06, 1.0) = 0.77

# Interpretation: HIGH EFFORT (use thorough strategy)
```

---

## 4. Effort Strategies: Discrete Levels

### 4.1 Strategy Mapping

Map continuous effort score (0.0-1.0) to discrete strategies:

| Strategy | Effort Range | Description | Typical Parameters |
|----------|--------------|-------------|-------------------|
| **Minimal** | 0.0 - 0.2 | Fastest, lowest accuracy | Few reasoning steps, high temperature |
| **Fast** | 0.2 - 0.4 | Quick response, acceptable quality | Standard steps, moderate temperature |
| **Balanced** | 0.4 - 0.6 | Default, good quality/speed trade-off | Extended steps, balanced temperature |
| **Thorough** | 0.6 - 0.8 | Deep reasoning, high accuracy | Many steps, exploration encouraged |
| **Maximum** | 0.8 - 1.0 | Highest accuracy, slowest | Maximum steps, exhaustive search |

### 4.2 Model-Agnostic Parameter Mapping

**Challenge**: Different models have different "effort" mechanisms:
- **Thinking Budget Models** (Kimi K2, DeepSeek R1): Variable thinking token budgets
- **Standard Models** (Llama, Qwen, Mistral): Temperature, max_tokens, top_p
- **Proprietary with Modes** (Claude): Thinking mode on/off

**Solution**: Abstract "effort" to model-specific parameters via provider interface.

```python
def map_effort_to_params(
    effort_score: float,
    model_provider: LLMProvider
) -> Dict[str, Any]:
    """
    Convert effort score to model-specific parameters.
    """
    capabilities = model_provider.get_capabilities()
    params = {}

    # Strategy selection
    strategy = select_strategy(effort_score)

    # Thinking budget models
    if capabilities.get("supports_thinking"):
        min_budget, max_budget, default = model_provider.get_thinking_budget()

        if max_budget > 0:
            # Map effort to thinking budget
            thinking_budget = int(
                min_budget + (max_budget - min_budget) * effort_score
            )

            # Model-specific parameter names (provider handles this)
            params.update(model_provider.effort_to_params(effort_score, strategy))

    # Temperature (universal parameter)
    # Higher effort = higher exploration (for complex tasks)
    params["temperature"] = 0.3 + (effort_score * 0.6)  # Range: 0.3-0.9

    # Max tokens (universal parameter)
    if effort_score < 0.3:
        params["max_tokens"] = 2048
    elif effort_score < 0.6:
        params["max_tokens"] = 4096
    else:
        params["max_tokens"] = 8192

    # Top-p (nucleus sampling)
    params["top_p"] = 0.9 + (effort_score * 0.09)  # Range: 0.9-0.99

    return params

def select_strategy(effort_score: float) -> str:
    """Map continuous effort to discrete strategy."""
    if effort_score < 0.2:
        return "minimal"
    elif effort_score < 0.4:
        return "fast"
    elif effort_score < 0.6:
        return "balanced"
    elif effort_score < 0.8:
        return "thorough"
    else:
        return "maximum"
```

---

## 5. Adaptive Retry with Quality Gates

### 5.1 Quality-Based Escalation

**Principle**: If initial effort produces low-quality output, automatically retry with more effort.

```python
def execute_with_adaptive_effort(task, initial_effort_config):
    """
    Execute task, evaluate quality, retry with more effort if needed.
    """
    max_retries = 3
    current_effort = initial_effort_config["effort_score"]
    effort_ladder = [0.2, 0.4, 0.6, 0.8, 1.0]  # Escalation path

    for attempt in range(max_retries):
        # Execute with current effort
        result = execute_task(task, current_effort)

        # Evaluate quality (Ragas)
        quality = evaluate_quality(result, task)

        # Check quality gates
        if quality["faithfulness"] >= 0.8 and quality["relevance"] >= 0.7:
            # Success
            log_effort_success(task, current_effort, quality, attempt)
            return result
        else:
            # Quality insufficient
            logger.warning(
                f"Attempt {attempt+1} failed quality gates. "
                f"Faithfulness: {quality['faithfulness']:.2f}, "
                f"Relevance: {quality['relevance']:.2f}"
            )

            # Escalate effort
            current_idx = min(
                range(len(effort_ladder)),
                key=lambda i: abs(effort_ladder[i] - current_effort)
            )
            if current_idx < len(effort_ladder) - 1:
                current_effort = effort_ladder[current_idx + 1]
                logger.info(f"Escalating to effort {current_effort}")
            else:
                # Already at maximum, can't escalate further
                logger.error("At maximum effort, still failing quality gates")
                flag_for_human_review(task, result, quality)
                return result  # Return best attempt

    # All retries exhausted
    flag_for_human_review(task, result, quality)
    return result
```

### 5.2 Quality Thresholds

**Configurable per deployment**:

```python
QUALITY_THRESHOLDS = {
    "faithfulness": {
        "reject": 0.5,   # Below this = hallucination, reject immediately
        "retry": 0.8,    # Below this = retry with more effort
        "accept": 0.8    # Above this = good enough
    },
    "answer_relevance": {
        "reject": 0.4,
        "retry": 0.7,
        "accept": 0.7
    },
    "context_precision": {
        "retry": 0.6,
        "accept": 0.6
    }
}

def evaluate_quality(result, task):
    """Use Ragas to evaluate response quality."""
    from ragas.metrics import faithfulness, answer_relevance, context_precision

    evaluation = {
        "faithfulness": faithfulness.score(result, task),
        "answer_relevance": answer_relevance.score(result, task),
        "context_precision": context_precision.score(result, task)
    }

    return evaluation
```

---

## 6. Learning: Optimizing Effort Over Time

### 6.1 Historical Outcome Tracking

```python
def record_effort_outcome(
    task_type: str,
    effort_allocated: float,
    quality_achieved: Dict[str, float],
    latency_seconds: float
):
    """
    Store outcome in knowledge graph for later analysis.
    """
    knowledge_graph.create_node(
        "EffortOutcome",
        {
            "task_type": task_type,
            "effort_allocated": effort_allocated,
            "faithfulness": quality_achieved["faithfulness"],
            "relevance": quality_achieved["answer_relevance"],
            "latency": latency_seconds,
            "timestamp": datetime.now()
        }
    )
```

### 6.2 Pareto Optimization

**Goal**: For each task type, find the **minimum effort** that achieves quality thresholds.

```python
def learn_optimal_effort(task_type: str) -> float:
    """
    Analyze last 30 days of outcomes for this task type.
    Find Pareto frontier: quality vs effort.
    """
    # Query historical outcomes
    outcomes = knowledge_graph.query(f"""
        MATCH (o:EffortOutcome {{task_type: '{task_type}'}})
        WHERE o.timestamp > datetime() - duration('P30D')
        RETURN o.effort_allocated as effort,
               o.faithfulness as faithfulness,
               o.relevance as relevance,
               o.latency as latency
        ORDER BY o.timestamp DESC
        LIMIT 100
    """)

    if len(outcomes) < 10:
        return 0.5  # Not enough data, use default

    # Find Pareto frontier
    pareto_points = []
    for outcome in outcomes:
        # Is this outcome dominated by any other?
        dominated = False
        for other in outcomes:
            if (other["faithfulness"] >= outcome["faithfulness"] and
                other["relevance"] >= outcome["relevance"] and
                other["effort"] <= outcome["effort"]):
                # Other is better or equal in quality, and uses less effort
                if (other["faithfulness"] > outcome["faithfulness"] or
                    other["relevance"] > outcome["relevance"] or
                    other["effort"] < outcome["effort"]):
                    dominated = True
                    break

        if not dominated:
            pareto_points.append(outcome)

    # From Pareto frontier, select minimum effort that meets thresholds
    acceptable = [
        p for p in pareto_points
        if p["faithfulness"] >= 0.8 and p["relevance"] >= 0.7
    ]

    if acceptable:
        optimal = min(acceptable, key=lambda p: p["effort"])
        return optimal["effort"]
    else:
        # No point meets thresholds, need more effort
        return 0.8  # Default to thorough
```

### 6.3 Continuous Improvement

```python
class EffortLearningModule:
    """
    Continuously learn and update optimal effort levels.
    """

    def __init__(self, update_frequency_hours=24):
        self.update_frequency = update_frequency_hours
        self.task_type_efforts = {}  # Cache

    def run_learning_loop(self):
        """Background thread that updates effort recommendations."""
        while True:
            # Get all task types seen recently
            task_types = knowledge_graph.query("""
                MATCH (o:EffortOutcome)
                WHERE o.timestamp > datetime() - duration('P7D')
                RETURN DISTINCT o.task_type as task_type
            """)

            # Learn optimal effort for each
            for task_type in task_types:
                optimal_effort = learn_optimal_effort(task_type["task_type"])
                self.task_type_efforts[task_type["task_type"]] = optimal_effort

                logger.info(
                    f"Learned optimal effort for '{task_type['task_type']}': "
                    f"{optimal_effort:.2f}"
                )

            # Sleep until next update
            time.sleep(self.update_frequency * 3600)

    def get_recommended_effort(self, task_type: str) -> float:
        """Get learned effort for task type (or default if unknown)."""
        return self.task_type_efforts.get(task_type, 0.5)
```

---

## 7. Integration with ROMA: Per-Subtask Effort Allocation

### 7.1 Recursive Effort Budget

**Challenge**: In ROMA's recursive decomposition, how to allocate effort to subtasks?

**Principle**: Parent task has effort budget → distribute to subtasks based on their importance.

```python
class ROMAWithEffortRegulation:
    def decompose_with_effort(self, task, parent_effort_budget):
        """
        ROMA planning phase enhanced with effort allocation.
        """
        # Decompose into subtasks (standard ROMA)
        subtasks = self.planner.decompose(task)

        # Analyze each subtask
        for subtask in subtasks:
            # Intrinsic complexity of subtask
            subtask_complexity = compute_intrinsic_complexity(subtask["description"])

            # Priority in task graph (critical path = higher effort)
            priority = self.calculate_priority(subtask, subtasks)

            # Allocate effort
            subtask_effort = self.allocate_subtask_effort(
                parent_effort_budget=parent_effort_budget,
                subtask_complexity=subtask_complexity,
                priority=priority
            )

            subtask["effort_allocation"] = subtask_effort

        return subtasks

    def allocate_subtask_effort(
        self,
        parent_effort_budget: float,
        subtask_complexity: float,
        priority: float
    ) -> float:
        """
        Allocate portion of parent's effort budget to subtask.

        Args:
            parent_effort_budget: Total effort parent has (0.0-1.0)
            subtask_complexity: Intrinsic complexity of subtask (0.0-1.0)
            priority: How critical is this subtask (0.0-1.0)

        Returns:
            Effort allocation for subtask (0.0-1.0)
        """
        # Base: inherit from parent (but reserve some for aggregation)
        base_allocation = parent_effort_budget * 0.85  # 15% reserved

        # Adjust by priority
        priority_multiplier = 0.5 + (priority * 1.0)  # Range: 0.5-1.5

        # Adjust by complexity
        complexity_multiplier = 0.7 + (subtask_complexity * 0.6)  # Range: 0.7-1.3

        # Final allocation
        subtask_effort = base_allocation * priority_multiplier * complexity_multiplier

        return min(subtask_effort, 1.0)  # Clamp to max

    def calculate_priority(self, subtask, all_subtasks) -> float:
        """
        Determine how critical this subtask is (0.0-1.0).
        Critical path tasks = higher priority.
        """
        # Is this subtask on critical path?
        dependency_graph = self.build_dependency_graph(all_subtasks)
        critical_path = self.find_critical_path(dependency_graph)

        if subtask in critical_path:
            return 1.0  # Maximum priority
        else:
            # Non-critical: base priority on dependency depth
            depth = self.get_depth_in_graph(subtask, dependency_graph)
            return 0.5 + (depth / max_depth) * 0.4  # Range: 0.5-0.9
```

### 7.2 Example: Recursive Budget Allocation

```
Task: "Analyze Q3 spending and recommend 10% budget cuts"
Parent Effort: 0.77 (thorough strategy)

Decomposition:
├─ Subtask 1: "Retrieve Q3 spending by department"
│  ├─ Complexity: 0.2 (simple query)
│  ├─ Priority: 0.9 (critical path - data needed first)
│  ├─ Allocated: 0.77 * 0.85 * 1.4 * 0.82 = 0.75 (thorough)
│  └─ Rationale: Data quality critical, even if query simple
│
├─ Subtask 2: "Analyze spending vs budget vs revenue"
│  ├─ Complexity: 0.6 (multi-factor analysis)
│  ├─ Priority: 1.0 (critical path - core reasoning)
│  ├─ Allocated: 0.77 * 0.85 * 1.5 * 1.06 = 1.04 → 1.0 (maximum)
│  └─ Rationale: Core insight generation, needs deep thinking
│
├─ Subtask 3: "Identify low-impact cost reductions"
│  ├─ Complexity: 0.7 (creative reasoning)
│  ├─ Priority: 1.0 (critical path)
│  ├─ Allocated: 0.77 * 0.85 * 1.5 * 1.12 = 1.10 → 1.0 (maximum)
│  └─ Rationale: Novel problem, high stakes
│
├─ Subtask 4: "Simulate 10% cut scenarios"
│  ├─ Complexity: 0.5 (tool orchestration)
│  ├─ Priority: 0.8 (validation, not on critical path)
│  ├─ Allocated: 0.77 * 0.85 * 1.3 * 1.0 = 0.85 (maximum)
│  └─ Rationale: Accuracy important for financial modeling
│
└─ Subtask 5: "Synthesize recommendations"
   ├─ Complexity: 0.8 (synthesis + risk assessment)
   ├─ Priority: 1.0 (final output, critical)
   ├─ Allocated: 0.77 * 0.85 * 1.5 * 1.18 = 1.15 → 1.0 (maximum)
   └─ Rationale: Executive decision quality paramount
```

---

## 8. Monitoring & Observability

### 8.1 Key Metrics

**Effort Regulation Effectiveness**:
```python
# Prometheus metrics
effort_distribution = Histogram(
    "effort_allocation_score",
    "Distribution of effort scores",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

effort_strategy_count = Counter(
    "effort_strategy_total",
    "Count by strategy",
    ["strategy"]  # minimal, fast, balanced, thorough, maximum
)

quality_vs_effort = Histogram(
    "quality_score_by_effort",
    "Quality achieved vs effort allocated",
    labelnames=["metric", "effort_strategy"]
)

retry_escalation_rate = Counter(
    "effort_retry_escalations",
    "How often we escalated effort on retry",
    ["from_strategy", "to_strategy"]
)

learning_module_updates = Counter(
    "effort_learning_updates",
    "Optimal effort updates per task type",
    ["task_type"]
)
```

### 8.2 Dashboards

**Effort Allocation Dashboard**:
- Distribution of effort scores (histogram)
- Strategy usage over time (stacked area chart)
- Effort vs quality scatter plot (identify optimal points)
- Retry escalation funnel (how often minimal → fast → balanced → thorough?)

**Learning Dashboard**:
- Optimal effort by task type (table)
- Pareto frontiers per task type (scatter plots)
- Trend: Is optimal effort increasing or decreasing? (line chart)

---

## 9. Summary

This framework provides:

✅ **6-Dimensional Complexity Scoring**: Reasoning depth, knowledge breadth, tool orchestration, ambiguity, constraints, novelty
✅ **Context-Aware Modulation**: Urgency, risk, budget, user preferences
✅ **Adaptive Retry**: Escalate effort if quality insufficient
✅ **Learning Optimization**: Pareto frontier analysis to find minimum viable effort
✅ **ROMA Integration**: Per-subtask effort allocation in recursive decomposition
✅ **Model-Agnostic**: Maps effort score to provider-specific parameters
✅ **Monitoring**: Comprehensive metrics for continuous improvement

**Key Takeaway**: This is a **theoretical framework**. Specific parameter values (weights, thresholds, etc.) should be tuned based on real-world outcomes in your deployment.

---

## 10. References

**Effort Regulation Concepts**:
- Thinking Budgets: Kimi K2 (https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- Reasoning Budgets: DeepSeek R1 research
- Adaptive Compute: Test-time compute scaling literature

**Quality Evaluation**:
- Ragas Framework: https://docs.ragas.io/

**Pareto Optimization**:
- Multi-objective optimization literature
- Efficiency frontiers in ML systems

---

**Document Version**: 2.1
**Date**: 2025-11-08
**Status**: Theoretical framework, model-agnostic
