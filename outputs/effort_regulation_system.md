# Executive Brain: Effort Regulation & Thinking Budget Management System

## Document Overview

**Purpose**: Design sophisticated effort regulation system for dynamic task complexity management
**Model**: Kimi K2 Thinking (Moonshot AI) as primary LLM
**Version**: 2.0 (extends base architecture)
**Date**: 2025-11-08

---

## 1. Kimi K2 Thinking Model Specifications

### 1.1 Model Architecture

| Parameter | Value | Implication |
|-----------|-------|-------------|
| **Total Parameters** | 1 trillion (MoE) | Massive capacity for complex reasoning |
| **Active Parameters** | 32 billion per forward pass | Cost-efficient inference |
| **Context Window** | 256K tokens | Can maintain very long reasoning chains |
| **Quantization** | Native INT4 | Reduced inference latency and GPU memory |
| **Training Cost** | $4.6M | Cost-efficient for trillion-parameter model |
| **Inference Cost** | 1/12th of Claude 4 Sonnet | Significant cost advantage for self-hosting |
| **License** | Modified MIT (open source) | Self-hostable, customizable |

### 1.2 Thinking Budget Capabilities

**Variable Thinking Token Budgets** (from Moonshot AI benchmarks):

| Task Type | Max Steps | Thinking Budget per Step | Total Budget | Use Case |
|-----------|-----------|-------------------------|--------------|----------|
| **Simple Tasks** | 1-10 | 1K-4K tokens | 4K-40K | Factual queries, simple retrieval |
| **Standard Reasoning** | 10-50 | 4K-24K tokens | 40K-96K | Multi-step analysis, synthesis |
| **Complex Reasoning** | 50-120 | 24K-48K tokens | 96K-128K | Mathematical proofs, deep analysis |
| **Agentic Workflows** | 100-300 | 24K tokens | 128K-256K | Research, coding, tool orchestration |

**Key Insight**: Kimi K2 can execute **200-300 sequential tool calls** with interleaved reasoning, enabling autonomous multi-step workflows.

### 1.3 Performance Benchmarks

| Benchmark | Kimi K2 Thinking | GPT-5 | Claude 4.5 Sonnet |
|-----------|------------------|-------|-------------------|
| **BrowseComp** | 60.2% | 54.9% | 24.1% |
| **GPQA Diamond** | 85.7% | 84.5% | N/A |
| **HLE (Humanity's Last Exam)** | 44.9% | N/A | N/A |
| **SWE-Bench Verified** | 71.3% | N/A | N/A |
| **LiveCodeBench v6** | 83.1% | N/A | N/A |

**Conclusion**: Kimi K2 Thinking outperforms or matches top proprietary models while being open source and self-hostable.

---

## 2. Effort Regulation Architecture

### 2.1 Core Concept: Dynamic Thinking Budget Allocation

The **Effort Regulation Orchestrator** analyzes each incoming task and dynamically allocates:
1. **Thinking Budget** (token count for reasoning)
2. **Max Steps** (iteration limit for multi-step reasoning)
3. **Timeout** (wall-clock time limit)
4. **Quality Threshold** (minimum acceptable accuracy)

**Trade-off Space**:
```
            Fast ←──────────→ Slow
            Cheap ←─────────→ Expensive
            Lower Accuracy ←→ Higher Accuracy
```

### 2.2 Effort Regulation Orchestrator Component

**New Layer in Architecture** (insert between Input Normalization and Orchestration):

```
┌─────────────────────────────────────────────────────────────┐
│            EFFORT REGULATION ORCHESTRATOR                    │
│                                                              │
│  Input: Normalized Event + User Context                     │
│  Output: Effort Allocation Strategy                         │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Task Complexity Analyzer                           │ │
│  │     → Intrinsic Complexity Score (0.0 - 1.0)          │ │
│  │     → Reasoning Depth Required                         │ │
│  │     → Tool Orchestration Complexity                    │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  2. Context Analyzer                                   │ │
│  │     → User Urgency (real-time vs batch)               │ │
│  │     → Risk Level (low-risk can use less effort)       │ │
│  │     → Budget Constraints (remaining daily budget)     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  3. Effort Allocation Strategy Selector               │ │
│  │     → Select from: Fast, Balanced, Thorough, Maximum  │ │
│  │     → Map to Kimi K2 budget parameters                │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  4. Feedback Loop from Evaluation                     │ │
│  │     → If quality insufficient, retry with more effort │ │
│  │     → Learn optimal effort levels per task type       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Task Complexity Analysis Framework

### 3.1 Intrinsic Complexity Scoring

**Multi-Dimensional Complexity Score**:

```python
def compute_intrinsic_complexity(task):
    """
    Compute intrinsic complexity score (0.0 - 1.0) based on multiple dimensions.
    """
    dimensions = {
        "reasoning_depth": analyze_reasoning_depth(task),
        "knowledge_breadth": analyze_knowledge_breadth(task),
        "tool_orchestration": analyze_tool_complexity(task),
        "ambiguity": analyze_ambiguity(task),
        "constraints": analyze_constraints(task),
        "novelty": analyze_novelty(task)
    }

    # Weighted average
    weights = {
        "reasoning_depth": 0.25,
        "knowledge_breadth": 0.15,
        "tool_orchestration": 0.20,
        "ambiguity": 0.15,
        "constraints": 0.15,
        "novelty": 0.10
    }

    complexity_score = sum(
        dimensions[dim] * weights[dim]
        for dim in dimensions
    )

    return complexity_score
```

#### 3.1.1 Reasoning Depth Analysis

```python
def analyze_reasoning_depth(task):
    """
    Score: 0.0 (single-step) to 1.0 (deep multi-step reasoning)
    """
    indicators = {
        # Simple factual retrieval
        "single_step_patterns": [
            r"what is",
            r"who is",
            r"when did",
            r"define",
            r"status of"
        ],
        # Multi-step reasoning
        "multi_step_patterns": [
            r"analyze.*and.*recommend",
            r"compare.*and.*decide",
            r"why.*because",
            r"prove",
            r"optimize"
        ],
        # Deep reasoning
        "deep_reasoning_patterns": [
            r"synthesize.*from.*multiple",
            r"derive.*from.*first principles",
            r"design.*system.*with.*constraints",
            r"simulate.*outcomes"
        ]
    }

    if any(re.search(p, task.lower()) for p in indicators["deep_reasoning_patterns"]):
        return 1.0
    elif any(re.search(p, task.lower()) for p in indicators["multi_step_patterns"]):
        return 0.6
    elif any(re.search(p, task.lower()) for p in indicators["single_step_patterns"]):
        return 0.2
    else:
        # Default: use LLM to classify
        return llm_classify_reasoning_depth(task)
```

#### 3.1.2 Knowledge Breadth Analysis

```python
def analyze_knowledge_breadth(task):
    """
    Score: 0.0 (narrow domain) to 1.0 (cross-domain synthesis)
    """
    # Extract entities and concepts
    entities = extract_entities(task)
    concepts = extract_concepts(task)

    # Query knowledge graph for domain diversity
    domains = set()
    for entity in entities:
        entity_node = neo4j.query(f"MATCH (e {{name: '{entity}'}}) RETURN labels(e)")
        domains.update(entity_node.labels)

    # More domains = higher complexity
    domain_count = len(domains)

    if domain_count == 1:
        return 0.2  # Single domain (e.g., "What's the Phoenix project status?")
    elif domain_count == 2:
        return 0.4  # Two domains (e.g., "How does Project X impact budget?")
    elif domain_count >= 3:
        return 0.7  # Multi-domain (e.g., "Analyze Q3 financials, HR capacity, and market trends")

    return 0.5  # Default
```

#### 3.1.3 Tool Orchestration Complexity

```python
def analyze_tool_complexity(task):
    """
    Score: 0.0 (no tools) to 1.0 (complex tool orchestration)
    """
    # Predict required tools
    likely_tools = predict_required_tools(task)

    tool_count = len(likely_tools)
    tool_dependencies = analyze_tool_dependencies(likely_tools)

    # Complexity factors
    base_complexity = min(tool_count / 10, 1.0)  # Max out at 10 tools
    dependency_complexity = len(tool_dependencies) / 20  # Sequential dependencies

    return min(base_complexity + dependency_complexity, 1.0)

def predict_required_tools(task):
    """
    Use lightweight LLM to predict which tools will be needed.
    """
    prompt = f"""
    Given this task: "{task}"

    Which tools from this list will likely be needed?
    - database_query
    - email_send
    - calendar_create
    - web_search
    - file_read
    - crm_update
    - financial_model

    Return JSON list of tool names.
    """

    result = kimi_k2_fast.generate(prompt, max_tokens=100)
    return json.loads(result)
```

#### 3.1.4 Ambiguity Analysis

```python
def analyze_ambiguity(task):
    """
    Score: 0.0 (clear, unambiguous) to 1.0 (highly ambiguous)
    """
    ambiguity_indicators = [
        r"\bor\b",  # "Should we do X or Y?"
        r"maybe",
        r"possibly",
        r"unclear",
        r"not sure",
        r"depends on",
        r"various",
        r"several options"
    ]

    # Count ambiguity markers
    ambiguity_count = sum(
        1 for indicator in ambiguity_indicators
        if re.search(indicator, task.lower())
    )

    # Missing key information
    if "?" in task and len(task.split()) < 10:
        ambiguity_count += 1  # Vague short question

    return min(ambiguity_count / 5, 1.0)
```

#### 3.1.5 Constraint Analysis

```python
def analyze_constraints(task):
    """
    Score: 0.0 (no constraints) to 1.0 (many complex constraints)
    """
    constraint_indicators = [
        r"without",
        r"must not",
        r"cannot",
        r"while maintaining",
        r"subject to",
        r"within.*budget",
        r"by.*deadline",
        r"compliant with"
    ]

    constraint_count = sum(
        1 for indicator in constraint_indicators
        if re.search(indicator, task.lower())
    )

    return min(constraint_count / 5, 1.0)
```

#### 3.1.6 Novelty Analysis

```python
def analyze_novelty(task):
    """
    Score: 0.0 (seen before, cached) to 1.0 (completely novel)
    """
    # Check if similar task has been processed before
    similar_tasks = neo4j.vector_search(
        embedding=embed(task),
        node_label="Decision",
        top_k=5
    )

    if similar_tasks and similar_tasks[0]["similarity"] > 0.95:
        # Nearly identical task seen before
        return 0.1
    elif similar_tasks and similar_tasks[0]["similarity"] > 0.8:
        # Similar task seen before
        return 0.3
    else:
        # Novel task
        return 0.8
```

### 3.2 Composite Complexity Score Example

```python
task = "Analyze Q3 spending across all departments, identify cost reduction opportunities that won't impact revenue, simulate 10% budget cut scenarios, and recommend which departments should absorb cuts while maintaining compliance with labor laws."

complexity = compute_intrinsic_complexity(task)
# Result:
# {
#   "reasoning_depth": 0.9,  # Deep analysis + simulation
#   "knowledge_breadth": 0.8,  # Finance + HR + Legal domains
#   "tool_orchestration": 0.7,  # Database, financial model, simulation
#   "ambiguity": 0.3,  # Relatively clear objective
#   "constraints": 0.8,  # Revenue constraint, legal compliance
#   "novelty": 0.6  # Novel combination
# }
#
# Weighted score: 0.73 (HIGH COMPLEXITY)
```

---

## 4. Context-Aware Effort Adjustment

### 4.1 User Context Factors

```python
class ContextAnalyzer:
    def analyze_user_context(self, task, user, system_state):
        """
        Adjust effort based on user context and system constraints.
        """
        factors = {
            "urgency": self.assess_urgency(task, user),
            "risk": self.assess_risk(task, user),
            "budget": self.check_budget_remaining(system_state),
            "user_preference": self.get_user_effort_preference(user),
            "time_of_day": self.assess_time_sensitivity(task)
        }

        return factors

    def assess_urgency(self, task, user):
        """
        Score: 0.0 (batch/async) to 1.0 (real-time critical)
        """
        urgency_markers = {
            "critical": ["urgent", "asap", "immediately", "critical", "emergency"],
            "high": ["today", "this morning", "by EOD"],
            "medium": ["this week", "soon"],
            "low": ["when you can", "no rush", "batch"]
        }

        task_lower = task.lower()

        if any(marker in task_lower for marker in urgency_markers["critical"]):
            return 1.0
        elif any(marker in task_lower for marker in urgency_markers["high"]):
            return 0.7
        elif any(marker in task_lower for marker in urgency_markers["medium"]):
            return 0.5
        else:
            return 0.2

    def assess_risk(self, task, user):
        """
        Score: 0.0 (low-risk, can use less effort) to 1.0 (high-risk, need max effort)
        """
        # Get risk score from governance layer (calculated earlier)
        risk_score = calculate_risk_score(task, user)
        return risk_score

    def check_budget_remaining(self, system_state):
        """
        Score: 0.0 (budget exhausted, minimize effort) to 1.0 (plenty of budget)
        """
        daily_budget = system_state["daily_llm_budget"]
        spent_today = system_state["llm_cost_today"]

        remaining_ratio = (daily_budget - spent_today) / daily_budget
        return max(remaining_ratio, 0.0)

    def get_user_effort_preference(self, user):
        """
        Some users may prefer speed, others prefer accuracy.
        Score: 0.0 (speed preferred) to 1.0 (accuracy preferred)
        """
        return user.preferences.get("effort_preference", 0.5)
```

### 4.2 Effort Adjustment Formula

```python
def compute_final_effort_allocation(intrinsic_complexity, context_factors):
    """
    Combine intrinsic complexity with context to determine final effort allocation.
    """
    # Base effort from intrinsic complexity
    base_effort = intrinsic_complexity

    # Adjustments
    urgency_multiplier = 1.0 - (context_factors["urgency"] * 0.3)  # High urgency = reduce effort (trade accuracy for speed)
    risk_multiplier = 1.0 + (context_factors["risk"] * 0.5)  # High risk = increase effort
    budget_multiplier = context_factors["budget"]  # Low budget = reduce effort
    preference_multiplier = 0.7 + (context_factors["user_preference"] * 0.6)  # Range: 0.7 - 1.3

    # Composite multiplier
    multiplier = urgency_multiplier * risk_multiplier * budget_multiplier * preference_multiplier

    final_effort = min(base_effort * multiplier, 1.0)

    return final_effort
```

**Example**:
```python
task = "Analyze Q3 spending..."
intrinsic_complexity = 0.73

context_factors = {
    "urgency": 0.3,  # No rush
    "risk": 0.6,  # Medium-high risk (financial decisions)
    "budget": 0.8,  # Plenty of budget remaining
    "user_preference": 0.7,  # User prefers accuracy
    "time_of_day": "business_hours"
}

final_effort = compute_final_effort_allocation(0.73, context_factors)
# urgency_multiplier = 1.0 - (0.3 * 0.3) = 0.91
# risk_multiplier = 1.0 + (0.6 * 0.5) = 1.3
# budget_multiplier = 0.8
# preference_multiplier = 0.7 + (0.7 * 0.6) = 1.12
#
# multiplier = 0.91 * 1.3 * 0.8 * 1.12 = 1.06
# final_effort = min(0.73 * 1.06, 1.0) = 0.77

# Result: HIGH EFFORT allocation
```

---

## 5. Effort Allocation Strategies

### 5.1 Strategy Mapping

```python
EFFORT_STRATEGIES = {
    "minimal": {
        "description": "Fastest, lowest cost, acceptable for low-stakes queries",
        "effort_range": (0.0, 0.2),
        "kimi_k2_params": {
            "max_steps": 5,
            "thinking_budget_per_step": "1K tokens",
            "total_budget": "5K tokens",
            "temperature": 0.3,
            "timeout": "10 seconds"
        },
        "use_cases": ["simple factual queries", "cache hits", "status checks"]
    },
    "fast": {
        "description": "Quick response, good for routine tasks",
        "effort_range": (0.2, 0.4),
        "kimi_k2_params": {
            "max_steps": 10,
            "thinking_budget_per_step": "4K tokens",
            "total_budget": "40K tokens",
            "temperature": 0.5,
            "timeout": "30 seconds"
        },
        "use_cases": ["routine analysis", "simple recommendations", "data retrieval"]
    },
    "balanced": {
        "description": "Default strategy, balances speed and accuracy",
        "effort_range": (0.4, 0.6),
        "kimi_k2_params": {
            "max_steps": 50,
            "thinking_budget_per_step": "12K tokens",
            "total_budget": "96K tokens",
            "temperature": 0.7,
            "timeout": "2 minutes"
        },
        "use_cases": ["standard reasoning", "multi-step analysis", "synthesis tasks"]
    },
    "thorough": {
        "description": "Deep reasoning, higher accuracy, slower",
        "effort_range": (0.6, 0.8),
        "kimi_k2_params": {
            "max_steps": 120,
            "thinking_budget_per_step": "32K tokens",
            "total_budget": "128K tokens",
            "temperature": 0.8,
            "timeout": "5 minutes"
        },
        "use_cases": ["complex analysis", "high-risk decisions", "research tasks"]
    },
    "maximum": {
        "description": "Maximum reasoning effort, highest accuracy",
        "effort_range": (0.8, 1.0),
        "kimi_k2_params": {
            "max_steps": 300,
            "thinking_budget_per_step": "48K tokens",
            "total_budget": "256K tokens",
            "temperature": 0.9,
            "timeout": "15 minutes"
        },
        "use_cases": ["critical decisions", "novel problems", "agentic workflows", "proofs"]
    }
}

def select_strategy(final_effort_score):
    """
    Map final effort score to strategy.
    """
    for strategy_name, strategy in EFFORT_STRATEGIES.items():
        min_effort, max_effort = strategy["effort_range"]
        if min_effort <= final_effort_score < max_effort:
            return strategy_name, strategy

    # Default to maximum if score is 1.0
    return "maximum", EFFORT_STRATEGIES["maximum"]
```

### 5.2 Strategy Application

```python
class EffortRegulationOrchestrator:
    def regulate_effort(self, task, user, system_state):
        """
        Main entry point for effort regulation.
        """
        # 1. Analyze intrinsic complexity
        intrinsic_complexity = compute_intrinsic_complexity(task)

        # 2. Analyze context
        context_factors = self.context_analyzer.analyze_user_context(
            task, user, system_state
        )

        # 3. Compute final effort allocation
        final_effort = compute_final_effort_allocation(
            intrinsic_complexity, context_factors
        )

        # 4. Select strategy
        strategy_name, strategy = select_strategy(final_effort)

        # 5. Log decision
        self.log_effort_decision(
            task=task,
            intrinsic_complexity=intrinsic_complexity,
            context_factors=context_factors,
            final_effort=final_effort,
            strategy=strategy_name
        )

        # 6. Return configuration for Kimi K2
        return {
            "strategy": strategy_name,
            "kimi_k2_params": strategy["kimi_k2_params"],
            "intrinsic_complexity": intrinsic_complexity,
            "final_effort": final_effort
        }
```

---

## 6. Integration with ROMA Recursive Meta-Agent

### 6.1 Per-Subtask Effort Allocation

```python
class ROMAWithEffortRegulation(MetaAgent):
    """
    ROMA meta-agent enhanced with per-subtask effort allocation.
    """

    def plan(self, task, parent_effort_budget):
        """
        Override ROMA planner to include effort allocation for each subtask.
        """
        # Decompose task into subtasks (ROMA's standard planning)
        subtasks = self.standard_planner(task)

        # Allocate effort budget across subtasks
        for subtask in subtasks:
            # Analyze subtask complexity
            subtask_complexity = compute_intrinsic_complexity(subtask["task"])

            # Inherit context from parent (urgency, risk, budget)
            # But allow subtask-specific adjustments
            subtask_effort = self.allocate_subtask_effort(
                subtask_complexity=subtask_complexity,
                parent_effort_budget=parent_effort_budget,
                subtask_priority=subtask.get("priority", 0.5)
            )

            # Assign effort strategy to subtask
            strategy_name, strategy = select_strategy(subtask_effort)
            subtask["effort_strategy"] = strategy_name
            subtask["kimi_k2_params"] = strategy["kimi_k2_params"]
            subtask["allocated_effort"] = subtask_effort

        return subtasks

    def allocate_subtask_effort(self, subtask_complexity, parent_effort_budget, subtask_priority):
        """
        Allocate effort to a subtask based on:
        - Its intrinsic complexity
        - Parent's total effort budget
        - Subtask's priority (critical path gets more effort)
        """
        # Base allocation from parent budget
        base_allocation = parent_effort_budget * 0.8  # Reserve 20% for aggregation

        # Adjust by subtask priority
        priority_multiplier = 0.5 + (subtask_priority * 1.0)  # Range: 0.5 - 1.5

        # Adjust by subtask complexity
        complexity_multiplier = 0.7 + (subtask_complexity * 0.6)  # Range: 0.7 - 1.3

        subtask_effort = base_allocation * priority_multiplier * complexity_multiplier

        return min(subtask_effort, 1.0)
```

### 6.2 Example: Recursive Effort Allocation

```
Task: "Analyze Q3 spending and recommend 10% budget cuts"
Parent Effort Allocation: 0.77 (thorough strategy)

Decomposition:
├─ Subtask 1: "Retrieve Q3 spending by department"
│  ├─ Complexity: 0.2 (simple database query)
│  ├─ Priority: 0.9 (critical path)
│  ├─ Allocated Effort: 0.77 * 0.8 * 1.4 * 0.82 = 0.71 → "thorough" strategy
│  └─ Why high effort? Critical data quality needed
│
├─ Subtask 2: "Analyze spending vs budget vs revenue impact"
│  ├─ Complexity: 0.6 (multi-step analysis)
│  ├─ Priority: 1.0 (critical path)
│  ├─ Allocated Effort: 0.77 * 0.8 * 1.5 * 1.06 = 0.98 → "maximum" strategy
│  └─ Why max effort? Core reasoning task
│
├─ Subtask 3: "Identify low-impact cost reduction opportunities"
│  ├─ Complexity: 0.7 (creative reasoning)
│  ├─ Priority: 1.0 (critical path)
│  ├─ Allocated Effort: 0.77 * 0.8 * 1.5 * 1.12 = 1.03 → "maximum" strategy
│  └─ Why max effort? Novel problem, high stakes
│
├─ Subtask 4: "Simulate Q4 scenarios with 10% cuts"
│  ├─ Complexity: 0.5 (tool orchestration)
│  ├─ Priority: 0.8 (validation step)
│  ├─ Allocated Effort: 0.77 * 0.8 * 1.3 * 1.0 = 0.80 → "maximum" strategy
│  └─ Why max effort? Financial model accuracy critical
│
└─ Subtask 5: "Synthesize recommendations with risk assessment"
   ├─ Complexity: 0.8 (synthesis + risk analysis)
   ├─ Priority: 1.0 (final output)
   ├─ Allocated Effort: 0.77 * 0.8 * 1.5 * 1.18 = 1.09 → "maximum" strategy
   └─ Why max effort? Executive decision quality critical
```

---

## 7. Adaptive Effort with Feedback Loop

### 7.1 Quality-Based Retry Logic

```python
class AdaptiveEffortOrchestrator:
    def execute_with_adaptive_effort(self, task, initial_effort_config):
        """
        Execute task with adaptive effort - retry with more effort if quality is insufficient.
        """
        max_retries = 3
        current_effort_config = initial_effort_config

        for attempt in range(max_retries):
            # Execute task with current effort configuration
            result = self.execute_task(task, current_effort_config)

            # Evaluate result quality (using Ragas)
            evaluation = self.evaluate_result(task, result)

            # Check quality gates
            if evaluation["faithfulness"] >= 0.8 and evaluation["answer_relevance"] >= 0.7:
                # Quality acceptable
                self.log_success(task, current_effort_config, evaluation, attempt)
                return result
            else:
                # Quality insufficient - increase effort
                logger.warning(f"Attempt {attempt + 1} failed quality gates. Increasing effort.")

                # Increase effort for retry
                current_effort_config = self.increase_effort(current_effort_config)

                if attempt == max_retries - 1:
                    # Final attempt failed - flag for human review
                    self.flag_for_human_review(task, result, evaluation)
                    return result  # Return best attempt

    def increase_effort(self, current_config):
        """
        Move up to next effort strategy.
        """
        strategy_ladder = ["minimal", "fast", "balanced", "thorough", "maximum"]
        current_strategy = current_config["strategy"]
        current_index = strategy_ladder.index(current_strategy)

        if current_index < len(strategy_ladder) - 1:
            next_strategy = strategy_ladder[current_index + 1]
            return {
                "strategy": next_strategy,
                "kimi_k2_params": EFFORT_STRATEGIES[next_strategy]["kimi_k2_params"],
                "retry_attempt": current_config.get("retry_attempt", 0) + 1
            }
        else:
            # Already at maximum - can't increase further
            return current_config
```

### 7.2 Learning Optimal Effort Levels

```python
class EffortLearningModule:
    """
    Learn optimal effort levels for different task types over time.
    """

    def record_effort_outcome(self, task_type, effort_allocated, quality_achieved, cost):
        """
        Record outcome in Neo4j for later analysis.
        """
        neo4j.create_node(
            "EffortOutcome",
            {
                "task_type": task_type,
                "effort_allocated": effort_allocated,
                "quality_faithfulness": quality_achieved["faithfulness"],
                "quality_relevance": quality_achieved["answer_relevance"],
                "cost_dollars": cost,
                "timestamp": datetime.now()
            }
        )

    def learn_optimal_effort(self, task_type):
        """
        Analyze historical outcomes to find optimal effort level.

        Goal: Minimize cost while maintaining quality > threshold
        """
        # Query historical outcomes
        outcomes = neo4j.query(f"""
            MATCH (o:EffortOutcome {{task_type: '{task_type}'}})
            WHERE o.timestamp > datetime() - duration('P30D')
            RETURN o.effort_allocated as effort,
                   o.quality_faithfulness as faithfulness,
                   o.quality_relevance as relevance,
                   o.cost_dollars as cost
            ORDER BY o.timestamp DESC
            LIMIT 100
        """)

        # Find Pareto frontier (quality vs cost)
        pareto_points = self.find_pareto_frontier(outcomes)

        # Select point that meets quality threshold with minimum cost
        optimal_effort = min(
            (p for p in pareto_points if p["faithfulness"] >= 0.8 and p["relevance"] >= 0.7),
            key=lambda p: p["cost"],
            default=None
        )

        if optimal_effort:
            # Update default effort for this task type
            self.update_task_type_effort(task_type, optimal_effort["effort"])
            return optimal_effort["effort"]
        else:
            # Not enough data or quality not achievable - use default
            return 0.5
```

---

## 8. Cost Optimization with Effort Regulation

### 8.1 Cost Modeling

```python
def estimate_cost(effort_config):
    """
    Estimate cost of executing task with given effort configuration.

    Kimi K2 Thinking pricing (self-hosted or API):
    - Input: $X per 1M tokens
    - Output (thinking): $Y per 1M tokens
    - Output (response): $Z per 1M tokens
    """
    kimi_k2_params = effort_config["kimi_k2_params"]

    # Estimate input tokens (context + task)
    input_tokens = 10000  # Typical context size

    # Estimate thinking tokens
    thinking_tokens = (
        kimi_k2_params["max_steps"] *
        parse_token_count(kimi_k2_params["thinking_budget_per_step"])
    )

    # Estimate response tokens
    response_tokens = 2000  # Typical response

    # Cost calculation (example pricing)
    INPUT_COST_PER_M = 0.50  # $0.50 per 1M input tokens
    THINKING_COST_PER_M = 1.00  # $1.00 per 1M thinking tokens
    OUTPUT_COST_PER_M = 2.00  # $2.00 per 1M output tokens

    total_cost = (
        (input_tokens / 1_000_000) * INPUT_COST_PER_M +
        (thinking_tokens / 1_000_000) * THINKING_COST_PER_M +
        (response_tokens / 1_000_000) * OUTPUT_COST_PER_M
    )

    return total_cost

def parse_token_count(budget_str):
    """Parse '4K tokens' -> 4000"""
    if "K" in budget_str:
        return int(budget_str.split("K")[0]) * 1000
    return int(budget_str.split()[0])
```

### 8.2 Cost-Quality Trade-off Optimization

```python
def optimize_effort_for_budget(task, max_budget_dollars):
    """
    Find highest effort level that fits within budget.
    """
    strategies_by_cost = sorted(
        EFFORT_STRATEGIES.items(),
        key=lambda x: estimate_cost({"kimi_k2_params": x[1]["kimi_k2_params"]})
    )

    for strategy_name, strategy in strategies_by_cost:
        estimated_cost = estimate_cost({"kimi_k2_params": strategy["kimi_k2_params"]})
        if estimated_cost <= max_budget_dollars:
            # Found highest effort within budget
            return strategy_name, strategy

    # Even minimal strategy exceeds budget - flag error
    raise BudgetExceededError(f"Task cannot be executed within budget ${max_budget_dollars}")
```

---

## 9. Implementation in Executive Brain Architecture

### 9.1 Updated System Flow

```
1. Input Normalization Layer
   ↓
2. Effort Regulation Orchestrator ← NEW
   ├─ Task Complexity Analysis
   ├─ Context Analysis (urgency, risk, budget)
   ├─ Effort Allocation Strategy Selection
   └─ Kimi K2 Parameter Configuration
   ↓
3. Orchestration & Reasoning Layer
   ├─ Kimi K2 Thinking (with configured effort parameters)
   ├─ ROMA (with per-subtask effort allocation)
   └─ LangGraph (deterministic workflows)
   ↓
4. [Rest of architecture unchanged]
```

### 9.2 Kimi K2 Thinking Integration

```python
class KimiK2ThinkingExecutor:
    def __init__(self, api_endpoint="https://api.moonshot.ai/v1"):
        self.api_endpoint = api_endpoint
        self.client = MoonshotClient(api_endpoint)

    def execute(self, task, effort_config, contexts):
        """
        Execute task using Kimi K2 Thinking with effort-regulated parameters.
        """
        kimi_params = effort_config["kimi_k2_params"]

        # Construct prompt with context
        prompt = self.construct_prompt(task, contexts)

        # Call Kimi K2 Thinking API
        response = self.client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=[
                {"role": "system", "content": "You are an executive AI assistant with access to tools and knowledge graphs."},
                {"role": "user", "content": prompt}
            ],
            max_steps=kimi_params["max_steps"],
            thinking_budget_per_step=parse_token_count(kimi_params["thinking_budget_per_step"]),
            temperature=kimi_params["temperature"],
            timeout=kimi_params["timeout"],
            stream=True  # Enable streaming for real-time feedback
        )

        # Process streaming response
        full_response = ""
        thinking_trace = []

        for chunk in response:
            if chunk.type == "thinking":
                thinking_trace.append(chunk.content)
            elif chunk.type == "content":
                full_response += chunk.content

        return {
            "response": full_response,
            "thinking_trace": thinking_trace,
            "effort_config": effort_config,
            "steps_used": len(thinking_trace),
            "tokens_used": self.count_tokens(thinking_trace + [full_response])
        }
```

### 9.3 Monitoring Dashboard

**New Metrics to Track**:

```python
# Prometheus metrics for effort regulation
effort_allocation_distribution = Histogram(
    "executive_brain_effort_allocation",
    "Distribution of effort allocations",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

effort_strategy_count = Counter(
    "executive_brain_effort_strategy_total",
    "Count of tasks by effort strategy",
    ["strategy"]
)

effort_retry_count = Counter(
    "executive_brain_effort_retry_total",
    "Count of retries due to insufficient quality",
    ["initial_strategy", "final_strategy"]
)

effort_vs_quality = Histogram(
    "executive_brain_effort_vs_quality",
    "Quality achieved vs effort allocated",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

cost_per_effort_level = Histogram(
    "executive_brain_cost_per_effort_dollars",
    "Cost per effort level",
    ["strategy"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
)
```

---

## 10. Deployment Considerations for Kimi K2

### 10.1 Self-Hosting vs API

| Deployment | Pros | Cons | Recommendation |
|------------|------|------|----------------|
| **Self-Hosted** | - Full control<br>- No API rate limits<br>- Data privacy<br>- Long-term cost savings | - Requires GPU infrastructure (A100s)<br>- Operational complexity<br>- Upfront hardware cost | For high-volume (>100K requests/day) |
| **API (Moonshot.ai)** | - Zero infrastructure<br>- Instant start<br>- Managed scaling | - Rate limits<br>- Data leaves premises<br>- Per-token pricing | For MVP and low-medium volume |
| **Hybrid** | - Self-host for routine tasks<br>- API for burst capacity | - Complexity of dual deployment | For enterprise with variable load |

### 10.2 Self-Hosting Infrastructure

**Minimum Requirements**:
- **GPU**: 4x NVIDIA A100 80GB (for INT4 quantized Kimi K2)
- **RAM**: 256GB system RAM
- **Storage**: 500GB SSD for model weights
- **Network**: 10Gbps for fast model loading

**Deployment Stack**:
```yaml
# Kubernetes deployment for self-hosted Kimi K2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kimi-k2-inference
spec:
  replicas: 2  # For redundancy
  selector:
    matchLabels:
      app: kimi-k2
  template:
    spec:
      nodeSelector:
        gpu: "nvidia-a100"
      containers:
      - name: kimi-k2
        image: moonshotai/kimi-k2-thinking:int4
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "128Gi"
          requests:
            nvidia.com/gpu: 4
            memory: "128Gi"
        env:
        - name: MAX_CONCURRENT_REQUESTS
          value: "10"
        - name: THINKING_BUDGET_LIMIT
          value: "256000"  # 256K token max
```

**Estimated Self-Hosting Cost**:
- **Cloud GPU (AWS p4d.24xlarge)**: ~$32/hour = $23,040/month (full-time)
- **Spot instances**: ~$10/hour = $7,200/month (60% savings)
- **Amortized hardware (4x A100)**: ~$80,000 / 36 months = $2,222/month

**Break-even Analysis**:
- If API cost > $7,200/month → Self-host on spot instances
- If API cost > $23,000/month → Self-host on reserved instances

---

## 11. Summary

### 11.1 Key Innovations

1. **Dynamic Thinking Budget Allocation**: Kimi K2's variable thinking budgets (1K - 256K tokens) enable cost-efficient execution
2. **Multi-Dimensional Complexity Analysis**: 6-factor scoring (reasoning depth, knowledge breadth, tool orchestration, ambiguity, constraints, novelty)
3. **Context-Aware Effort Adjustment**: Urgency, risk, budget, and user preferences modulate effort allocation
4. **Adaptive Retry with Escalation**: Quality-based retry logic automatically increases effort if initial attempt fails
5. **Per-Subtask Effort in ROMA**: Recursive meta-agent allocates effort to each subtask based on priority and complexity
6. **Learning-Based Optimization**: Historical outcomes inform future effort allocations (Pareto optimization)

### 11.2 Expected Benefits

| Benefit | Impact | Measurement |
|---------|--------|-------------|
| **Cost Reduction** | 40-60% vs uniform maximum effort | Monthly LLM API cost |
| **Latency Improvement** | 3-5x faster for simple tasks | p95 latency by task complexity |
| **Quality Maintenance** | >90% tasks meet quality thresholds on first attempt | Ragas scores distribution |
| **Budget Predictability** | ±10% variance from forecast | Actual vs budgeted spend |
| **Autonomous Optimization** | Continuous improvement over 6 months | Effort allocation drift |

### 11.3 Open Questions for User

1. **Self-host vs API**: Do you have GPU infrastructure or prefer API deployment?
2. **Effort Preferences**: Default to "balanced" or allow per-user customization?
3. **Cost vs Quality**: What's acceptable quality score for cost savings (e.g., 0.75 faithfulness OK if 50% cheaper)?
4. **Urgency Handling**: Should urgent requests always get maximum effort, or trade quality for speed?

---

**Document Version**: 2.0
**Date**: 2025-11-08
**Status**: Draft - requires user feedback on open questions
**Integration**: Extends `outputs/architecture.md` with effort regulation layer
