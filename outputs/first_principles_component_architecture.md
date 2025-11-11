# Executive Brain: First-Principles Component Architecture

## Document Overview

**Purpose**: Derive minimal necessary components for autonomous task execution from first principles
**Approach**: Bottom-up reasoning from fundamental constraints and requirements
**Based On**: Cold email campaign analysis + distributed systems theory + control theory + economics
**Date**: 2025-11-09

---

## 1. First-Principles Analysis

### 1.1 What Is the System Fundamentally Doing?

**Core Function**: Transform high-level goals into successful real-world outcomes through autonomous agent coordination.

**Fundamental Loop**:
```
Goal → Understand → Plan → Execute → Observe → Learn → Adapt → Success/Failure
         ↑                                                            |
         └────────────────────────────────────────────────────────────┘
```

### 1.2 Fundamental Constraints (Physics of Autonomous Systems)

**Constraint 1: Causality**
- Actions have consequences
- Time flows forward (no undo without cost)
- Effects propagate through dependency chains

**Constraint 2: Scarcity**
- Resources are finite (compute, money, time, API quotas)
- Contention is inevitable (multiple agents want same resource)
- Allocation determines outcomes

**Constraint 3: Uncertainty**
- Future is unknown (actions may fail)
- Information is incomplete (partial observability)
- Probability governs outcomes

**Constraint 4: Complexity**
- Tasks decompose into subtasks (hierarchical)
- Dependencies create ordering constraints (DAG)
- Parallelism has limits (Amdahl's law)

**Constraint 5: Adaptation**
- Context changes during execution
- Plans become stale
- Learning requires memory

**Constraint 6: Reliability**
- Agents fail
- Networks partition
- State becomes inconsistent

### 1.3 Derived Requirements

From these constraints, we can derive what the system MUST have:

| Constraint | Requires Component |
|------------|-------------------|
| Causality | Causal knowledge graph, reasoning engine |
| Scarcity | Resource manager, cost accounting, scheduler |
| Uncertainty | Probabilistic reasoning, confidence tracking |
| Complexity | Task decomposer, dependency graph, parallel executor |
| Adaptation | Context synchronization, plan reviser |
| Reliability | Health monitor, rollback mechanism, retry logic |

---

## 2. Minimal Component Set (First Principles)

### 2.1 Seven Core Subsystems

From first principles, the system needs exactly **7 subsystems**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTIVE BRAIN CORE                          │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │  1. KNOWLEDGE │  │  2. REASONING │  │  3. PLANNING  │      │
│  │     SYSTEM    │──│     ENGINE    │──│    SYSTEM     │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│  ┌───────────────┐  ┌───────▼───────┐  ┌───────────────┐      │
│  │  4. EXECUTION │──│  5. RESOURCE  │──│  6. OBSERVATION│      │
│  │     SYSTEM    │  │     SYSTEM    │  │     SYSTEM    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │  7. LEARNING    │                          │
│                    │     SYSTEM      │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Why exactly 7?**
- **Knowledge**: What the system knows (facts, concepts, causality)
- **Reasoning**: How it makes decisions (logic, inference, hypothesis testing)
- **Planning**: How it decomposes goals (decomposition, scheduling, optimization)
- **Execution**: How it runs tasks (agent spawning, coordination, communication)
- **Resource**: How it manages constraints (allocation, contention, cost)
- **Observation**: How it tracks state (monitoring, error detection, metrics)
- **Learning**: How it improves (experience, adaptation, self-improvement)

Any fewer → system is incomplete. Any more → redundancy or over-specialization.

---

## 3. Component Specifications

### 3.1 Knowledge System (What the System Knows)

**Purpose**: Maintain persistent, queryable, causal model of world

**Sub-Components**:

```
Knowledge System
├── Fact Storage (Neo4j)
│   ├── Entities (Person, Project, Resource)
│   ├── Observations (revenue=$1M, status=completed)
│   └── Relations (WORKS_ON, REPORTS_TO)
│
├── Causal Graph (Neo4j + reasoning layer)
│   ├── Events (ProjectX_Completed, John_Hired)
│   ├── Causal Edges (CAUSES, ENABLES, PREVENTS)
│   └── Confidence Scores (probabilistic)
│
├── Temporal Index
│   ├── Timestamps on all nodes/edges
│   ├── Version history (fact evolution)
│   └── HAPPENED_BEFORE relations
│
└── Schema Manager
    ├── Node type definitions
    ├── Edge type constraints
    └── Validation rules
```

**Critical Operations**:
1. `query_facts(pattern)` → facts matching pattern
2. `query_causal_path(cause, effect)` → causal chain with confidence
3. `counterfactual(event_to_remove)` → what would have happened
4. `update_belief(fact, evidence)` → Bayesian update
5. `detect_contradiction(new_fact)` → consistency check

**Why This Design?**
- Graph structure captures relationships naturally (vs relational DB)
- Temporal indexing enables "what led to X?" queries
- Causal edges make reasoning explicit (vs implicit in LLM)
- Confidence scores quantify uncertainty

**Evidence**: Neo4j GraphRAG (verified), Pearl's causal graphs (research-backed)

---

### 3.2 Reasoning Engine (How It Makes Decisions)

**Purpose**: Combine neural (LLM) and symbolic (graph) reasoning

**Sub-Components**:

```
Reasoning Engine
├── Neural Reasoner (LLM)
│   ├── Pattern recognition
│   ├── Hypothesis generation
│   ├── Natural language understanding
│   └── Creative problem solving
│
├── Symbolic Reasoner (Graph Algorithms)
│   ├── Logical inference (if A→B and B→C then A→C)
│   ├── Constraint satisfaction (does plan satisfy constraints?)
│   ├── Graph traversal (find causal paths)
│   └── Consistency checking (detect contradictions)
│
├── Probabilistic Reasoner (Bayesian)
│   ├── Confidence propagation
│   ├── Uncertainty quantification
│   ├── Bayesian updates
│   └── Monte Carlo simulation
│
├── Hybrid Coordinator
│   ├── Route queries to appropriate reasoner
│   ├── Combine neural + symbolic results
│   └── Validate LLM outputs against graph
│
└── Adversarial Validator
    ├── Critique plans before execution
    ├── Identify flawed assumptions
    └── Suggest alternatives
```

**Critical Operations**:
1. `infer(premises, query)` → conclusion with confidence
2. `test_hypothesis(hypothesis, evidence)` → posterior probability
3. `validate_plan(plan, context)` → {approved, issues, alternatives}
4. `explain_decision(conclusion)` → reasoning trace
5. `detect_fallacy(reasoning)` → logical errors

**Why This Design?**
- LLMs are good at pattern matching but unreliable for logic
- Graph algorithms are reliable but brittle (need explicit rules)
- Hybrid combines strengths: LLM generates, graph validates
- Adversarial validation catches errors before expensive execution

**Evidence**: Neurosymbolic AI research (Garcez et al. 2019), adversarial collaboration (human cognition)

---

### 3.3 Planning System (How It Decomposes Goals)

**Purpose**: Transform goals into executable task graphs with optimal scheduling

**Sub-Components**:

```
Planning System
├── Task Decomposer (ROMA-style)
│   ├── Atomizer: Break goal into atomic tasks
│   ├── Complexity analyzer: Score each task
│   ├── Effort allocator: Assign compute budget
│   └── LLM-based decomposition
│
├── Dependency Analyzer
│   ├── Extract dependencies from causal graph
│   ├── Build DAG (directed acyclic graph)
│   ├── Detect cycles (invalid plans)
│   └── Infer implicit dependencies via reasoning
│
├── Scheduler
│   ├── Critical path analysis (CPM)
│   ├── Priority assignment (critical tasks first)
│   ├── Resource-aware scheduling (consider availability)
│   └── Parallel execution planning
│
├── Cost Estimator
│   ├── Query historical costs from knowledge graph
│   ├── Estimate task duration
│   ├── Calculate resource requirements
│   └── Compute expected value (benefit - cost)
│
└── Risk Assessor
    ├── Identify high-risk tasks (legal, reputation)
    ├── Estimate failure probability
    ├── Suggest mitigations
    └── Flag for human approval if risk > threshold
```

**Critical Operations**:
1. `decompose(goal)` → list of atomic tasks
2. `build_dependency_graph(tasks)` → DAG with critical path
3. `schedule(dag, resources)` → execution timeline
4. `estimate_cost(plan)` → {time, money, risk}
5. `optimize(plan, objective)` → Pareto-optimal plan

**Why This Design?**
- Decomposition separates concerns (what vs when vs how)
- DAG enforces ordering, prevents deadlocks
- Critical path identifies bottlenecks (what to optimize)
- Cost estimation enables budget-aware planning
- Risk assessment prevents catastrophic failures

**Evidence**: Project management (CPM/PERT), ROMA framework (verified), constraint programming

---

### 3.4 Execution System (How It Runs Tasks)

**Purpose**: Spawn, coordinate, and supervise agent execution

**Sub-Components**:

```
Execution System
├── Agent Pool
│   ├── Agent spawner (create new agents)
│   ├── Agent registry (track active agents)
│   ├── Agent templates (pre-configured for task types)
│   └── Agent capabilities (what each agent can do)
│
├── Coordinator
│   ├── Dispatch tasks to agents
│   ├── Route messages between agents
│   ├── Synchronize shared state
│   └── Handle agent-to-agent communication
│
├── Supervisor
│   ├── Health monitoring (heartbeat checks)
│   ├── Progress tracking (% complete)
│   ├── Timeout enforcement (kill stuck agents)
│   ├── Retry logic (transient failures)
│   └── Termination (graceful shutdown)
│
├── Context Manager
│   ├── Shared context store (global state)
│   ├── Context versioning (detect stale context)
│   ├── Event bus (pub-sub for updates)
│   └── Context synchronization (propagate changes)
│
└── Transaction Manager (Saga)
    ├── Compensation actions (undo operations)
    ├── Rollback on failure
    ├── Partial completion handling
    └── Idempotency tracking (prevent duplicate work)
```

**Critical Operations**:
1. `spawn_agent(task, resources, context)` → agent_id
2. `supervise_agent(agent_id)` → {status, progress, health}
3. `terminate_agent(agent_id, reason)` → cleanup
4. `synchronize_context(update)` → notify all agents
5. `rollback_saga(saga_id)` → undo completed steps

**Why This Design?**
- Agent pool provides isolation and scalability
- Supervisor prevents runaway agents (reliability)
- Context manager keeps agents synchronized (consistency)
- Saga pattern handles partial failures (transactions)

**Evidence**: Process supervision (systemd), distributed sagas (Garcia-Molina 1987), event sourcing (CQRS pattern)

---

### 3.5 Resource System (How It Manages Constraints)

**Purpose**: Allocate scarce resources, prevent contention, track costs

**Sub-Components**:

```
Resource System
├── Resource Registry
│   ├── Resource definitions (APIs, accounts, compute)
│   ├── Capacity limits (rate limits, quotas)
│   ├── Cost models ($ per API call)
│   └── Access policies (who can use what)
│
├── Allocation Manager
│   ├── Semaphores (mutual exclusion)
│   ├── Priority queues (critical tasks first)
│   ├── Deadlock detection (circular waits)
│   ├── Fair scheduling (prevent starvation)
│   └── Preemption (take resources from low-priority tasks)
│
├── Budget Manager
│   ├── Hierarchical budgets (total → per-task)
│   ├── Real-time tracking (spent vs allocated)
│   ├── Pre-flight checks (can afford operation?)
│   ├── Overspend alerts
│   └── Budget rebalancing (reallocate mid-execution)
│
├── Tool Access Control (RBAC)
│   ├── Role definitions (agent capabilities)
│   ├── Permission checks (can agent use tool?)
│   ├── Credential management (API keys, secrets)
│   ├── Usage tracking (audit log)
│   └── Policy engine (OPA/Cedar integration)
│
└── Cost Tracker
    ├── Per-task accounting (what did each task cost?)
    ├── Per-agent accounting (which agent spent most?)
    ├── Real-time dashboards (current burn rate)
    └── Cost optimization (suggest cheaper alternatives)
```

**Critical Operations**:
1. `allocate_resource(resource_id, agent_id)` → {granted, denied, wait_time}
2. `release_resource(resource_id, agent_id)` → success
3. `check_budget(agent_id, amount)` → can_spend
4. `charge_budget(agent_id, amount)` → new_balance
5. `check_access(agent_id, tool_id)` → {allowed, reason}

**Why This Design?**
- Semaphores prevent resource conflicts (mutual exclusion)
- Budget tracking prevents overspend (cost control)
- RBAC prevents unauthorized tool use (security)
- Hierarchical budgets enable delegation (scalability)

**Evidence**: OS resource management (semaphores), AWS IAM (RBAC), financial ERP systems (accounting)

---

### 3.6 Observation System (How It Tracks State)

**Purpose**: Monitor execution, detect errors, measure quality

**Sub-Components**:

```
Observation System
├── Metrics Collector
│   ├── Agent metrics (CPU, memory, duration)
│   ├── Task metrics (success rate, quality scores)
│   ├── Resource metrics (utilization, wait times)
│   ├── Cost metrics (spend by category)
│   └── System metrics (throughput, latency)
│
├── State Tracker
│   ├── Task status (pending, running, completed, failed)
│   ├── Agent status (healthy, stuck, crashed)
│   ├── Resource status (available, allocated, exhausted)
│   └── System status (operational, degraded, failing)
│
├── Error Detector
│   ├── Exception monitoring (catch failures)
│   ├── Anomaly detection (outlier behavior)
│   ├── Invariant checking (constraints violated?)
│   ├── Health checks (periodic liveness probes)
│   └── Alert system (notify on critical issues)
│
├── Quality Evaluator (Ragas)
│   ├── Faithfulness (no hallucinations?)
│   ├── Relevance (answers the question?)
│   ├── Context precision (uses right information?)
│   └── Custom metrics (domain-specific)
│
└── Distributed Tracing
    ├── Request correlation (trace ID propagation)
    ├── Span creation (timing each operation)
    ├── Dependency tracking (which services called)
    └── Performance profiling (bottleneck identification)
```

**Critical Operations**:
1. `track_metric(name, value, tags)` → stored
2. `get_agent_status(agent_id)` → {status, health, progress}
3. `detect_anomaly(metrics)` → {is_anomaly, severity}
4. `evaluate_quality(output, task)` → quality_scores
5. `query_trace(trace_id)` → execution_timeline

**Why This Design?**
- Metrics enable data-driven optimization
- State tracking enables coordination (who's doing what?)
- Error detection enables fast recovery (fail-fast)
- Quality evaluation enables adaptive retry (effort regulation)
- Distributed tracing enables debugging (observability)

**Evidence**: Prometheus (metrics), OpenTelemetry (tracing), Ragas (quality evaluation), anomaly detection (statistics)

---

### 3.7 Learning System (How It Improves)

**Purpose**: Extract lessons from experience, update beliefs, improve future performance

**Sub-Components**:

```
Learning System
├── Experience Recorder
│   ├── Task-outcome pairs (what worked, what didn't)
│   ├── Reasoning traces (how decisions were made)
│   ├── Error logs (what went wrong)
│   └── Success patterns (what led to success)
│
├── Post-Mortem Analyzer
│   ├── Compare actual vs expected outcomes
│   ├── Identify deviations from plan
│   ├── Generate hypotheses for failures
│   ├── Test hypotheses against evidence
│   └── Update causal graph with learnings
│
├── Pattern Extractor (ACE Curator)
│   ├── Find recurring successful patterns
│   ├── Extract reusable templates
│   ├── Identify anti-patterns (avoid these)
│   └── Update context library
│
├── Belief Updater (Bayesian)
│   ├── Update causal edge confidences
│   ├── Refine cost estimates
│   ├── Calibrate effort allocations
│   └── Adjust risk assessments
│
└── Policy Optimizer
    ├── A/B test different strategies
    ├── Measure strategy performance
    ├── Update routing policies (which model for which task)
    └── Refine resource allocation policies
```

**Critical Operations**:
1. `record_experience(task, outcome, trace)` → stored
2. `analyze_failure(task, outcome)` → {hypotheses, learnings}
3. `extract_pattern(experiences)` → reusable_template
4. `update_belief(causal_edge, outcome)` → new_confidence
5. `optimize_policy(policy, outcomes)` → improved_policy

**Why This Design?**
- Experience recording enables offline analysis
- Post-mortem finds root causes (causal attribution)
- Pattern extraction enables reuse (efficiency)
- Belief updates make system more accurate over time (convergence)
- Policy optimization makes better decisions (meta-learning)

**Evidence**: ACE framework (verified), Bayesian inference (mathematics), reinforcement learning (ML research)

---

## 4. Component Interactions (Information Flow)

### 4.1 Cold Email Campaign Trace (With All Components)

```
User: "Launch cold email campaign, 1000 customers, $5K, 2 weeks"
│
├─> [KNOWLEDGE SYSTEM] Query: Similar campaigns?
│   └─> Returns: 3 similar campaigns with outcomes
│
├─> [REASONING ENGINE]
│   ├─> Infer: This is a complex, risky task (legal + reputation)
│   └─> Generate: Initial strategy hypothesis
│
├─> [PLANNING SYSTEM]
│   ├─> Decompose: 6 major subtasks, 20 atomic tasks
│   ├─> Build DAG: Extract dependencies
│   ├─> Estimate: $4200 estimated cost, 2 weeks duration, risk=0.8
│   └─> Optimize: Prioritize legal compliance, then infrastructure
│
├─> [REASONING ENGINE - ADVERSARIAL]
│   ├─> Critique: "LinkedIn scraping violates ToS, high IP ban risk"
│   ├─> Suggest: "Use paid API (ZoomInfo) instead"
│   └─> Result: Plan revised
│
├─> [RESOURCE SYSTEM]
│   ├─> Register: linkedin_api (capacity=1), email_account (capacity=1)
│   ├─> Allocate budgets: Subtask 1 gets $1500, Subtask 2 gets $800...
│   └─> Grant access: Check RBAC, provide credentials
│
├─> [EXECUTION SYSTEM]
│   ├─> Spawn: Agent A (define ICP), Agent B (legal check)
│   ├─> Context: Provide snapshot, subscribe to updates
│   └─> Supervise: Start heartbeat monitoring
│
├─> [OBSERVATION SYSTEM] (continuous)
│   ├─> Track: Agent A at 60% complete, healthy
│   ├─> Detect: Agent B discovered GDPR issue
│   └─> Alert: High-impact context change detected
│
├─> [EXECUTION SYSTEM - CONTEXT UPDATE]
│   ├─> Publish: "No EU prospects allowed" (GDPR constraint)
│   ├─> Notify: All agents receive update
│   └─> Adapt: Agent C (build list) filters out EU leads
│
├─> [RESOURCE SYSTEM - CONTENTION]
│   ├─> Agent D requests: linkedin_api
│   ├─> Check: Already allocated to Agent E
│   ├─> Queue: Agent D waits 45 seconds
│   └─> Grant: Agent E releases, Agent D acquires
│
├─> [EXECUTION SYSTEM - HEALTH]
│   ├─> Agent F: No heartbeat for 6 minutes
│   ├─> Terminate: Kill Agent F (stuck in infinite loop)
│   └─> Retry: Re-spawn Agent F with different approach
│
├─> [EXECUTION SYSTEM - ROLLBACK]
│   ├─> Subtask 2.3 (DNS config) fails
│   ├─> Saga rollback: Undo 2.1 (domain registration), 2.2 (email service)
│   └─> Budget refund: $200 returned to pool
│
├─> [PLANNING SYSTEM - RE-PLAN]
│   ├─> Context: Budget now $3800 remaining (spent $1200)
│   ├─> Revise: Reduce prospect list to 700 (from 1000)
│   └─> Update: New plan with adjusted targets
│
... (2 weeks of execution)
│
├─> Campaign completes: 580 customers acquired, $4800 spent
│
└─> [LEARNING SYSTEM]
    ├─> Post-mortem: Expected 1000, got 580 (58% of goal)
    ├─> Analyze: Email open rate was 12% vs 20% expected
    ├─> Hypothesize: "Subject lines not compelling" (confidence: 0.7)
    ├─> Update causal graph: cold_email_subject → open_rate (lower weight)
    └─> Store: Template for future campaigns (with learnings)
```

### 4.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER GOAL                                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   REASONING     │◄──────── KNOWLEDGE (facts)
        │     ENGINE      │
        └────────┬────────┘
                 │ (validated hypothesis)
                 ▼
        ┌─────────────────┐
        │    PLANNING     │◄──────── KNOWLEDGE (historical costs)
        │     SYSTEM      │
        └────────┬────────┘
                 │ (DAG + budget)
                 ▼
        ┌─────────────────┐
        │   EXECUTION     │◄──────── RESOURCE (allocations)
        │     SYSTEM      │
        └────────┬────────┘
                 │ (spawned agents)
                 ▼
        ┌─────────────────┐
        │  OBSERVATION    │──────────► KNOWLEDGE (new facts)
        │     SYSTEM      │──────────► EXECUTION (status updates)
        └────────┬────────┘
                 │ (outcomes + metrics)
                 ▼
        ┌─────────────────┐
        │    LEARNING     │──────────► KNOWLEDGE (updated beliefs)
        │     SYSTEM      │──────────► PLANNING (refined strategies)
        └─────────────────┘
```

---

## 5. Implementation Priority (Minimum Viable System)

### 5.1 Phase 1: Core Infrastructure (MVP - Week 1-3)

**Must Have**:
1. **Knowledge System**: Neo4j with basic schema (entities, relations, temporal)
2. **Planning System**: Task decomposer + dependency graph (DAG)
3. **Execution System**: Agent spawner + supervisor (health monitoring)
4. **Resource System**: Budget manager + basic allocation (semaphores)

**Rationale**: Without these, system cannot execute tasks safely.

### 5.2 Phase 2: Reliability (Week 4-6)

**Should Have**:
5. **Execution System**: Transaction manager (saga pattern for rollback)
6. **Resource System**: Tool access control (RBAC)
7. **Observation System**: Metrics + error detection + quality evaluation

**Rationale**: These prevent catastrophic failures and resource misuse.

### 5.3 Phase 3: Intelligence (Week 7-9)

**Nice to Have**:
8. **Knowledge System**: Causal graph + counterfactual reasoning
9. **Reasoning Engine**: Adversarial validation + hybrid neural-symbolic
10. **Learning System**: Post-mortem analysis + belief updates

**Rationale**: These improve decision quality and enable self-improvement.

### 5.4 Phase 4: Optimization (Week 10-12)

**Advanced Features**:
11. **Observation System**: Distributed tracing (OpenTelemetry)
12. **Planning System**: Multi-objective optimization (Pareto)
13. **Resource System**: Advanced scheduling (fair queueing, preemption)

**Rationale**: These optimize performance but aren't blocking.

---

## 6. Component Interfaces (API Contracts)

### 6.1 Inter-Component Communication

**All components expose standard interfaces**:

```python
class Component(ABC):
    """Base class for all system components."""

    @abstractmethod
    def initialize(self, config: Dict) -> None:
        """Set up component with configuration."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return component health status."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return current metrics."""
        pass

    @abstractmethod
    def shutdown(self, graceful: bool = True) -> None:
        """Clean shutdown."""
        pass
```

### 6.2 Key Interfaces

```python
# Knowledge System
class IKnowledgeSystem(Component):
    def query_facts(self, pattern: str) -> List[Dict]
    def query_causal_path(self, cause: str, effect: str) -> List[CausalEdge]
    def update_fact(self, fact: Fact, confidence: float) -> None
    def counterfactual(self, event_id: str) -> Dict[str, Any]

# Reasoning Engine
class IReasoningEngine(Component):
    def infer(self, premises: List[str], query: str) -> Conclusion
    def test_hypothesis(self, hypothesis: str, evidence: List[str]) -> float
    def validate_plan(self, plan: Plan, context: Context) -> ValidationResult

# Planning System
class IPlanningSystem(Component):
    def decompose(self, goal: str) -> List[Task]
    def build_dag(self, tasks: List[Task]) -> DAG
    def schedule(self, dag: DAG, resources: Dict) -> Schedule
    def estimate_cost(self, plan: Plan) -> CostEstimate

# Execution System
class IExecutionSystem(Component):
    def spawn_agent(self, task: Task, context: Context) -> AgentID
    def supervise_agent(self, agent_id: AgentID) -> AgentStatus
    def synchronize_context(self, update: ContextUpdate) -> None
    def rollback_saga(self, saga_id: str) -> RollbackResult

# Resource System
class IResourceSystem(Component):
    def allocate_resource(self, resource_id: str, agent_id: str) -> AllocationResult
    def release_resource(self, resource_id: str, agent_id: str) -> None
    def check_budget(self, agent_id: str, amount: float) -> bool
    def charge_budget(self, agent_id: str, amount: float) -> None

# Observation System
class IObservationSystem(Component):
    def track_metric(self, name: str, value: float, tags: Dict) -> None
    def get_agent_status(self, agent_id: str) -> AgentStatus
    def evaluate_quality(self, output: str, task: str) -> QualityScores
    def query_trace(self, trace_id: str) -> Trace

# Learning System
class ILearningSystem(Component):
    def record_experience(self, task: Task, outcome: Outcome) -> None
    def analyze_failure(self, task: Task, outcome: Outcome) -> Analysis
    def update_belief(self, edge: CausalEdge, outcome: Outcome) -> None
```

### 6.3 Event Bus (Asynchronous Communication)

```python
class EventBus:
    """
    Central message bus for async communication.
    Decouples components (no direct dependencies).
    """

    def publish(self, event: Event) -> None:
        """Broadcast event to all subscribers."""

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Register callback for event type."""

# Example events
class ContextUpdated(Event):
    context_key: str
    new_value: Any
    impact: str  # "low", "medium", "high", "critical"

class AgentFailed(Event):
    agent_id: str
    task_id: str
    reason: str
    retryable: bool

class BudgetExceeded(Event):
    agent_id: str
    allocated: float
    spent: float

class QualityBelowThreshold(Event):
    task_id: str
    quality_scores: Dict[str, float]
    retry_recommended: bool
```

---

## 7. Why This Architecture (First-Principles Justification)

### 7.1 Separation of Concerns

**Principle**: Each component has ONE primary responsibility

**Why?**
- Changes to knowledge storage don't affect execution logic
- Can replace LLM (reasoning) without touching resource management
- Can optimize scheduler without touching learning system

**Evidence**: SOLID principles (software engineering), Unix philosophy (do one thing well)

### 7.2 Declarative Over Imperative

**Principle**: Components declare "what" not "how"

**Example**:
```python
# BAD (imperative)
def execute_task():
    if resource_available():
        acquire_lock()
        try:
            do_work()
        finally:
            release_lock()

# GOOD (declarative)
@requires_resource("linkedin_api")
def execute_task():
    do_work()  # Resource manager handles locking
```

**Why?**
- Resource management is centralized (no bugs from manual lock management)
- Easier to reason about (intent is clear)
- Composable (can add timeout, retry, etc. without changing task code)

### 7.3 Observable State

**Principle**: All state changes are observable

**Why?**
- Debugging: Can trace why system made decision
- Auditing: Who did what, when, why
- Learning: Can analyze past behavior to improve

**Evidence**: Event sourcing (CQRS pattern), audit logs (compliance), observability (DevOps)

### 7.4 Eventual Consistency

**Principle**: System tolerates temporary inconsistency, converges to consistent state

**Example**: Agent A reads context at t=0, context updated at t=1, Agent A adapts at t=2

**Why?**
- Distributed systems cannot achieve strong consistency without sacrificing availability (CAP theorem)
- Better to make progress with slight staleness than block for perfect consistency

**Evidence**: CAP theorem (provable), Dynamo (Amazon), Cassandra (Netflix)

### 7.5 Fail-Fast + Retry

**Principle**: Detect failures quickly, retry with different approach

**Why?**
- Faster recovery than trying to prevent all failures
- Allows exploration (try multiple strategies)
- Adapts to changing conditions (what worked yesterday may not work today)

**Evidence**: Circuit breaker pattern (Netflix Hysteria), chaos engineering (Simian Army)

---

## 8. What's NOT Included (Explicit Non-Goals)

### 8.1 Perfect Planning

**NOT A GOAL**: Generate optimal plan before execution

**WHY?**
- Optimal planning is NP-hard (exponential time)
- Environment changes during execution (plan becomes stale)
- Better: Good-enough plan + fast adaptation

**INSTEAD**: Generate reasonable plan, monitor, adapt

### 8.2 Human-Level Reasoning

**NOT A GOAL**: Understand intent like humans do

**WHY?**
- Current AI cannot match human intuition
- Trying to achieve this delays deployment indefinitely

**INSTEAD**: Hybrid approach - LLM for pattern matching, graph for logic, human for critical decisions

### 8.3 Zero Failures

**NOT A GOAL**: Prevent all agent failures

**WHY?**
- Impossible in distributed systems (network partitions, hardware failures)
- Cost of prevention > cost of recovery

**INSTEAD**: Detect failures fast, recover gracefully (rollback, retry, escalate)

### 8.4 Complete Knowledge

**NOT A GOAL**: Know everything before acting

**WHY?**
- Perfect information is unattainable (hidden variables, future unknown)
- Waiting for perfect info causes paralysis

**INSTEAD**: Act with best available information, update beliefs as evidence arrives (Bayesian)

---

## 9. Minimal Viable Implementation (Week 1 Prototype)

### 9.1 What Can Be Built in 1 Week?

**Goal**: Prove the architecture works end-to-end

**Scope**:
1. **Knowledge System**: Neo4j with basic schema (no causal graph yet)
2. **Planning System**: Simple decomposer (LLM-based), manual dependency graph
3. **Execution System**: Agent spawner (subprocess), basic supervisor (timeout only)
4. **Resource System**: In-memory budget tracker (no RBAC)

**Test Case**: Simple task (not cold email campaign)
- Goal: "Research top 3 competitors and summarize their pricing"
- Decompose: [scrape_competitor1, scrape_competitor2, scrape_competitor3, summarize]
- Execute: Spawn 3 agents in parallel (scraping), 1 sequential (summarize)
- Observe: Track completion, kill if timeout >5 min
- Learn: Store results in Neo4j

**Success Criteria**:
- All 4 agents complete successfully
- Summary generated from 3 competitor reports
- Total execution time <10 minutes
- No resource conflicts (because tasks are independent)

### 9.2 What's Deferred?

**Week 2+**:
- Causal graph (use simple fact storage first)
- Adversarial validation (LLM-generated plans assumed correct)
- Transaction rollback (no undo)
- Tool access control (all agents can use all tools)
- Context synchronization (agents don't react to updates)
- Learning from failures (no post-mortem)

**Rationale**: Prove core loop works before adding sophistication

---

## 10. Success Metrics (How to Measure System Performance)

### 10.1 Operational Metrics

**Reliability**:
- Agent success rate: >90% (target)
- Rollback rate: <5% (want most plans to work first try)
- Deadlock incidents: 0 (per week)

**Efficiency**:
- Parallel execution ratio: >60% (tasks running concurrently)
- Resource utilization: 70-85% (not idle, not thrashing)
- Budget accuracy: ±10% (estimated vs actual cost)

**Latency**:
- Planning time: <30 seconds (for 20-task plan)
- Context propagation: <5 seconds (update to all agents)
- Error detection: <60 seconds (time to detect stuck agent)

### 10.2 Quality Metrics

**Outcomes**:
- Goal achievement rate: >70% (% of goals successfully completed)
- User satisfaction: >4.0/5.0 (survey after task completion)
- Adaptation effectiveness: >80% (% of context changes handled correctly)

**Learning**:
- Belief accuracy: improving over time (Bayesian calibration)
- Strategy improvement: cost decreasing, quality increasing (trend analysis)
- Error reduction: fewer repeated mistakes (learning curve)

### 10.3 Economic Metrics

**Cost**:
- Cost per task: decreasing over time (learning efficiency)
- Wasted spend: <10% (failed tasks, rolled back operations)
- ROI: benefit > 3x cost (user-defined value vs actual spend)

---

## 11. Conclusion

### 11.1 Essential Components (Minimum Viable System)

From first principles, the system MUST have:

1. **Knowledge System** - Persistent, queryable facts + causality
2. **Reasoning Engine** - Hybrid neural-symbolic decision making
3. **Planning System** - Decomposition + dependency + scheduling
4. **Execution System** - Agent coordination + supervision + context
5. **Resource System** - Allocation + contention + cost tracking
6. **Observation System** - Monitoring + error detection + quality
7. **Learning System** - Experience recording + belief updates + optimization

**Why these 7?** Each addresses a fundamental constraint:
- Knowledge → Causality constraint
- Reasoning → Uncertainty constraint
- Planning → Complexity constraint
- Execution → Reliability constraint
- Resource → Scarcity constraint
- Observation → Adaptation constraint
- Learning → Improvement over time

**Without any one**: System is incomplete and will fail in production.

### 11.2 Key Design Principles

1. **Separation of Concerns**: Each component has one responsibility
2. **Declarative APIs**: Components declare intent, not implementation
3. **Observable State**: All state changes are trackable
4. **Eventual Consistency**: Tolerate temporary inconsistency, converge
5. **Fail-Fast + Retry**: Detect failures quickly, recover gracefully

### 11.3 Implementation Strategy

**Phase 1 (Week 1-3)**: Core infrastructure (Knowledge, Planning, Execution, Resource)
**Phase 2 (Week 4-6)**: Reliability (Rollback, RBAC, Quality evaluation)
**Phase 3 (Week 7-9)**: Intelligence (Causal reasoning, Adversarial validation, Learning)
**Phase 4 (Week 10-12)**: Optimization (Tracing, Multi-objective, Advanced scheduling)

### 11.4 Success Criteria

**System is production-ready when**:
- Agent success rate >90%
- Goal achievement rate >70%
- Cost within ±10% of estimate
- No deadlocks
- Context changes propagate <5 seconds
- Failed tasks rollback successfully

### 11.5 What This Enables

With these 7 components, the system can:
- ✅ Execute complex multi-step tasks autonomously
- ✅ Handle resource contention and budget constraints
- ✅ Detect and recover from agent failures
- ✅ Adapt to changing context mid-execution
- ✅ Learn from experience to improve over time
- ✅ Provide full observability and debuggability
- ✅ Scale from single-machine to distributed deployment

**This is the minimal architecture that can handle the cold email campaign scenario reliably.**

---

**Document Version**: 1.0
**Date**: 2025-11-09
**Status**: First-principles component specification
**Complements**: cold_email_campaign_thought_experiment.md
**Next Step**: Prototype Phase 1 (core infrastructure)
