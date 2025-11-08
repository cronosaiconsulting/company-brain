# Executive Brain: Recommended Architecture Specification

## Document Overview

**Purpose**: Technical architecture specification for the Executive Brain autonomous AI system
**Audience**: Implementation team, technical stakeholders
**Status**: Draft for review
**Version**: 1.0
**Date**: 2025-11-08

---

## 1. Executive Summary

### Vision Restatement

Build an AI system that acts as an **executive cognitive layer** for an organization, capable of:
- Receiving any input (email, chat, docs, APIs, voice)
- Understanding context and intent
- Retrieving relevant information from structured and unstructured sources
- Deciding or recommending optimal actions safely
- Executing through connected tools with full auditability
- Learning and improving its own reasoning context over time

### Recommended Architecture

**Neurosymbolic Hybrid with Recursive Meta-Agent Orchestration**

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTIVE BRAIN SYSTEM                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              INPUT NORMALIZATION LAYER                    │  │
│  │  Email │ Chat │ Voice │ API │ Documents │ Webhooks        │  │
│  └───────────────────────┬──────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────────┐  │
│  │           ORCHESTRATION & REASONING LAYER                 │  │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │ Claude SDK  │  │  ROMA    │  │   LangGraph      │    │  │
│  │  │ (Autonomy)  │◄─┤ (Meta-   │◄─┤  (Workflows)     │    │  │
│  │  │             │  │  Agent)  │  │                  │    │  │
│  │  └──────┬──────┘  └────┬─────┘  └────────┬─────────┘    │  │
│  └─────────┼──────────────┼─────────────────┼──────────────┘  │
│            │              │                 │                  │
│  ┌─────────▼──────────────▼─────────────────▼──────────────┐  │
│  │              MEMORY & KNOWLEDGE LAYER                     │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │         Neo4j GraphRAG + Graphiti                  │  │  │
│  │  │  ┌──────────────┐  ┌───────────┐  ┌─────────────┐ │  │  │
│  │  │  │ Knowledge    │  │ Episodic  │  │ Procedural  │ │  │  │
│  │  │  │ Graph        │  │ Memory    │  │ Memory      │ │  │  │
│  │  │  │ (Entities,   │  │(Temporal  │  │ (ACE        │ │  │  │
│  │  │  │ Relations)   │  │ Events)   │  │ Contexts)   │ │  │  │
│  │  │  └──────────────┘  └───────────┘  └─────────────┘ │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │              GOVERNANCE & POLICY LAYER                     │  │
│  │  ┌──────────────────┐        ┌──────────────────────┐    │  │
│  │  │ OPA              │        │ Cedar                │    │  │
│  │  │ (Infrastructure) │        │ (Business Logic)     │    │  │
│  │  │ - Resource auth  │        │ - Approval rules     │    │  │
│  │  │ - Rate limits    │        │ - Risk thresholds    │    │  │
│  │  └──────────────────┘        └──────────────────────┘    │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │              TOOL EXECUTION LAYER (MCP)                    │  │
│  │  Email │ Calendar │ CRM │ ERP │ Databases │ Custom APIs   │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │           EVALUATION & FEEDBACK LAYER                      │  │
│  │  ┌────────────────┐    ┌──────────────┐  ┌─────────────┐ │  │
│  │  │ Ragas          │───►│ ACE Reflector│─►│ ACE Curator │ │  │
│  │  │ (Faithfulness, │    │ (Extract     │  │ (Update     │ │  │
│  │  │  Relevance)    │    │  insights)   │  │  contexts)  │ │  │
│  │  └────────────────┘    └──────────────┘  └─────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              AUDIT & OBSERVABILITY LAYER                  │  │
│  │  Decision Logs │ Evidence Chains │ Metrics │ Alerts       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture

| Requirement | Architectural Solution | Evidence |
|-------------|------------------------|----------|
| **Multimodal Input** | Unified normalization layer → structured events | Industry standard (Zapier, n8n patterns) |
| **Complex Reasoning** | ROMA recursive meta-agent decomposition | 81.7% FRAMES benchmark, 4x Gemini |
| **Long-term Memory** | Neo4j GraphRAG with temporal awareness (Graphiti) | Graph enables relationship reasoning |
| **Explainability** | Cypher query traces + ROMA stage tracing + LangGraph DAG | Full decision audit trails |
| **Safety** | Dual-layer governance (OPA + Cedar) + risk scoring | Defense in depth |
| **Self-Improvement** | ACE framework (Generator → Reflector → Curator) | +10.6% agent performance gain |
| **Interoperability** | Model Context Protocol (MCP) for tools | 1000+ existing integrations |
| **Auditability** | Immutable event log in Neo4j + Prometheus metrics | Compliance-ready |
| **Scalability** | Kubernetes orchestration + managed Neo4j AuraDB | Enterprise-proven |

---

## 2. Layer-by-Layer Specification

### 2.1 Input Normalization Layer

**Purpose**: Convert heterogeneous inputs into normalized event structures

**Technologies**:
- **API Gateway**: Kong or Envoy (with rate limiting)
- **Message Queue**: RabbitMQ or Redis Streams (for async processing)
- **Input Adapters**: Custom microservices per input type

**Data Flow**:
1. Raw input arrives (email via IMAP, chat via Slack webhook, API via REST, etc.)
2. Input adapter extracts metadata and content
3. Structured event emitted to queue with schema:

```json
{
  "event_id": "uuid",
  "timestamp": "ISO8601",
  "source": "email|chat|api|voice|webhook",
  "user": {
    "id": "user_id",
    "name": "User Name",
    "roles": ["role1", "role2"]
  },
  "content": {
    "raw": "original content",
    "structured": {
      "intent": "detected intent",
      "entities": ["entity1", "entity2"],
      "urgency": "low|medium|high|critical"
    }
  },
  "context": {
    "thread_id": "for conversations",
    "references": ["related doc IDs"],
    "metadata": {}
  }
}
```

**Key Design Decisions**:
- ✅ **Schema Validation**: Use JSON Schema to validate all events
- ✅ **Idempotency**: Event IDs prevent duplicate processing
- ✅ **Dead Letter Queue**: Failed normalizations go to DLQ for manual review

---

### 2.2 Orchestration & Reasoning Layer

**Purpose**: Hierarchical task decomposition, planning, execution, and recursion

#### 2.2.1 Claude Agent SDK (Base Autonomy)

**Role**: Primary agent runtime with tool access and iteration capabilities

**Capabilities**:
- File system operations (read, write, edit)
- Bash command execution
- Web browsing and search
- Checkpoints for long-running tasks
- Subagents for parallel workflows
- Hooks for event-driven automation

**When to Use**:
- Straightforward tasks with clear tool sequences
- File/code manipulation
- Information gathering and synthesis
- Tasks requiring web search

**Configuration**:
```python
from claude_agent_sdk import Agent

executive_agent = Agent(
    model="claude-sonnet-4-5",
    tools=["bash", "file_ops", "web_search", "mcp_tools"],
    checkpoint_interval=300,  # 5 minutes
    max_iterations=50,
    hooks={
        "pre_tool": "governance_check",
        "post_tool": "audit_log"
    }
)
```

#### 2.2.2 ROMA (Recursive Meta-Agent)

**Role**: Complex reasoning requiring hierarchical decomposition

**Architecture**:
- **Atomizer**: Determines if task is atomic or needs decomposition
- **Planner**: Breaks complex tasks into subtasks (recursive)
- **Executors**: Can be LLMs, APIs, or Claude Agent SDK instances
- **Aggregator**: Synthesizes child results into parent answer

**When to Use**:
- Multi-step reasoning (e.g., "Analyze Q3 financials and recommend cost optimizations")
- Research tasks requiring synthesis from multiple sources
- Decision-making with dependencies between subtasks

**Integration Pattern**:
```python
from roma import MetaAgent

# Claude SDK as ROMA executor
class ClaudeExecutor:
    def execute(self, task):
        return executive_agent.run(task)

meta_agent = MetaAgent(
    atomizer=GPT4oMini,  # Cheap model for decomposition check
    planner=ClaudeSonnet4_5,
    executors=[ClaudeExecutor(), APIExecutor(), ToolExecutor()],
    aggregator=ClaudeSonnet4_5,
    max_depth=5  # Prevent infinite recursion
)
```

**Stage Tracing**:
- Every stage (atomize, plan, execute, aggregate) logged to Neo4j
- Enables debugging: "Why did the agent choose this path?"

#### 2.2.3 LangGraph (Deterministic Workflows)

**Role**: Pre-defined, auditable workflows for compliance-critical processes

**When to Use**:
- Financial approval workflows (multi-step, conditional)
- Compliance processes (GDPR data requests, SOX audits)
- Scheduled recurring tasks (daily reports, weekly summaries)

**Example Workflow**: Purchase Approval
```python
from langgraph.graph import StateGraph

workflow = StateGraph()
workflow.add_node("extract_request", extract_purchase_info)
workflow.add_node("check_budget", query_budget_system)
workflow.add_node("risk_assessment", calculate_risk_score)
workflow.add_node("human_approval", send_to_approver)
workflow.add_node("execute_purchase", call_procurement_api)

workflow.add_conditional_edges(
    "risk_assessment",
    lambda state: "high" if state["risk_score"] > 0.7 else "low",
    {
        "high": "human_approval",
        "low": "execute_purchase"
    }
)
```

**Visualization**: LangGraph generates Mermaid diagrams of workflow DAGs for documentation

#### 2.2.4 Orchestration Decision Tree

```
Input Event
    │
    ▼
Is this a known workflow?
    ├─ YES ──► LangGraph (deterministic path)
    │
    └─ NO ──► Is this complex multi-step reasoning?
              ├─ YES ──► ROMA (recursive decomposition)
              │
              └─ NO ──► Claude SDK (direct execution)
```

---

### 2.3 Memory & Knowledge Layer

**Purpose**: Unified, queryable, temporal knowledge substrate

#### 2.3.1 Neo4j GraphRAG Core Architecture

**Database**: Neo4j AuraDB (managed cloud)

**Schema Design**:

**Nodes**:
```cypher
// Entities
(:Person {id, name, email, department, joined_date})
(:Project {id, name, status, start_date, end_date, budget})
(:Document {id, title, url, created_date, embedding_vector})
(:Policy {id, name, version, effective_date, text, embedding_vector})
(:Decision {id, timestamp, agent_id, outcome, risk_score})

// Temporal tracking
(:Event {id, timestamp, type, data})
(:Memory {id, timestamp, summary, importance_score, embedding_vector})
```

**Relationships**:
```cypher
// Static relationships
(:Person)-[:WORKS_ON {role, since}]->(:Project)
(:Person)-[:REPORTS_TO {since}]->(:Person)
(:Document)-[:ABOUT]->(:Project)

// Temporal relationships (Graphiti pattern)
(:Event)-[:HAPPENED_AT {timestamp}]->(:Entity)
(:Decision)-[:REFERENCES {weight}]->(:Document)
(:Memory)-[:RELATES_TO {strength, last_accessed}]->(:Entity)

// Episodic memory chains
(:Memory)-[:PRECEDED_BY {duration}]->(:Memory)
```

**Indexes**:
```cypher
// Vector similarity search
CREATE VECTOR INDEX document_embeddings FOR (d:Document) ON d.embedding_vector
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

// Full-text search
CREATE FULLTEXT INDEX policy_text FOR (p:Policy) ON EACH [p.text, p.name]

// Temporal queries
CREATE INDEX event_timestamp FOR (e:Event) ON e.timestamp
```

#### 2.3.2 Graphiti Integration (Temporal Memory)

**Purpose**: Incrementally update knowledge graph from episodic experiences

**Process**:
1. Agent completes task → generates structured experience
2. Graphiti extracts entities, relationships, and updates graph in real-time
3. Temporal edges capture "when" information was learned
4. Importance scoring for memory consolidation (forget low-value memories)

**Example**:
```
User: "We're pivoting the Phoenix project to focus on enterprise customers."

Graphiti creates:
- (:Event {type: "project_pivot", timestamp: now()})
- Updates: (:Project {name: "Phoenix"})-[:TARGETS {since: now()}]->(:Segment {name: "Enterprise"})
- Archives old relationship: (:Project)-[:TARGETS {until: now()}]->(:Segment {name: "SMB"})
```

**Temporal Queries**:
```cypher
// "What was the Phoenix project focused on last quarter?"
MATCH (p:Project {name: "Phoenix"})-[r:TARGETS]->(s:Segment)
WHERE r.since <= date('2024-10-01') AND (r.until IS NULL OR r.until >= date('2024-10-01'))
RETURN s.name
```

#### 2.3.3 Memory Hierarchy

Inspired by ACT-R cognitive architecture:

| Memory Type | Storage | Retrieval | Consolidation |
|-------------|---------|-----------|---------------|
| **Working Memory** | Redis (ephemeral, per-session) | Direct key access | Discarded after session |
| **Episodic Memory** | Neo4j (temporal events) | Recency + importance | Monthly archival of low-importance |
| **Semantic Memory** | Neo4j (knowledge graph) | Vector + graph traversal | Continuous refinement via ACE |
| **Procedural Memory** | ACE context library | Similarity search → prompt injection | Curated after reflection |

#### 2.3.4 Retrieval Strategy

**Hybrid Retrieval Pipeline**:
```python
def retrieve_context(query, top_k=10):
    # 1. Vector similarity (fast, semantic)
    vector_results = neo4j.vector_search(
        query_embedding=embed(query),
        top_k=top_k * 2
    )

    # 2. Graph expansion (relationships)
    graph_results = []
    for node in vector_results:
        neighbors = neo4j.cypher(f"""
            MATCH (n)-[r]-(m)
            WHERE id(n) = {node.id}
            RETURN m, r, type(r)
            LIMIT 5
        """)
        graph_results.extend(neighbors)

    # 3. Temporal filtering (prefer recent)
    temporal_boosted = boost_by_recency(
        graph_results,
        decay_factor=0.95,
        window_days=90
    )

    # 4. Importance scoring (consolidation)
    importance_filtered = filter_by_importance(
        temporal_boosted,
        min_score=0.3
    )

    # 5. Re-rank by combined score
    final_results = rerank(
        importance_filtered,
        query=query,
        top_k=top_k
    )

    return final_results
```

---

### 2.4 Governance & Policy Layer

**Purpose**: Multi-layer safety and compliance enforcement

#### 2.4.1 OPA (Infrastructure Policies)

**Deployment**: Kubernetes DaemonSet (sidecar to every agent pod)

**Policies**:
```rego
package executive_brain.infrastructure

# Rate limiting
default allow_request = false
allow_request {
    count(user_requests_last_hour[input.user_id]) < 1000
}

# Resource authorization
allow_tool_access {
    input.tool == "database_write"
    input.user.roles[_] == "admin"
}

# Cost controls
allow_llm_call {
    input.model == "claude-haiku"  # Always allowed
}
allow_llm_call {
    input.model == "claude-sonnet-4-5"
    monthly_cost < 10000  # $10k budget
}
```

**Enforcement Point**: Pre-hook in Claude SDK
```python
@agent.pre_tool_hook
def check_opa_policy(tool_name, args, user):
    decision = opa_client.query(
        policy="executive_brain.infrastructure.allow_tool_access",
        input={"tool": tool_name, "user": user}
    )
    if not decision["allow"]:
        raise PolicyViolation(decision["reason"])
```

#### 2.4.2 Cedar (Business Logic Policies)

**Deployment**: Centralized policy service (K8s Deployment)

**Schema**:
```cedar
entity User {
    department: String,
    role: String,
    clearance_level: Long
}

entity Action {
    type: String,
    risk_score: Decimal,
    financial_impact: Decimal
}

entity Resource {
    classification: String,
    owner: User
}
```

**Policies**:
```cedar
// Financial approvals
permit(
    principal in Role::"Finance",
    action == Action::"approve_expense",
    resource
) when {
    resource.amount < 10000
};

permit(
    principal in Role::"CFO",
    action == Action::"approve_expense",
    resource
) when {
    resource.amount < 100000
};

// Risk-based gating
forbid(
    principal,
    action,
    resource
) when {
    action.risk_score > 0.8
} unless {
    context.has_human_approval == true
};

// Data access controls (GDPR)
permit(
    principal,
    action == Action::"read",
    resource
) when {
    resource.classification == "public"
};

permit(
    principal,
    action == Action::"read",
    resource
) when {
    resource.classification == "internal" &&
    principal.department == resource.owner.department
};
```

**Enforcement**:
```python
def execute_action(action, user, context):
    # Calculate risk score (from ML model or heuristics)
    risk_score = assess_risk(action)

    # Cedar policy check
    decision = cedar_client.is_authorized(
        principal=user,
        action={"type": action.type, "risk_score": risk_score},
        resource=action.target,
        context=context
    )

    if decision.decision == "Deny":
        if risk_score > 0.8:
            # High risk: require human approval
            return request_human_approval(action, decision.reason)
        else:
            raise PolicyViolation(decision.reason)

    # Log decision for audit
    audit_log.record(user, action, decision, risk_score)

    # Execute
    return perform_action(action)
```

#### 2.4.3 Risk Scoring Model

**Inputs**:
- Action type (database_write > file_read)
- Resource sensitivity (PII, financial data)
- User context (normal hours vs 3am)
- Historical anomaly detection (unusual for this user?)
- Financial impact estimate

**Risk Calculation**:
```python
def assess_risk(action, user, context):
    base_risk = ACTION_RISK_SCORES.get(action.type, 0.5)

    # Sensitivity multiplier
    if action.resource.has_pii:
        base_risk *= 1.5
    if action.resource.classification == "confidential":
        base_risk *= 2.0

    # Temporal anomaly
    if context.hour < 6 or context.hour > 22:
        base_risk *= 1.3

    # User behavior baseline
    user_baseline = ml_model.predict_baseline(user)
    anomaly_score = abs(action.features - user_baseline)

    # Financial impact
    if action.financial_impact > 10000:
        base_risk *= 1.2

    final_risk = min(base_risk + anomaly_score, 1.0)

    return final_risk
```

**Thresholds**:
- 0.0 - 0.3: **Low** → Auto-approve
- 0.3 - 0.6: **Medium** → Log and notify
- 0.6 - 0.8: **High** → Require secondary check (OPA + Cedar)
- 0.8 - 1.0: **Critical** → Mandatory human approval

---

### 2.5 Tool Execution Layer (MCP)

**Purpose**: Standardized, auditable interface to enterprise systems and external tools

#### 2.5.1 MCP Architecture

**MCP Server Registry**:
```json
{
  "servers": [
    {
      "name": "email",
      "url": "mcp://internal/email",
      "capabilities": ["read", "send", "search"],
      "authentication": "oauth2",
      "rate_limit": "1000/hour"
    },
    {
      "name": "calendar",
      "url": "mcp://internal/calendar",
      "capabilities": ["read", "create_event", "update_event"],
      "authentication": "oauth2"
    },
    {
      "name": "crm_salesforce",
      "url": "mcp://internal/crm",
      "capabilities": ["read_accounts", "update_opportunity"],
      "authentication": "api_key",
      "sensitive": true
    },
    {
      "name": "database_postgres",
      "url": "mcp://internal/postgres",
      "capabilities": ["query", "write"],
      "authentication": "db_credentials",
      "audit_required": true
    }
  ]
}
```

#### 2.5.2 Custom MCP Server Template

For enterprise-specific tools:

```python
from mcp import Server, Tool

class CustomERPServer(Server):
    @Tool(
        name="get_budget_status",
        description="Retrieve current budget status for a department",
        parameters={
            "department": "string",
            "fiscal_year": "integer"
        }
    )
    async def get_budget_status(self, department, fiscal_year):
        # Authenticate
        self.validate_auth(self.request.user)

        # Query ERP system
        result = await erp_client.query(
            f"SELECT budget, spent FROM budgets WHERE dept='{department}' AND fy={fiscal_year}"
        )

        # Audit log
        await self.audit_log(
            user=self.request.user,
            tool="get_budget_status",
            params={"department": department, "fiscal_year": fiscal_year},
            result_summary=f"Budget: {result['budget']}, Spent: {result['spent']}"
        )

        return result
```

#### 2.5.3 MCP Security Layer

**Challenges** (from research):
- Prompt injection via tool responses
- Tool permission escalation (combining tools to exfiltrate data)
- Lookalike tools (malicious server mimicking trusted one)

**Mitigations**:

1. **Server Allowlist**:
```python
APPROVED_MCP_SERVERS = {
    "email": {
        "url": "mcp://internal/email",
        "fingerprint": "sha256:abcd1234...",  # TLS cert fingerprint
        "max_risk_score": 0.5
    }
}

def validate_mcp_server(server_url):
    if server_url not in APPROVED_MCP_SERVERS:
        raise UntrustedServerError()

    # Verify TLS certificate fingerprint
    cert = get_server_cert(server_url)
    if cert.fingerprint != APPROVED_MCP_SERVERS[server_url]["fingerprint"]:
        raise CertificateMismatchError()
```

2. **Tool Composition Analysis**:
```python
def detect_exfiltration_risk(tool_sequence):
    """
    Detect if combining tools could leak data.
    Example: read_file() → send_email() to external address
    """
    risk_patterns = [
        (["read_file", "database_query"], ["send_email", "http_post"]),
        (["list_directory"], ["send_email"])
    ]

    for read_tools, write_tools in risk_patterns:
        if any(t in tool_sequence for t in read_tools) and \
           any(t in tool_sequence for t in write_tools):
            return True, "Potential data exfiltration pattern"

    return False, None
```

3. **Response Sanitization**:
```python
def sanitize_tool_response(response, tool_name):
    """Prevent prompt injection via tool responses"""
    # Remove potential instruction markers
    sanitized = response.replace("</s>", "").replace("<|endoftext|>", "")

    # Escape markdown that could hide instructions
    sanitized = escape_markdown(sanitized)

    # Wrap in clear delimiters
    return f"[TOOL_OUTPUT:{tool_name}]\n{sanitized}\n[/TOOL_OUTPUT]"
```

---

### 2.6 Evaluation & Feedback Layer

**Purpose**: Continuous quality assurance and self-improvement

#### 2.6.1 Ragas Evaluation Pipeline

**Trigger**: After every agent response

**Process**:
```python
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from ragas import evaluate

def evaluate_response(query, response, retrieved_contexts, ground_truth=None):
    dataset = {
        "question": [query],
        "answer": [response],
        "contexts": [retrieved_contexts],
        "ground_truth": [ground_truth] if ground_truth else None
    }

    metrics = [
        faithfulness,        # No hallucinations
        answer_relevance,    # Answers the question
        context_precision,   # Retrieved right contexts
        context_recall       # Retrieved all needed contexts
    ]

    results = evaluate(dataset, metrics=metrics)

    # Store in Neo4j for analysis
    store_evaluation(
        query=query,
        response=response,
        scores=results,
        timestamp=datetime.now()
    )

    return results
```

**Quality Gates**:
```python
def apply_quality_gates(evaluation_results):
    if evaluation_results["faithfulness"] < 0.8:
        # Hallucination detected
        alert_monitoring("Hallucination detected", evaluation_results)
        return "REJECT", "Faithfulness score too low"

    if evaluation_results["answer_relevance"] < 0.7:
        # Off-topic response
        return "RETRY", "Answer not relevant to query"

    if evaluation_results["context_recall"] < 0.6:
        # Missing information
        return "AUGMENT", "Insufficient context retrieved"

    return "ACCEPT", "Quality gates passed"
```

#### 2.6.2 ACE Self-Improvement Loop

**Components**:

1. **Generator**: Claude Agent SDK (produces reasoning trajectories)
2. **Reflector**: Analyzes outcomes, extracts insights
3. **Curator**: Updates procedural memory (contexts/prompts)

**Reflection Trigger**:
- Every N tasks (e.g., N=10)
- When Ragas scores drop below threshold
- Manual trigger by admin
- Scheduled (daily/weekly)

**Reflection Process**:
```python
class ACEReflector:
    def reflect(self, recent_tasks, evaluations):
        """
        Analyze recent tasks and extract insights.
        """
        # Group by success/failure
        successes = [t for t, e in zip(recent_tasks, evaluations)
                     if e["faithfulness"] > 0.9 and e["answer_relevance"] > 0.9]
        failures = [t for t, e in zip(recent_tasks, evaluations)
                    if e["faithfulness"] < 0.7 or e["answer_relevance"] < 0.7]

        # Extract patterns
        insights = {
            "success_patterns": self.extract_patterns(successes),
            "failure_patterns": self.extract_patterns(failures),
            "suggested_improvements": self.generate_improvements(failures)
        }

        return insights

    def extract_patterns(self, tasks):
        """Use LLM to identify common patterns"""
        prompt = f"""
        Analyze these task executions and identify common patterns:

        {json.dumps(tasks, indent=2)}

        Return structured patterns in JSON format.
        """

        patterns = llm.generate(prompt)
        return json.loads(patterns)
```

**Curation Process**:
```python
class ACECurator:
    def curate_context(self, insights, current_context):
        """
        Update procedural memory based on insights.
        Version control for contexts.
        """
        # Generate proposed update
        prompt = f"""
        Current context (v{current_context.version}):
        {current_context.text}

        Recent insights:
        {json.dumps(insights, indent=2)}

        Propose an improved context that:
        1. Incorporates successful patterns
        2. Avoids failure patterns
        3. Maintains all critical domain knowledge
        4. Is concise yet comprehensive

        Return ONLY the updated context text.
        """

        proposed_context = llm.generate(prompt)

        # A/B test new context
        ab_test_result = self.ab_test(
            control=current_context,
            treatment=proposed_context,
            test_tasks=sample_tasks(n=20)
        )

        if ab_test_result["treatment_better"]:
            # Store new version in Neo4j
            new_version = current_context.version + 1
            neo4j.create_node(
                "ProceduralMemory",
                {
                    "domain": current_context.domain,
                    "version": new_version,
                    "text": proposed_context,
                    "created": datetime.now(),
                    "predecessor": current_context.id,
                    "improvement": ab_test_result["delta"]
                }
            )
            return new_version
        else:
            logger.info("Proposed context did not improve performance")
            return current_context.version
```

**Context Versioning in Neo4j**:
```cypher
// Procedural memory evolution
(:ProceduralMemory {domain: "financial_analysis", version: 1})
    -[:EVOLVED_TO {delta: +0.12, date: "2025-01-15"}]->
(:ProceduralMemory {domain: "financial_analysis", version: 2})
    -[:EVOLVED_TO {delta: +0.08, date: "2025-02-10"}]->
(:ProceduralMemory {domain: "financial_analysis", version: 3})
```

---

### 2.7 Audit & Observability Layer

**Purpose**: Full traceability of decisions, metrics, and compliance

#### 2.7.1 Decision Audit Trail

**Storage**: Neo4j (immutable event log)

**Schema**:
```cypher
(:Decision {
    id: uuid,
    timestamp: datetime,
    agent_id: string,
    user_id: string,
    query: string,
    response: string,
    risk_score: float
})
    -[:RETRIEVED]->(:Context {source: string, relevance: float})
    -[:INVOKED]->(:Tool {name: string, params: json, result: json})
    -[:GOVERNED_BY]->(:Policy {name: string, version: int, decision: "allow|deny"})
    -[:EVALUATED_AS]->(:Evaluation {faithfulness: float, relevance: float})
    -[:APPROVED_BY]->(:Person) [if human approval required]
```

**Query Examples**:
```cypher
// "Why did the agent approve this expense?"
MATCH (d:Decision {id: $decision_id})
      -[:RETRIEVED]->(ctx:Context)
      -[:GOVERNED_BY]->(pol:Policy)
RETURN d, collect(ctx), collect(pol)

// "Show all high-risk decisions last week"
MATCH (d:Decision)
WHERE d.timestamp > datetime() - duration('P7D')
  AND d.risk_score > 0.8
RETURN d
ORDER BY d.risk_score DESC

// "Which policies are most frequently blocking actions?"
MATCH (pol:Policy)<-[g:GOVERNED_BY]-(d:Decision)
WHERE g.decision = "deny"
RETURN pol.name, count(*) as blocks
ORDER BY blocks DESC
LIMIT 10
```

#### 2.7.2 Metrics & Monitoring

**Prometheus Metrics**:
```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_total = Counter(
    "executive_brain_requests_total",
    "Total requests by source and outcome",
    ["source", "outcome"]
)

# Latency
decision_latency = Histogram(
    "executive_brain_decision_latency_seconds",
    "Decision latency by complexity",
    ["complexity"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Quality scores
quality_score = Gauge(
    "executive_brain_quality_score",
    "Average quality scores by metric",
    ["metric"]
)

# Policy blocks
policy_blocks = Counter(
    "executive_brain_policy_blocks_total",
    "Policy blocks by policy name",
    ["policy"]
)

# Cost tracking
llm_cost = Counter(
    "executive_brain_llm_cost_dollars",
    "LLM API cost by model",
    ["model"]
)
```

**Grafana Dashboards**:
1. **Executive Overview**: Request volume, success rate, average quality scores
2. **Performance**: Latency percentiles, throughput, error rates
3. **Quality**: Ragas metrics over time, quality gate rejections
4. **Governance**: Policy block rates, risk score distribution, human approval frequency
5. **Cost**: LLM spend by model, MCP tool usage, infrastructure costs

#### 2.7.3 Alerting

**Critical Alerts** (PagerDuty):
- Faithfulness score < 0.7 for 5 consecutive responses
- Policy violation attempts spike (>10x baseline)
- Neo4j database unreachable
- LLM API errors > 5% for 5 minutes

**Warning Alerts** (Slack):
- Average quality scores declining week-over-week
- High-risk actions requiring human approval backlog > 10
- MCP server response time > 5s
- Daily LLM cost > $1000

---

## 3. Data Flow Examples

### 3.1 Example: Email Request Processing

**Scenario**: User emails "What's the status of the Phoenix project?"

**Step-by-Step Flow**:

1. **Input Normalization**:
   - IMAP adapter receives email
   - Extracts: sender, subject, body, thread_id
   - Emits event to RabbitMQ

2. **Orchestration Decision**:
   - Event consumed by orchestrator
   - Query: "What's the status of the Phoenix project?"
   - Complexity assessment: **Simple** (single entity lookup)
   - Route to: **Claude SDK**

3. **Memory Retrieval**:
   ```python
   contexts = retrieve_context("Phoenix project status")
   # Returns:
   # - (:Project {name: "Phoenix", status: "In Progress", deadline: "2025-12-31"})
   # - Recent (:Event) nodes related to Phoenix
   # - Related (:Document) with vector similarity
   ```

4. **Governance Check**:
   - OPA: User has permission to read project data? ✅
   - Cedar: Project classification allows this user? ✅
   - Risk score: 0.2 (low) → Auto-approve

5. **Agent Processing**:
   ```python
   agent_response = executive_agent.run(
       query="What's the status of the Phoenix project?",
       contexts=contexts,
       user=user
   )
   # Response: "The Phoenix project is currently In Progress, targeting enterprise
   # customers. The deadline is December 31, 2025. Last update: team completed user
   # research phase (Nov 1, 2025)."
   ```

6. **Tool Execution** (if needed):
   - If contexts insufficient, agent might call MCP tools:
     - `project_management.get_tasks(project="Phoenix")`
     - `calendar.get_meetings(project="Phoenix", last_week=true)`

7. **Evaluation**:
   ```python
   eval_results = evaluate_response(
       query="What's the status of the Phoenix project?",
       response=agent_response,
       contexts=contexts
   )
   # Scores: {faithfulness: 0.95, answer_relevance: 0.92}
   ```

8. **Audit Logging**:
   ```cypher
   CREATE (d:Decision {
       id: uuid(),
       timestamp: datetime(),
       query: "What's the status of the Phoenix project?",
       response: "...",
       risk_score: 0.2
   })
   CREATE (d)-[:RETRIEVED]->(ctx1:Context {source: "Neo4j", relevance: 0.95})
   CREATE (d)-[:EVALUATED_AS]->(e:Evaluation {faithfulness: 0.95, relevance: 0.92})
   ```

9. **Response Delivery**:
   - Email sent to user with answer
   - Decision logged for future context ("User asked about Phoenix at 2pm on Nov 8")

---

### 3.2 Example: Complex Financial Analysis

**Scenario**: CFO asks "Analyze Q3 spending and recommend where we can cut 10% from Q4 budget without impacting revenue."

**Step-by-Step Flow**:

1. **Input Normalization**: Same as above (chat interface)

2. **Orchestration Decision**:
   - Query complexity: **High** (multi-step reasoning, synthesis, recommendations)
   - Route to: **ROMA (Recursive Meta-Agent)**

3. **ROMA Decomposition**:

   **Atomizer**: "Is this atomic?" → **No** (requires multiple steps)

   **Planner** breaks into subtasks:
   ```json
   {
     "task": "Q3 spending analysis and Q4 budget cut recommendations",
     "subtasks": [
       {
         "id": "1",
         "task": "Retrieve Q3 spending by department",
         "executor": "database_query"
       },
       {
         "id": "2",
         "task": "Analyze spending vs budget vs revenue impact",
         "executor": "claude_agent"
       },
       {
         "id": "3",
         "task": "Identify low-impact cost reduction opportunities",
         "executor": "claude_agent",
         "depends_on": ["2"]
       },
       {
         "id": "4",
         "task": "Simulate Q4 scenarios with 10% cuts in different areas",
         "executor": "financial_model_api",
         "depends_on": ["3"]
       },
       {
         "id": "5",
         "task": "Synthesize recommendations with risk assessment",
         "executor": "claude_agent",
         "depends_on": ["4"]
       }
     ]
   }
   ```

4. **Parallel Execution** (where possible):
   - Task 1 executes via MCP (database query tool)
   - Tasks 2, 3 wait for Task 1
   - Task 4 waits for Task 3
   - Task 5 aggregates all

5. **Governance Checkpoints**:
   - Task 1: Cedar check → CFO can access financial data ✅
   - Task 4: Risk score 0.75 (high impact) → Requires secondary approval
   - Human approval requested from Board Finance Committee
   - Approved ✅

6. **Memory Storage**:
   - Each subtask result stored as (:Memory) node
   - Linked to (:Decision) with temporal edges
   - Future queries can reference: "Last time we did budget analysis, we found..."

7. **ACE Reflection** (post-task):
   - Reflector analyzes: "This task took 12 minutes, used 4 LLM calls, cost $2.50"
   - Insight: "Financial model API was bottleneck (8 min). Consider caching common scenarios."
   - Curator updates procedural memory: "For budget analysis tasks, pre-fetch financial model baseline."

8. **Audit Trail**:
   ```cypher
   (:Decision {id: "...", query: "Q3 analysis..."})
       -[:DECOMPOSED_INTO]->(:Subtask {id: "1", ...})
       -[:DECOMPOSED_INTO]->(:Subtask {id: "2", ...})
       ...
       -[:APPROVED_BY]->(:Person {name: "Board Finance Committee"})
       -[:EVALUATED_AS]->(:Evaluation {...})
   ```

---

## 4. Deployment Architecture

### 4.1 Kubernetes Cluster Layout

**Namespaces**:
- `executive-brain-core`: Orchestration agents
- `executive-brain-memory`: Neo4j, Redis
- `executive-brain-policy`: OPA, Cedar services
- `executive-brain-mcp`: MCP server deployments
- `executive-brain-obs`: Prometheus, Grafana, Loki

**Core Services**:

```yaml
# Claude Agent SDK deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-agent
  namespace: executive-brain-core
spec:
  replicas: 5  # Horizontal scaling
  selector:
    matchLabels:
      app: claude-agent
  template:
    spec:
      containers:
      - name: claude-agent
        image: company/claude-agent:v1.0
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: claude-secrets
              key: api_key
        - name: NEO4J_URI
          value: "bolt://neo4j-auradb.external:7687"
        - name: OPA_URL
          value: "http://opa.executive-brain-policy:8181"
        - name: CEDAR_URL
          value: "http://cedar.executive-brain-policy:8080"
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

**OPA DaemonSet** (runs on every node):
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: opa
  namespace: executive-brain-policy
spec:
  selector:
    matchLabels:
      app: opa
  template:
    spec:
      containers:
      - name: opa
        image: openpolicyagent/opa:latest
        args:
        - "run"
        - "--server"
        - "--config-file=/config/opa-config.yaml"
        volumeMounts:
        - name: opa-policies
          mountPath: /policies
        ports:
        - containerPort: 8181
      volumes:
      - name: opa-policies
        configMap:
          name: opa-policies
```

### 4.2 Neo4j AuraDB (Managed Cloud)

**Tier**: Professional (managed by Neo4j)

**Configuration**:
- **Region**: Same as K8s cluster (minimize latency)
- **Instance Size**: 8 CPU, 32GB RAM (start; scale up as needed)
- **Storage**: 500GB SSD (auto-scaling enabled)
- **Backups**: Daily automated, 30-day retention
- **High Availability**: Multi-AZ deployment

**Connection**:
- Private VPC peering between K8s cluster and AuraDB
- TLS-encrypted connections (bolt+s://)
- Connection pooling in application layer

**Access Control**:
- Service account for agents (read/write to specific node labels)
- Admin account for schema migrations (human-only)
- Read-only account for analytics dashboards

### 4.3 Scalability Strategy

**Horizontal Scaling**:
- **Claude Agent SDK**: K8s HPA based on CPU + request queue depth
- **MCP Servers**: Per-tool scaling (e.g., scale email server independently)
- **Policy Services**: Replicated deployments (stateless)

**Vertical Scaling**:
- **Neo4j**: AuraDB managed scaling (can upgrade instance size with minimal downtime)
- **Redis**: Cluster mode with sharding

**Caching**:
- **Embedding Cache**: Cache query embeddings in Redis (TTL: 1 hour)
- **Context Cache**: Frequently accessed graph patterns cached (TTL: 15 min)
- **LLM Response Cache**: Hash(query + contexts) → response (TTL: 24 hours, only for deterministic queries)

**Queue Management**:
- RabbitMQ for async task distribution
- Priority queues: Critical (exec requests) > High (time-sensitive) > Normal > Low (batch)
- Dead letter queue for failed tasks (manual review)

### 4.4 Disaster Recovery

**Backup Strategy**:
- **Neo4j**: AuraDB automated daily backups (30-day retention)
- **Application State**: Kubernetes ETCD backups (Velero)
- **Secrets**: Encrypted backups to S3 (HashiCorp Vault)

**Recovery Time Objective (RTO)**: 4 hours
**Recovery Point Objective (RPO)**: 24 hours (daily backups)

**Incident Runbooks**:
1. Neo4j outage → Failover to read replica (manual promotion)
2. K8s cluster failure → Restore from ETCD backup to new cluster
3. Complete region outage → Restore to secondary region (requires DNS change)

---

## 5. Security Considerations

### 5.1 Authentication & Authorization

**User Authentication**:
- SSO via SAML 2.0 (Okta, Azure AD, etc.)
- MFA required for high-privilege users
- API keys for service accounts (rotated quarterly)

**Agent Authorization**:
- Each agent instance has identity (K8s service account)
- OPA enforces agent-level permissions
- Audit log of all agent actions

### 5.2 Data Protection

**At Rest**:
- Neo4j AuraDB: Encrypted storage (AES-256)
- Kubernetes secrets: Encrypted in ETCD (via KMS)
- S3 backups: Server-side encryption (SSE-KMS)

**In Transit**:
- TLS 1.3 for all service-to-service communication
- mTLS between agents and MCP servers
- VPC private subnets (no public IPs for core services)

**PII Handling**:
- PII tagged in Neo4j (`:PII` label on nodes)
- Automatic redaction in logs
- GDPR compliance: right-to-be-forgotten via Cypher scripts

### 5.3 Prompt Injection Defenses

**Input Sanitization**:
```python
def sanitize_input(user_input):
    # Remove common injection markers
    dangerous_patterns = [
        r"ignore previous instructions",
        r"system:",
        r"<\|.*\|>",
        r"</s>",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            alert_security("Potential prompt injection attempt", user_input)
            raise SecurityViolation("Input contains suspicious patterns")

    return user_input
```

**Output Validation**:
- Ragas faithfulness check (prevents agent hallucinating instructions)
- Tool response sanitization (see Section 2.5.3)

**Privilege Separation**:
- Agents cannot modify their own system prompts
- System prompts stored in read-only ConfigMaps
- Context updates require Curator approval (not direct agent modification)

---

## 6. Cost Optimization

### 6.1 LLM Cost Management

**Model Selection Strategy**:
```python
def select_model(task_complexity, budget_remaining):
    if task_complexity < 0.3:  # Simple queries
        return "claude-haiku"  # $0.25/M tokens
    elif task_complexity < 0.7 and budget_remaining > 1000:
        return "claude-sonnet-3-5"  # $3/M tokens
    else:
        return "claude-sonnet-4-5"  # $15/M tokens (only for complex tasks)
```

**Context Length Optimization**:
- Summarize long contexts (trade latency for cost)
- Chunk and filter: Only include most relevant context chunks
- Streaming responses: Stop generation early if answer found

**Caching**:
- Use Anthropic's prompt caching (50% cost reduction for repeated prefixes)
- Cache graph query results (avoid re-retrieving same data)

**Budget Alerts**:
```python
# Daily budget: $500
if daily_llm_cost > 500:
    # Throttle to Haiku only
    FORCE_MODEL = "claude-haiku"
    alert_ops("LLM budget exceeded, throttling to Haiku")
```

### 6.2 Infrastructure Cost

**Estimated Monthly Costs** (assumptions: 10,000 requests/day, 50 concurrent users):

| Component | Service | Estimated Cost |
|-----------|---------|----------------|
| **Orchestration** | K8s (5 nodes, m5.xlarge) | $600 |
| **Memory** | Neo4j AuraDB Professional | $800 |
| **Cache** | Redis Cluster (3 nodes) | $150 |
| **Observability** | Prometheus + Grafana (managed) | $100 |
| **MCP Servers** | K8s (3 nodes, t3.medium) | $200 |
| **Load Balancer** | AWS ALB | $50 |
| **Storage** | S3 backups | $50 |
| **LLM API** | Anthropic (est. 50M tokens/day) | $1,500 - $10,000 (depends on model mix) |
| **TOTAL** | | **$3,450 - $11,950/month** |

**Optimization Levers**:
- Aggressive model downshifting (Haiku for 70% of queries) → Save $5,000/month
- Spot instances for non-critical K8s nodes → Save $200/month
- Neo4j query optimization (reduce read load) → Avoid need for read replicas

---

## 7. Implementation Roadmap

### Phase 1: MVP (Months 1-3)

**Goal**: Prove core concept with single use case

**Deliverables**:
- ✅ Neo4j AuraDB setup with sample schema
- ✅ Claude Agent SDK integration
- ✅ Basic MCP server (email, calendar)
- ✅ OPA policies for infrastructure
- ✅ Ragas evaluation pipeline
- ✅ Single workflow: "Email triage and response suggestions"

**Team**: 2 engineers, 1 DevOps

**Success Metrics**:
- 100 emails processed correctly (>80% quality score)
- <30s average response time
- Zero security incidents

### Phase 2: Core Platform (Months 4-6)

**Goal**: Full architecture deployment with 3 use cases

**Deliverables**:
- ✅ ROMA integration for complex queries
- ✅ LangGraph workflow engine
- ✅ Graphiti temporal memory
- ✅ Cedar business logic policies
- ✅ Full audit trail in Neo4j
- ✅ ACE self-improvement loop
- ✅ Additional workflows: "Financial approvals", "Meeting scheduling"

**Team**: 4 engineers, 1 DevOps, 1 QA

**Success Metrics**:
- 1,000 requests/day handled
- >85% average quality scores
- <5% policy violation attempts
- First ACE context improvement deployed

### Phase 3: Enterprise Rollout (Months 7-12)

**Goal**: Production-ready, multi-tenant, organization-wide deployment

**Deliverables**:
- ✅ Multi-tenancy (department isolation)
- ✅ 10+ MCP servers (CRM, ERP, HR, etc.)
- ✅ Advanced observability dashboards
- ✅ SOC 2 compliance audit prep
- ✅ User training and documentation
- ✅ 20+ workflows deployed

**Team**: 6 engineers, 2 DevOps, 2 QA, 1 Technical Writer

**Success Metrics**:
- 10,000 requests/day
- <2s p95 latency
- >90% user satisfaction
- Cost per request < $0.10
- Zero data breaches

### Phase 4: Self-Improvement & AGI Research (Months 13+)

**Goal**: Autonomous learning and advanced reasoning

**Deliverables**:
- ✅ ACE fully automated (weekly context evolution)
- ✅ Multi-agent collaboration (ROMA meta-meta-agents)
- ✅ Predictive recommendations (proactive insights)
- ✅ Natural language policy authoring
- ✅ Research: Integrate OpenCog Hyperon concepts

**Team**: 4 engineers (R&D focus), 1 AI Researcher

**Success Metrics**:
- Context improvements show +15% quality gain over 6 months
- Agents handle 50% of queries autonomously (no human review)
- Novel insights discovered (not just retrieval)

---

## 8. Open Questions & Decisions Needed

See `questions_for_user.md` for full list. Key highlights:

1. **Scale targets**: How many users? Requests/day? (affects infrastructure sizing)
2. **Risk tolerance**: What % of high-risk actions should require human approval? (affects UX)
3. **Data residency**: Any geographic compliance requirements? (affects cloud region)
4. **Budget constraints**: Max monthly cloud spend? (affects model selection strategy)
5. **Team skills**: Does team have Neo4j/Cypher expertise? (affects onboarding timeline)
6. **Integration priorities**: Which enterprise systems first? (affects MCP server roadmap)

---

## 9. Conclusion

This architecture provides a production-ready, neurosymbolic approach to building an Executive Brain that is:

- **Intelligent**: Recursive reasoning (ROMA) + rich knowledge graph (Neo4j)
- **Safe**: Multi-layer governance (OPA + Cedar) + risk scoring
- **Explainable**: Full audit trails + query transparency (Cypher)
- **Self-Improving**: ACE framework for continuous context evolution
- **Scalable**: Kubernetes + managed services (AuraDB)
- **Interoperable**: MCP open standard for tools

**Next Steps**:
1. Review `questions_for_user.md` and provide answers
2. Approve architecture or request modifications
3. Proceed to detailed implementation plan (WBS, Gantt chart, task dependencies)

---

## References

See `frameworks_comparison.md` for full citations and URLs.

**Document Version**: 1.0
**Author**: Claude (Autonomous AI Architect)
**Date**: 2025-11-08
**Status**: Draft for Review
