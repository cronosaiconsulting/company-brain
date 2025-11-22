# Executive Brain: Causal Reasoning & Hypothesis Testing Architecture

## Document Overview

**Purpose**: Enable reliable reasoning about facts, causality, effects, and hypothetical scenarios
**Version**: 1.0
**Date**: 2025-11-09
**Status**: Architectural extension to v2.1.1

---

## ⚠️ Evidence & Design Rationale

**This architecture is based on:**
- ✅ **Causal inference theory**: Pearl's causal graphs, do-calculus (Judea Pearl, "The Book of Why")
- ✅ **Knowledge graph research**: Temporal knowledge graphs, causal relation extraction
- ✅ **Hybrid reasoning**: Combine neural (LLM) with symbolic (graph traversal)
- ⚠️ **Graph schema**: Designed for this use case, not yet validated in production

**Key References:**
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
- Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks"
- OpenCog Hyperon: Symbolic reasoning + neural integration
- Neo4j Temporal Graph Patterns

---

## 1. Problem Statement

### 1.1 Why LLMs Alone Are Insufficient

**Current Limitation**: LLMs can generate plausible reasoning, but:

1. **Opaque**: Can't trace *why* a conclusion was reached
2. **Unreliable**: Hallucinate causal relationships
3. **Not compositional**: Can't combine facts reliably
4. **No counterfactuals**: Struggle with "what if X was false?"
5. **Inconsistent**: Same query → different causal chains

**Example Failure:**
```
User: "What if we hadn't hired John last quarter?"
LLM: "Revenue would have decreased by 15%"
     ← No justification, made up number, no causal chain
```

### 1.2 Requirements for Reliable Causal Reasoning

1. **Explicit causality**: Store "X causes Y" with confidence, not just "X relates to Y"
2. **Counterfactual support**: Answer "what if X didn't happen?"
3. **Hypothesis tracking**: Represent uncertain beliefs, update with evidence
4. **Reasoning transparency**: Trace conclusions back to facts
5. **Contradiction detection**: Flag when new facts conflict with existing beliefs
6. **Temporal awareness**: "X caused Y" implies X happened before Y

---

## 2. Causal Graph Architecture

### 2.1 Core Concept: Three-Layer Graph System

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: REASONING LAYER                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │ Hypothesis │  │ Conclusion │  │ Inference  │               │
│  │  Nodes     │  │   Nodes    │  │   Rules    │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│        │                │                │                       │
│        └────────────────┼────────────────┘                       │
│                         │ INFERRED_FROM                          │
│                         ▼                                         │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 2: CAUSAL LAYER                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │   Event    │──│   Action   │──│   Effect   │               │
│  │   Nodes    │  │   Nodes    │  │   Nodes    │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│        │ CAUSES     │ ENABLES      │ PREVENTS                   │
│        └────────────┼──────────────┘                            │
│                     │                                            │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 1: FACT LAYER                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │   Entity   │──│ Attribute  │──│ Relation   │               │
│  │   Nodes    │  │   Nodes    │  │   Edges    │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│        (People, Projects, Resources, Decisions, etc.)            │
└─────────────────────────────────────────────────────────────────┘
```

**Layer 1 (Fact Layer)**: Observed reality
- Entities: John (Person), ProjectX (Project), Q3 (TimePeriod)
- Attributes: revenue=$1M, status=completed
- Relations: WORKS_ON, REPORTS_TO, BELONGS_TO

**Layer 2 (Causal Layer)**: Cause-effect relationships
- Events: ProjectX_Completed, John_Hired
- Actions: Hire_John, Allocate_Budget
- Causal edges: Hire_John → CAUSES → Team_Productivity_Increased

**Layer 3 (Reasoning Layer)**: Hypotheses and conclusions
- Hypothesis: "If we hire 2 more engineers, revenue increases 10%"
- Conclusion: "Budget cuts will delay ProjectX by 3 weeks"
- Inference rules: Stored reasoning patterns

---

## 3. Graph Schema Design

### 3.1 Node Types

```cypher
// LAYER 1: FACT NODES
(:Entity {
  id: string,
  type: "Person|Project|Resource|Department|...",
  name: string,
  created_at: datetime,
  confidence: float,  // 1.0 = verified fact, <1.0 = uncertain
  source: string      // "database", "user_input", "llm_inference"
})

(:Observation {
  id: string,
  property: string,   // e.g., "revenue", "status", "headcount"
  value: any,
  timestamp: datetime,
  confidence: float,
  source: string
})

// LAYER 2: CAUSAL NODES
(:Event {
  id: string,
  description: string,
  timestamp: datetime,
  event_type: "decision|milestone|incident|change",
  confidence: float
})

(:Action {
  id: string,
  description: string,
  actor: string,      // Who took the action
  timestamp: datetime,
  reversible: boolean // Can this action be undone?
})

// LAYER 3: REASONING NODES
(:Hypothesis {
  id: string,
  statement: string,  // "Hiring 2 engineers increases revenue 10%"
  confidence: float,  // Prior probability
  created_at: datetime,
  status: "proposed|testing|validated|refuted"
})

(:Conclusion {
  id: string,
  statement: string,
  confidence: float,
  reasoning_chain: string,  // JSON of inference steps
  created_at: datetime
})

(:Counterfactual {
  id: string,
  original_event_id: string,     // What actually happened
  hypothetical_state: string,    // What if it didn't happen
  predicted_outcome: string,
  confidence: float
})
```

### 3.2 Edge Types (Causal Relationships)

```cypher
// TEMPORAL
-[:HAPPENED_BEFORE]-> {time_delta: duration}
-[:HAPPENED_AFTER]->
-[:CONCURRENT_WITH]->

// CAUSAL (LAYER 2)
-[:CAUSES]-> {
  confidence: float,          // 0.0-1.0
  strength: float,            // Effect size
  mechanism: string,          // How does X cause Y?
  evidence: [string],         // Supporting observations
  created_at: datetime
}

-[:ENABLES]-> {
  confidence: float,
  necessary: boolean,         // Is X necessary for Y?
  sufficient: boolean         // Is X sufficient for Y?
}

-[:PREVENTS]-> {
  confidence: float,
  effectiveness: float        // How much does X prevent Y?
}

-[:REQUIRES]-> {             // Dependency
  optional: boolean
}

-[:CONTRADICTS]-> {          // Logical contradiction
  severity: float
}

// EVIDENTIAL (LAYER 3)
-[:SUPPORTS]-> {
  confidence: float,
  evidence_type: "empirical|logical|testimonial"
}

-[:REFUTES]-> {
  confidence: float
}

-[:INFERRED_FROM]-> {
  inference_rule: string,    // What logic was used?
  llm_model: string,         // Which model generated this?
  reasoning_trace: string    // Step-by-step reasoning
}
```

---

## 4. Causal Reasoning Operations

### 4.1 Operation 1: Causal Query

**Question**: "What caused revenue to drop in Q3?"

**Graph Traversal**:
```cypher
// Find all events that happened before Q3 revenue drop
MATCH (e:Event)-[:HAPPENED_BEFORE]->(drop:Observation {property: "revenue", timestamp: Q3})
WHERE drop.value < (previous_value)

// Find causal paths
MATCH path = (e)-[:CAUSES*1..3]->(drop)
RETURN e, path,
       reduce(conf = 1.0, rel IN relationships(path) | conf * rel.confidence) AS total_confidence
ORDER BY total_confidence DESC
LIMIT 10
```

**Result**: Ranked list of causal factors with confidence scores

---

### 4.2 Operation 2: Counterfactual Reasoning

**Question**: "What if we hadn't hired John?"

**Algorithm**:
```python
def counterfactual_query(event_to_remove: str) -> Dict:
    """
    Simulate world where specified event didn't occur.
    Uses Pearl's do-calculus: P(Y | do(X=false))
    """
    # 1. Find all effects of the event
    downstream_effects = graph.query("""
        MATCH (event:Event {id: $event_id})-[:CAUSES*]->(effect)
        RETURN effect,
               length(path) as causal_distance,
               reduce(c=1.0, r IN relationships(path) | c*r.confidence) as conf
        ORDER BY causal_distance
    """, event_id=event_to_remove)

    # 2. For each effect, check if there are alternative causes
    counterfactual_world = {}
    for effect in downstream_effects:
        # Find alternative causal paths (not going through removed event)
        alternative_causes = graph.query("""
            MATCH path = (other:Event)-[:CAUSES*]->(effect:Event {id: $effect_id})
            WHERE NOT (removed_event)-[:CAUSES*]->(other)
            RETURN path,
                   reduce(c=1.0, r IN relationships(path) | c*r.confidence) as conf
        """, effect_id=effect.id, removed_event=event_to_remove)

        if alternative_causes:
            # Effect still occurs via alternative path
            counterfactual_world[effect.id] = {
                "status": "still_occurs",
                "confidence": max(c.conf for c in alternative_causes),
                "via": alternative_causes[0].path
            }
        else:
            # Effect would not have occurred
            counterfactual_world[effect.id] = {
                "status": "prevented",
                "confidence": effect.conf  # Original confidence
            }

    # 3. Generate natural language summary via LLM
    summary = llm.generate(f"""
        Given:
        - Event removed: {event_to_remove}
        - Downstream effects: {counterfactual_world}

        Summarize what would have happened differently.
    """)

    return {
        "counterfactual_world": counterfactual_world,
        "summary": summary,
        "confidence": compute_overall_confidence(counterfactual_world)
    }
```

**Example Output**:
```json
{
  "counterfactual_world": {
    "ProjectX_Completed": {
      "status": "prevented",
      "confidence": 0.85,
      "reason": "John was lead engineer, no alternative path"
    },
    "Revenue_Increase_Q3": {
      "status": "prevented",
      "confidence": 0.72,
      "reason": "Caused by ProjectX completion"
    }
  },
  "summary": "If John hadn't been hired, ProjectX would not have completed on time (85% confidence), resulting in Q3 revenue remaining flat (72% confidence)."
}
```

---

### 4.3 Operation 3: Hypothesis Testing

**Question**: "Will hiring 2 more engineers increase revenue by 10%?"

**Algorithm**:
```python
def evaluate_hypothesis(hypothesis: str) -> Dict:
    """
    Test hypothesis against historical causal patterns.
    """
    # 1. Parse hypothesis into structured form
    parsed = llm.extract_structured(hypothesis, schema={
        "action": str,      # "hire 2 engineers"
        "outcome": str,     # "revenue increases"
        "magnitude": float, # 0.10 (10%)
        "timeframe": str    # "next quarter"
    })

    # 2. Find similar historical actions
    similar_actions = graph.query("""
        MATCH (past_action:Action)
        WHERE past_action.description CONTAINS "hire"
          AND past_action.description CONTAINS "engineer"

        MATCH (past_action)-[:CAUSES*1..3]->(outcome:Observation {property: "revenue"})

        RETURN past_action,
               outcome,
               outcome.value - lag_value AS revenue_change,
               path
        ORDER BY past_action.timestamp DESC
        LIMIT 10
    """)

    # 3. Compute empirical distribution of effects
    historical_effects = [a.revenue_change for a in similar_actions]
    mean_effect = np.mean(historical_effects)
    std_effect = np.std(historical_effects)

    # 4. Bayesian update
    prior = 0.5  # Uninformed prior
    likelihood = scipy.stats.norm(mean_effect, std_effect).pdf(parsed.magnitude)
    posterior = update_belief(prior, likelihood)

    # 5. Check for confounding factors
    confounders = graph.query("""
        MATCH (confounder)-[:CAUSES]->(action:Action {type: "hiring"})
        MATCH (confounder)-[:CAUSES]->(outcome:Observation {property: "revenue"})
        RETURN confounder
    """)

    return {
        "hypothesis": hypothesis,
        "posterior_probability": posterior,
        "supporting_evidence": similar_actions,
        "expected_effect": mean_effect,
        "uncertainty": std_effect,
        "confounders": confounders,
        "recommendation": "test" if posterior > 0.3 else "reject"
    }
```

**Example Output**:
```json
{
  "hypothesis": "Hiring 2 engineers increases revenue 10%",
  "posterior_probability": 0.42,
  "supporting_evidence": [
    {"action": "Hired 3 engineers Q1 2024", "revenue_change": 0.08},
    {"action": "Hired 1 engineer Q3 2024", "revenue_change": 0.03}
  ],
  "expected_effect": 0.055,
  "uncertainty": 0.025,
  "confounders": ["New product launch", "Market expansion"],
  "recommendation": "test - hypothesis plausible but uncertain"
}
```

---

### 4.4 Operation 4: Reasoning Chain Extraction

**Capture LLM reasoning in graph form**:

```python
def extract_reasoning_to_graph(llm_output: str, query: str) -> str:
    """
    Parse LLM reasoning trace and store as graph.
    Enables future reuse and validation.
    """
    # 1. Ask LLM to structure its reasoning
    structured_reasoning = llm.generate(f"""
        Original query: {query}
        Your reasoning: {llm_output}

        Extract reasoning as JSON:
        {{
          "facts_used": [string],
          "assumptions": [string],
          "inference_steps": [
            {{"from": [string], "to": string, "rule": string}}
          ],
          "conclusion": string,
          "confidence": float
        }}
    """)

    reasoning = json.loads(structured_reasoning)

    # 2. Store in graph
    conclusion_node = graph.create_node("Conclusion", {
        "statement": reasoning["conclusion"],
        "confidence": reasoning["confidence"],
        "query": query,
        "created_at": datetime.now()
    })

    # 3. Link to facts
    for fact in reasoning["facts_used"]:
        fact_node = graph.find_or_create("Observation", {"description": fact})
        graph.create_edge(conclusion_node, fact_node, "INFERRED_FROM", {
            "reasoning_trace": json.dumps(reasoning["inference_steps"])
        })

    # 4. Flag assumptions
    for assumption in reasoning["assumptions"]:
        assumption_node = graph.create_node("Hypothesis", {
            "statement": assumption,
            "status": "assumed",
            "needs_validation": True
        })
        graph.create_edge(conclusion_node, assumption_node, "DEPENDS_ON")

    return conclusion_node.id
```

**Benefit**: Later, if assumption is validated/refuted, we can automatically update all dependent conclusions.

---

## 5. Integration with Existing Architecture

### 5.1 ROMA Integration

**Planner Enhancement**: Query causal graph before task decomposition

```python
class CausalAwareROMAPlan:
    def plan_with_causal_awareness(self, task: str):
        """
        Use causal graph to improve planning.
        """
        # 1. Extract goal from task
        goal = extract_goal(task)  # e.g., "increase revenue"

        # 2. Query causal graph for known paths to goal
        causal_paths = graph.query("""
            MATCH path = (action:Action)-[:CAUSES*]->(outcome:Observation)
            WHERE outcome.property = $goal_property
              AND outcome.value > current_value
            RETURN action, path,
                   reduce(c=1.0, r IN relationships(path) | c*r.confidence) as conf
            ORDER BY conf DESC
            LIMIT 5
        """, goal_property=goal)

        # 3. Use known causal paths as priors for planning
        plan_candidates = []
        for path in causal_paths:
            # Adapt historical action to current context
            adapted_plan = self.adapt_causal_path(path, current_context)
            plan_candidates.append(adapted_plan)

        # 4. Let ROMA fill in gaps / create novel plans
        roma_plan = self.roma.plan(task, priors=plan_candidates)

        return roma_plan
```

---

### 5.2 Effort Regulation Integration

**Allocate more effort to causal queries**:

```python
def analyze_task_complexity(task: str) -> float:
    # Existing dimensions...
    complexity = {
        "reasoning_depth": 0.5,
        # ...
    }

    # NEW: Check if task involves causal reasoning
    causal_indicators = [
        "why", "what caused", "what if", "would have",
        "because", "due to", "resulted in", "led to"
    ]

    if any(indicator in task.lower() for indicator in causal_indicators):
        # Causal reasoning is expensive - needs graph traversal + LLM
        complexity["causal_reasoning"] = 0.8  # High complexity

    return compute_weighted_complexity(complexity)
```

---

### 5.3 ACE Integration (Self-Improvement)

**Curator updates causal graph**:

```python
class ACE_CausalCurator:
    def curate_reasoning(self, task: str, result: str, quality: float):
        """
        Extract causal knowledge from successful reasoning.
        """
        if quality > 0.8:  # High-quality reasoning
            # Extract causal relationships mentioned
            causal_relations = llm.extract(f"""
                From this reasoning: {result}
                Extract causal statements in format:
                [
                  {{"cause": "X", "effect": "Y", "confidence": 0.0-1.0}}
                ]
            """)

            # Add to causal graph
            for relation in causal_relations:
                cause_node = graph.find_or_create("Event", {"description": relation["cause"]})
                effect_node = graph.find_or_create("Event", {"description": relation["effect"]})

                # Check if edge already exists
                existing = graph.find_edge(cause_node, effect_node, "CAUSES")
                if existing:
                    # Update confidence (moving average)
                    existing.confidence = 0.9 * existing.confidence + 0.1 * relation["confidence"]
                else:
                    # Create new edge
                    graph.create_edge(cause_node, effect_node, "CAUSES", {
                        "confidence": relation["confidence"],
                        "source": "llm_reasoning",
                        "created_at": datetime.now()
                    })
```

---

## 6. Implementation Approach

### 6.1 Phase 1: Core Infrastructure (Week 1-2)

**Tasks**:
1. Extend Neo4j schema with new node/edge types
2. Implement causal query functions (causal_query, counterfactual_query)
3. Build reasoning chain extractor
4. Unit tests for graph operations

**Deliverables**:
- Updated Neo4j schema
- Python library: `causal_reasoning.py`
- Test suite

---

### 6.2 Phase 2: LLM Integration (Week 3-4)

**Tasks**:
1. Integrate causal queries into ROMA planner
2. Build hypothesis testing workflow
3. Implement reasoning trace extraction
4. Add causal complexity dimension to effort regulation

**Deliverables**:
- ROMA + causal graph integration
- Hypothesis evaluation API
- Effort regulation updated

---

### 6.3 Phase 3: Validation & Tuning (Week 5-6)

**Tasks**:
1. Populate causal graph with historical data
2. Benchmark counterfactual accuracy
3. Tune confidence thresholds
4. A/B test causal-aware vs baseline planning

**Deliverables**:
- Populated knowledge graph
- Benchmark report
- Tuned confidence parameters

---

## 7. Evidence-Based Design Decisions

### 7.1 Why Graph Database (Neo4j)?

**Evidence**:
- ✅ Graph traversal is O(k) for k-hop queries (constant time per hop)
- ✅ Cypher supports recursive queries (`:CAUSES*1..5`)
- ✅ Neo4j supports temporal queries natively
- ✅ Battle-tested for knowledge graphs (Google, LinkedIn, NASA)

**Alternatives Considered**:
- Relational DB: ⛔ Poor for multi-hop queries, no native graph traversal
- Triple store: ⚠️ Could work, but less mature tooling
- Custom graph: ⛔ Reinventing the wheel

---

### 7.2 Why Hybrid Neural-Symbolic?

**Evidence**:
- ✅ LLMs are good at: pattern matching, language understanding, generating hypotheses
- ✅ Graphs are good at: logical reasoning, consistency checking, causal inference
- ✅ Neurosymbolic AI research (Garcez et al. 2019): Combining both > either alone

**Approach**:
1. Use LLM to propose causal relationships
2. Store in graph for verification
3. Use graph traversal for logical inference
4. Use LLM to interpret results in natural language

---

### 7.3 Confidence Scoring

**Approach**: Probabilistic confidence propagation

```python
def propagate_confidence(path: List[Edge]) -> float:
    """
    Confidence in conclusion = product of edge confidences.
    Based on probability theory: P(A ∧ B) ≤ min(P(A), P(B))
    """
    return reduce(lambda acc, edge: acc * edge.confidence, path, 1.0)
```

**⚠️ ASSUMPTION**: Edges are independent (may not hold in practice)

**TODO**: Implement Bayesian network for dependent causal factors

---

## 8. Example: End-to-End Causal Reasoning

**User Query**: "Should we cut the marketing budget by 20%?"

**System Flow**:

```
1. PARSING
   ├─ Action: "cut marketing budget 20%"
   ├─ Implicit question: "What would be the effects?"
   └─ Type: Counterfactual reasoning

2. CAUSAL GRAPH QUERY
   ├─ Find: marketing_budget -[:CAUSES]-> ?
   └─ Results:
       ├─ marketing_budget -[:CAUSES]-> lead_generation (conf: 0.85)
       ├─ lead_generation -[:CAUSES]-> sales (conf: 0.78)
       └─ sales -[:CAUSES]-> revenue (conf: 0.92)

3. COUNTERFACTUAL SIMULATION
   ├─ Remove: 20% of marketing_budget
   └─ Propagate effects:
       ├─ lead_generation: -15% (0.85 × 0.78 = 0.66 conf)
       ├─ sales: -10% (0.66 × 0.92 = 0.61 conf)
       └─ revenue: -8% (0.61 conf)

4. CONFIDENCE ASSESSMENT
   └─ Overall confidence: 0.61 (moderate)

5. HYPOTHESIS CHECK
   ├─ Query: Has this been tested before?
   └─ Historical: "2023 Q2 budget cut → -12% revenue" (0.90 conf)

6. LLM SYNTHESIS
   └─ Generate report:
       "Cutting marketing budget by 20% would likely reduce:
       - Lead generation by ~15% (high confidence)
       - Sales by ~10% (moderate confidence)
       - Revenue by ~8% (moderate confidence)

       Historical precedent: Similar cut in Q2 2023 led to 12% revenue drop.

       Recommendation: Do not proceed unless:
       1. Alternative lead sources identified
       2. Current marketing ROI < 1.2x
       3. Short-term cash flow critical"

7. STORE REASONING
   └─ Save reasoning chain to graph for future reference
```

---

## 9. Monitoring & Validation

### 9.1 Causal Graph Metrics

```python
# Quality metrics
causal_graph_density = num_causal_edges / (num_events * (num_events - 1))
avg_confidence = mean([edge.confidence for edge in causal_edges])
contradiction_rate = count_contradictions() / num_conclusions

# Coverage metrics
hypothesis_validation_rate = validated / total_hypotheses
counterfactual_accuracy = correct_predictions / total_counterfactual_queries

# Prometheus metrics
causal_query_latency = Histogram("causal_query_duration_seconds")
reasoning_chain_length = Histogram("reasoning_chain_hops")
confidence_distribution = Histogram("causal_confidence_score")
```

### 9.2 Validation Strategy

**Continuous Validation**:
1. When predictions are made, track actual outcomes
2. Update causal edge confidences based on outcomes
3. Flag causal relationships that consistently fail
4. Retrain/re-extract from new data

**Example**:
```python
def validate_prediction(prediction_id: str, actual_outcome: Any):
    """
    Update causal graph based on outcome.
    """
    prediction = graph.get_node("Conclusion", prediction_id)

    # Compare predicted vs actual
    error = abs(prediction.predicted_value - actual_outcome)

    # Update confidence of causal path
    causal_path = graph.query("""
        MATCH path = ()-[:INFERRED_FROM*]->(:Conclusion {id: $pred_id})
        RETURN path
    """, pred_id=prediction_id)

    for edge in causal_path.edges:
        # Bayesian update
        edge.confidence = bayesian_update(
            prior=edge.confidence,
            likelihood=1.0 - error,  # Low error = high likelihood
            evidence_weight=0.1      # 10% update per observation
        )
```

---

## 10. Open Questions & Future Work

### 10.1 Challenges

1. **Causal Discovery**: How to automatically extract causal relationships from text?
   - ⚠️ Active research area (LLM-based causal extraction)
   - Consider: CausalBERT, Causal-BERT models

2. **Confounding Variables**: How to detect hidden confounders?
   - ⚠️ Requires domain expertise + statistical testing
   - Consider: Propensity score matching, instrumental variables

3. **Temporal Consistency**: How to handle time-varying causal relationships?
   - ⚠️ Marketing effectiveness changes over time
   - Consider: Temporal knowledge graphs, dynamic Bayesian networks

4. **Scalability**: Graph queries on large causal networks?
   - ⚠️ 10K+ nodes, 50K+ edges
   - Consider: Graph sampling, approximate queries

### 10.2 Extensions

1. **Multi-Agent Causal Reasoning**: Multiple agents propose/debate causal hypotheses
2. **Causal RL**: Learn which actions maximize desired outcomes
3. **Explanation Generation**: Natural language explanations of causal chains
4. **Interactive Refinement**: User corrects causal graph, system learns

---

## 11. Conclusion

This causal reasoning architecture provides:

✅ **Explicit causality modeling** via graph edges
✅ **Counterfactual reasoning** via graph traversal + do-calculus
✅ **Hypothesis testing** via historical pattern matching
✅ **Reasoning transparency** via stored reasoning chains
✅ **Contradiction detection** via graph consistency checks
✅ **Temporal awareness** via timestamp properties + HAPPENED_BEFORE edges

**Integration points**:
- ROMA: Use causal graph for better planning
- Effort Regulation: Allocate more compute to causal tasks
- ACE: Extract and validate causal knowledge over time
- Neo4j: Natural extension of existing knowledge graph

**Next Steps**:
1. Review schema with team
2. Prototype Phase 1 (core infrastructure)
3. Validate with real queries
4. Iterate based on results

---

## 12. References

**Causal Inference**:
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
- Pearl, J., & Mackenzie, D. (2018). "The Book of Why"

**Knowledge Graphs**:
- Hogan et al. (2021). "Knowledge Graphs" (survey paper)
- Schlichtkrull et al. (2018). "Modeling Relational Data with GCNs"

**Neurosymbolic AI**:
- Garcez et al. (2019). "Neurosymbolic AI: The State of the Art"
- Lamb et al. (2020). "Graph Neural Networks Meet Neural-Symbolic Computing"

**Temporal Knowledge Graphs**:
- Trivedi et al. (2017). "Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs"

**Causal Discovery from Text**:
- Zhao et al. (2021). "CausalBERT: Injecting Causal Knowledge Into Pre-trained Language Models"

---

**Document Version**: 1.0
**Date**: 2025-11-09
**Status**: Proposed architecture - requires validation
**Complements**: architecture_v2.1_model_agnostic.md, effort_regulation_framework.md
