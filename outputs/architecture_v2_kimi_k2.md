# Executive Brain v2.0: Architecture with Kimi K2 Thinking & Effort Regulation

## Document Overview

**Purpose**: Updated architecture specification using Kimi K2 Thinking as primary LLM with sophisticated effort regulation
**Version**: 2.0 (supersedes v1.0 Claude-based architecture)
**Date**: 2025-11-08
**Key Changes**:
- Primary LLM: Claude → **Kimi K2 Thinking** (open source, self-hostable)
- New layer: **Effort Regulation Orchestrator**
- Dynamic thinking budget allocation
- Cost optimization: 1/12th of Claude 4 Sonnet

---

## 1. Executive Summary

### 1.1 Vision (Unchanged)

Build an AI system that acts as an **executive cognitive layer** for an organization, capable of autonomous reasoning, decision-making, and self-improvement.

### 1.2 Architecture v2.0 Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTIVE BRAIN v2.0                          │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              INPUT NORMALIZATION LAYER                    │  │
│  │  [No changes from v1.0]                                   │  │
│  └───────────────────────┬──────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────────┐  │
│  │       EFFORT REGULATION ORCHESTRATOR ← NEW               │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │ Task Complexity Analyzer (6 dimensions)           │   │  │
│  │  │ Context Analyzer (urgency, risk, budget)          │   │  │
│  │  │ Strategy Selector (minimal/fast/balanced/         │   │  │
│  │  │                    thorough/maximum)              │   │  │
│  │  │ Kimi K2 Parameter Configurator                    │   │  │
│  │  │ Adaptive Retry & Learning Module                  │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  └───────────────────────┬──────────────────────────────────┘  │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────────┐  │
│  │       ORCHESTRATION & REASONING LAYER (Updated)          │  │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │ Kimi K2     │  │  ROMA    │  │   LangGraph      │    │  │
│  │  │ Thinking    │◄─┤ (Meta-   │◄─┤  (Workflows)     │    │  │
│  │  │ (Primary)   │  │  Agent)  │  │                  │    │  │
│  │  │ + Effort    │  │ + Effort │  │                  │    │  │
│  │  │   Params    │  │   Alloc  │  │                  │    │  │
│  │  └──────┬──────┘  └────┬─────┘  └────────┬─────────┘    │  │
│  └─────────┼──────────────┼─────────────────┼──────────────┘  │
│            │              │                 │                  │
│  [Memory, Governance, Tools, Evaluation, Audit layers         │
│   remain largely unchanged from v1.0]                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Advantages of v2.0

| Dimension | v1.0 (Claude-based) | v2.0 (Kimi K2-based) | Improvement |
|-----------|---------------------|----------------------|-------------|
| **Cost** | $15/M tokens (Sonnet 4.5) | ~$1.25/M tokens (estimated) | **12x cheaper** |
| **Open Source** | No | Yes (Modified MIT) | Self-hostable, customizable |
| **Context Window** | 200K | 256K | 28% larger |
| **Thinking Budget** | Fixed | Variable (1K-256K) | Dynamic optimization |
| **Agentic Capability** | Good | Excellent (200-300 tool calls) | Superior for workflows |
| **Effort Regulation** | No | Yes | 40-60% cost reduction |
| **Benchmarks** | SOTA (Claude 4.5) | Beats GPT-5, Claude 4.5 on many | Competitive or better |
| **Vendor Lock-in** | High (Anthropic-specific) | None (self-host) | Strategic flexibility |

---

## 2. Updated Core Components

### 2.1 Primary LLM: Kimi K2 Thinking

**Specifications**:
- **Model**: Moonshot AI Kimi K2 Thinking
- **Architecture**: Mixture-of-Experts (MoE), 1T params, 32B active
- **Quantization**: Native INT4 (lossless)
- **Context**: 256K tokens
- **Thinking Budget**: Variable (1K - 256K tokens per task)
- **License**: Modified MIT (open source)
- **Availability**:
  - API: platform.moonshot.ai, kimi.com
  - Self-hosted: Hugging Face (moonshotai/Kimi-K2-Thinking)

**Integration Pattern**:
```python
from moonshot_ai import KimiK2Client

client = KimiK2Client(api_key=os.getenv("MOONSHOT_API_KEY"))

def execute_with_kimi_k2(task, effort_config, contexts):
    response = client.chat.completions.create(
        model="kimi-k2-thinking",
        messages=[
            {"role": "system", "content": EXECUTIVE_BRAIN_SYSTEM_PROMPT},
            {"role": "user", "content": construct_prompt(task, contexts)}
        ],
        # Effort-regulated parameters
        max_steps=effort_config["max_steps"],
        thinking_budget_per_step=effort_config["thinking_budget_per_step"],
        temperature=effort_config["temperature"],
        timeout=effort_config["timeout"],
        stream=True  # Real-time thinking trace
    )

    return process_streaming_response(response)
```

### 2.2 Effort Regulation Orchestrator (NEW)

**See**: `effort_regulation_system.md` for complete specification

**Key Functions**:
1. **Task Complexity Analysis** (6 dimensions):
   - Reasoning depth (single-step → deep multi-step)
   - Knowledge breadth (single domain → cross-domain)
   - Tool orchestration complexity (no tools → 10+ tools)
   - Ambiguity (clear → highly ambiguous)
   - Constraints (none → many complex constraints)
   - Novelty (cached → completely novel)

2. **Context Analysis**:
   - Urgency (batch → real-time critical)
   - Risk (low-risk → high-risk)
   - Budget remaining (exhausted → plenty)
   - User preferences (speed vs accuracy)

3. **Strategy Selection** (5 levels):
   - **Minimal**: 5 steps, 5K tokens total (simple queries)
   - **Fast**: 10 steps, 40K tokens (routine tasks)
   - **Balanced**: 50 steps, 96K tokens (standard reasoning)
   - **Thorough**: 120 steps, 128K tokens (complex analysis)
   - **Maximum**: 300 steps, 256K tokens (critical decisions)

4. **Adaptive Retry**:
   - Execute with initial effort level
   - Evaluate quality (Ragas)
   - If quality insufficient → retry with more effort
   - Learn optimal effort levels over time

**Example Allocation**:
```
Task: "What's the Phoenix project status?"
├─ Intrinsic Complexity: 0.2 (simple factual query)
├─ Context: Low urgency, low risk, budget OK
├─ Final Effort: 0.2
└─ Strategy: FAST (10 steps, 40K tokens, <30s)

Task: "Analyze Q3 spending and recommend 10% budget cuts"
├─ Intrinsic Complexity: 0.73 (deep analysis + synthesis)
├─ Context: Medium urgency, high risk, budget OK
├─ Final Effort: 0.77
└─ Strategy: THOROUGH (120 steps, 128K tokens, <5min)
```

### 2.3 ROMA Integration with Per-Subtask Effort Allocation

**Enhancement**: ROMA meta-agent now allocates effort to each subtask based on:
- Subtask intrinsic complexity
- Parent task effort budget
- Subtask priority (critical path gets more effort)

**Example**:
```
Parent Task Effort: 0.77 (thorough)

Decomposition:
├─ Subtask 1: "Retrieve Q3 spending" → 0.71 (thorough)
├─ Subtask 2: "Analyze spending vs revenue" → 0.98 (maximum)
├─ Subtask 3: "Identify cost reductions" → 1.0 (maximum)
├─ Subtask 4: "Simulate scenarios" → 0.80 (maximum)
└─ Subtask 5: "Synthesize recommendations" → 1.0 (maximum)
```

**Why Different Efforts**:
- Critical reasoning tasks get max effort (accuracy critical)
- Simple data retrieval can use less effort (data quality more important than reasoning)
- Novel creative tasks get max effort (no cached patterns)

---

## 3. Cost Analysis v2.0

### 3.1 Kimi K2 Pricing Model

**Estimated Pricing** (based on "1/12th of Claude 4 Sonnet"):

| Token Type | Claude Sonnet 4.5 | Kimi K2 Thinking (est.) | Savings |
|------------|-------------------|------------------------|---------|
| **Input** | $3/M tokens | $0.25/M tokens | 12x cheaper |
| **Output (thinking)** | N/A | $0.50/M tokens | New capability |
| **Output (response)** | $15/M tokens | $1.25/M tokens | 12x cheaper |

**Note**: If self-hosting, cost is infrastructure-based (GPU amortization), not per-token.

### 3.2 Cost Comparison by Strategy

| Strategy | Avg Tokens | Cost (API) | Cost (Self-hosted) | Use Case |
|----------|-----------|------------|-------------------|----------|
| **Minimal** | 15K | $0.019 | $0.001 | Simple queries |
| **Fast** | 60K | $0.075 | $0.005 | Routine tasks |
| **Balanced** | 150K | $0.188 | $0.012 | Standard reasoning |
| **Thorough** | 250K | $0.313 | $0.020 | Complex analysis |
| **Maximum** | 500K | $0.625 | $0.040 | Critical decisions |

**Estimated Monthly Cost** (10,000 requests/day, mixed strategies):

| Deployment | Effort Distribution | Monthly Cost |
|------------|---------------------|--------------|
| **API (No Optimization)** | 100% maximum effort | $187,500 |
| **API (With Effort Regulation)** | 30% minimal, 40% fast, 20% balanced, 8% thorough, 2% maximum | $31,250 (83% savings) |
| **Self-Hosted (Spot)** | All strategies | $7,200 (infrastructure) |
| **Self-Hosted (Reserved)** | All strategies | $23,040 (infrastructure) |

**Break-even Analysis**:
- If monthly request volume > 23,000 → Self-host on spot instances cheaper
- If monthly request volume > 73,000 → Self-host on reserved instances cheaper

### 3.3 v1.0 vs v2.0 Cost Comparison

**Scenario**: 10,000 requests/day, mixed complexity

| Architecture | Monthly Cost | Notes |
|--------------|--------------|-------|
| **v1.0 (Claude, no optimization)** | $225,000 | All tasks use Sonnet 4.5 |
| **v1.0 (Claude, with model selection)** | $90,000 | Mix Haiku/Sonnet/Opus |
| **v2.0 (Kimi K2 API, with effort regulation)** | $31,250 | Dynamic thinking budgets |
| **v2.0 (Kimi K2 self-hosted, spot)** | $7,200 | Plus one-time setup cost |

**Savings**: v2.0 achieves **65-97% cost reduction** vs v1.0

---

## 4. Deployment Architecture v2.0

### 4.1 Kimi K2 Deployment Options

#### Option A: API Deployment (MVP / Low-Medium Volume)

**Pros**:
- Zero infrastructure setup
- Instant start
- No GPU management
- Elastic scaling

**Cons**:
- Data leaves premises (privacy concern)
- Rate limits
- Per-token costs
- Dependent on Moonshot AI availability

**When to Use**: MVP, < 23,000 requests/month, no data residency requirements

**Architecture**:
```
Executive Brain K8s Cluster
        ↓
    API Gateway
        ↓
Moonshot AI API (kimi-k2-thinking)
```

#### Option B: Self-Hosted (High Volume / Data Privacy)

**Pros**:
- Full data control
- No rate limits
- Predictable infrastructure costs
- Customizable (can fine-tune)

**Cons**:
- Requires GPU infrastructure (4x A100 minimum)
- Operational complexity
- Upfront hardware/cloud cost

**When to Use**: > 23,000 requests/month, data residency requirements, long-term deployment

**Infrastructure Requirements**:
```yaml
GPU: 4x NVIDIA A100 80GB (INT4 quantized model)
RAM: 256GB system memory
Storage: 500GB SSD (model weights + cache)
Network: 10Gbps for fast model loading

Kubernetes Node Spec:
- AWS: p4d.24xlarge (~$32/hour reserved, ~$10/hour spot)
- GCP: a2-ultragpu-4g (~$28/hour)
- Azure: ND96asr_v4 (~$30/hour)
```

**Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kimi-k2-inference
  namespace: executive-brain-core
spec:
  replicas: 2  # Redundancy + load balancing
  selector:
    matchLabels:
      app: kimi-k2
  template:
    spec:
      nodeSelector:
        gpu: nvidia-a100
      containers:
      - name: kimi-k2
        image: moonshotai/kimi-k2-thinking:int4-latest
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: 128Gi
          requests:
            nvidia.com/gpu: 4
            memory: 128Gi
        env:
        - name: MAX_CONCURRENT_REQUESTS
          value: "10"
        - name: THINKING_BUDGET_LIMIT
          value: "256000"
        - name: QUANTIZATION
          value: "int4"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

#### Option C: Hybrid (Recommended for Enterprise)

**Strategy**:
- **Self-host**: Handle routine/high-volume tasks (minimal, fast strategies)
- **API**: Handle burst capacity and maximum-effort tasks (thorough, maximum strategies)

**Benefits**:
- Cost optimization (self-host for 80% of volume)
- Reliability (API as backup if self-hosted fails)
- Elastic scaling (API handles spikes)

**Cost Example** (10K requests/day):
- 80% handled by self-host (8K req/day) → $7,200/month infrastructure
- 20% handled by API (2K req/day, complex tasks) → $9,375/month API
- Total: $16,575/month (vs $31,250 API-only or $225,000 Claude-only)

### 4.2 Updated Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              KUBERNETES CLUSTER (v2.0)                       │
│                                                              │
│  Namespace: executive-brain-core                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Effort Regulation Orchestrator (Deployment, 5 pods)   │  │
│  │ - Task Complexity Analyzer                            │  │
│  │ - Strategy Selector                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Kimi K2 Inference Service (Deployment, 2 pods)        │  │
│  │ - 4x A100 GPU per pod                                 │  │
│  │ - INT4 quantized model                                │  │
│  │ - Max 10 concurrent requests per pod                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ ROMA Meta-Agent (Deployment, 3 pods)                  │  │
│  │ - With per-subtask effort allocation                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  [Neo4j, Redis, OPA, Cedar, MCP servers unchanged]          │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Updated Requirements & Constraints

### 5.1 New Requirements (v2.0)

**Functional**:
- **Dynamic Effort Allocation**: System must analyze task complexity and allocate appropriate thinking budget
- **Adaptive Quality**: If initial effort insufficient, automatically retry with more effort
- **Cost Awareness**: System must stay within daily/monthly LLM budget via effort throttling
- **Effort Learning**: System should learn optimal effort levels for task types over time

**Non-Functional**:
- **Cost Efficiency Target**: 65-85% cost reduction vs v1.0 (Claude-based)
- **Quality Maintenance**: >90% of tasks meet quality thresholds (faithfulness >0.8, relevance >0.7) on first attempt
- **Latency Distribution**:
  - Simple queries (minimal/fast): <30s
  - Standard reasoning (balanced): <2min
  - Complex analysis (thorough/maximum): <15min

### 5.2 Updated Constraints

**Technical**:
- **GPU Availability** (if self-hosting): Requires A100 or equivalent GPUs
- **Kimi K2 Maturity**: Model is new (Nov 2025), ecosystem still growing
- **Self-Hosting Complexity**: Requires ML Ops expertise for model deployment, quantization, optimization

**Operational**:
- **Model Updates**: Moonshot AI may release new versions - need update strategy
- **API Rate Limits**: If using API, subject to Moonshot AI rate limits (need negotiation for enterprise)
- **Hybrid Orchestration**: Managing both self-hosted and API endpoints adds complexity

**See**: `constraints_v2_kimi_k2.json` for comprehensive updated constraints

---

## 6. Migration Path from v1.0 to v2.0

### 6.1 Incremental Migration Strategy

**Phase 1: Pilot (Month 1)**
1. Deploy Kimi K2 API alongside existing Claude-based system
2. Implement effort regulation orchestrator
3. Route 10% of traffic to Kimi K2 (low-risk queries)
4. Compare quality, latency, cost metrics
5. Collect feedback

**Phase 2: Expansion (Months 2-3)**
1. If pilot successful, route 50% of traffic to Kimi K2
2. Deploy self-hosted Kimi K2 (if volume justifies)
3. Hybrid deployment: self-hosted for routine, API for complex
4. ACE learning module trains on Kimi K2 outputs

**Phase 3: Full Migration (Months 4-6)**
1. Route 90% of traffic to Kimi K2
2. Keep Claude as fallback for edge cases
3. Optimize effort allocation based on 3 months of data
4. Finalize self-hosted vs API split

**Phase 4: Claude Deprecation (Month 7+)**
1. Evaluate if Claude still needed (likely for specific tasks only)
2. Consider full Kimi K2 migration
3. Document edge cases where Claude outperforms

### 6.2 Rollback Plan

If Kimi K2 underperforms:
- Immediately route traffic back to Claude
- Analyze failure modes (quality, latency, cost overruns)
- Decide: (a) Fix Kimi K2 integration issues, or (b) Stick with Claude v1.0

**Decision Criteria for Rollback**:
- Quality scores drop >10% vs Claude
- Latency increases >50% for same task complexity
- Cost savings <30% (insufficient ROI)
- User satisfaction drops significantly

---

## 7. Updated Success Metrics

### 7.1 v2.0-Specific KPIs

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Cost per Request** | <$0.03 (vs $0.10 v1.0) | 70% cost reduction |
| **Effort Allocation Accuracy** | >85% of tasks allocated optimal effort | Minimize retries |
| **Quality on First Attempt** | >90% meet thresholds | Adaptive retry working |
| **Thinking Budget Utilization** | 60-80% of allocated budget used | Not over/under-allocating |
| **Self-Hosted Uptime** (if applicable) | 99.5% | Reliability of custom deployment |
| **API → Self-Hosted Migration** | 80% of volume self-hosted by Month 6 | Cost optimization |

### 7.2 Comparison Metrics (v1.0 vs v2.0)

| Metric | v1.0 Baseline | v2.0 Target | Measurement |
|--------|---------------|-------------|-------------|
| **Monthly LLM Cost** | $90,000 | $31,250 (API) or $7,200 (self-hosted) | Actual spend |
| **Average Quality Score** | 0.85 | ≥0.85 | Ragas faithfulness |
| **p95 Latency (complex)** | 30s | <2min (allows more thinking) | Prometheus |
| **Autonomous Decision Rate** | 80% | ≥80% | No degradation |

---

## 8. Open Questions & User Decisions

### 8.1 Critical Decisions Needed

1. **Deployment Model**:
   - [ ] API-only (simplest, moderate cost)
   - [ ] Self-hosted only (complex, lowest long-term cost)
   - [ ] Hybrid (recommended, balanced)

2. **GPU Infrastructure** (if self-hosting):
   - [ ] Cloud GPU (AWS/GCP/Azure)
   - [ ] On-premises GPU cluster
   - [ ] Spot instances (60% cheaper, risk of interruption)
   - [ ] Reserved instances (stable, higher cost)

3. **Migration Pace**:
   - [ ] Aggressive (full migration in 3 months)
   - [ ] Moderate (6 months, recommended)
   - [ ] Conservative (12 months, pilot extensively)

4. **Effort Allocation Philosophy**:
   - [ ] Cost-optimized (default to minimal/fast, escalate only if needed)
   - [ ] Quality-optimized (default to balanced/thorough, accept higher cost)
   - [ ] User-customizable (let users choose speed vs accuracy)

5. **Claude Retention**:
   - [ ] Keep Claude as fallback indefinitely
   - [ ] Deprecate Claude after successful Kimi K2 migration
   - [ ] Use Claude only for specific edge cases (which ones?)

### 8.2 Data Collection for Informed Decision

**Pilot Program** (Month 1):
- Run A/B test: Claude vs Kimi K2 on same tasks
- Measure:
  - Quality scores (Ragas)
  - Latency (p50, p95, p99)
  - Cost per request
  - User satisfaction (explicit feedback)
- Analyze failure modes:
  - Where does Kimi K2 outperform Claude?
  - Where does Claude outperform Kimi K2?

**Decision Point** (End of Month 1):
- If Kimi K2 quality ≥ Claude and cost < 50% Claude → Proceed to Phase 2
- If Kimi K2 quality < Claude by >10% → Investigate issues, extend pilot
- If cost savings < 30% → Re-evaluate effort regulation strategy

---

## 9. Conclusion

### 9.1 Why v2.0 is Superior

1. **Cost**: 65-97% reduction vs v1.0 (depending on deployment)
2. **Open Source**: Self-hostable, customizable, no vendor lock-in
3. **Performance**: Beats GPT-5 and Claude 4.5 on many benchmarks
4. **Effort Regulation**: Dynamic thinking budgets enable cost-quality optimization
5. **Strategic Control**: Not dependent on Anthropic pricing/availability

### 9.2 Risks of v2.0

1. **Newer Model**: Kimi K2 released Nov 2025, less battle-tested than Claude
2. **Operational Complexity**: Self-hosting requires GPU expertise
3. **Ecosystem Maturity**: Fewer integrations, tools, community resources vs Claude
4. **Uncertain Longevity**: Will Moonshot AI maintain/improve Kimi K2 long-term?

### 9.3 Recommended Path Forward

1. **Immediate**: Answer questions in `questions_for_user.md` + new v2.0 questions
2. **Month 1**: Pilot Kimi K2 API with effort regulation on 10% of traffic
3. **Month 2-3**: If successful, scale to 50% and deploy self-hosted infrastructure
4. **Month 4-6**: Full migration to Kimi K2, optimize effort allocation
5. **Month 7+**: Continuous improvement via ACE learning, monitor for Claude deprecation

---

## 10. References

**v2.0-Specific**:
- Kimi K2 Thinking: https://huggingface.co/moonshotai/Kimi-K2-Thinking
- Moonshot AI Blog: https://kimi-k2.org/blog/15-kimi-k2-thinking-en
- Benchmarks: https://venturebeat.com/ai/moonshots-kimi-k2-thinking-emerges-as-leading-open-source-ai-outperforming
- API Docs: https://platform.moonshot.ai/docs
- OpenRouter Integration: https://openrouter.ai/moonshotai/kimi-k2-thinking

**v1.0 References** (still applicable):
- Neo4j GraphRAG: https://neo4j.com/docs/neo4j-graphrag-python/current/
- ROMA: https://github.com/sentient-agi/ROMA
- ACE: https://arxiv.org/abs/2510.04618
- Ragas: https://docs.ragas.io/
- MCP: https://www.anthropic.com/news/model-context-protocol
- OPA: https://www.openpolicyagent.org/
- Cedar: https://www.cedarpolicy.com/

---

**Document Version**: 2.0
**Author**: Claude (Autonomous AI Architect)
**Date**: 2025-11-08
**Status**: Draft - requires user validation of deployment model and migration strategy
**Supersedes**: architecture.md v1.0
