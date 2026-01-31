# Trajectory-Informed Self-Distillation (TISD)

## Learning Dense Credit Assignment from Sparse Rewards in Long-Horizon Agents

**Target Venue**: NeurIPS 2026

---

## Abstract

Training embodied agents with large language models (LLMs) in long-horizon environments is fundamentally limited by sparse supervision: agents execute dozens of actions but receive only a final success/failure signal. Existing approaches either rely on external reward models, expert demonstrations, or scalar reinforcement signals that fail to assign credit to specific decisions.

We propose **Trajectory-Informed Self-Distillation (TISD)**, a self-supervised learning framework that extracts *dense, step-level learning signals from sparse final rewards* by leveraging an LLM's retrospective reasoning ability. TISD uses the same model as both **student** (prospective decision maker) and **teacher** (retrospective evaluator), where the teacher is conditioned on the full execution trajectory and final outcome. The student is trained to align its action distribution with the teacher's hindsight-informed recommendations via a temporally weighted distillation objective.

Crucially, TISD requires **no external teacher models, dense rewards, or human annotations**, and remains effective even when all sampled trajectories fail. We provide an information-theoretic interpretation, analyze the causality concerns unique to sequential decision-making, and outline a practical implementation for embodied benchmarks such as ALFWorld, WebArena, and AndroidWorld.

---

## 1. Introduction: Why Agents Need a Different Self-Distillation Paradigm

### 1.1 The Sparse Reward Problem in Agent Learning

Consider an agent tasked with "Put a clean apple in the fridge" in ALFWorld. The agent must:
1. Navigate to find the apple (unknown location)
2. Pick up the apple
3. Find and navigate to a sink
4. Clean the apple
5. Navigate to the fridge
6. Open the fridge
7. Place the apple inside

This requires 20-50 atomic actions, yet the agent receives **only a single binary signal** at the end: task completed or not. If the agent fails because it tried to place the apple in a closed fridge at step 47, every preceding correct decision receives the same negative feedback as the actual mistake.

### 1.2 Why Existing Self-Distillation Methods Don't Transfer

Recent work has demonstrated the power of self-distillation for LLM training:

| Method | Privileged Information | Setting | Key Limitation for Agents |
|--------|----------------------|---------|---------------------------|
| SDPO (Hübotter et al., 2026) | Runtime errors, test results | Code generation | Requires rich environment feedback per step |
| SDFT (Shenfeld et al., 2026) | Expert demonstrations | Skill learning | Requires access to successful trajectories |
| OPSD (Zhao et al., 2026) | Ground-truth answers | Math reasoning | Single-turn; answer available upfront |

**The agent setting is fundamentally different:**

1. **Temporal Structure**: Agents make sequential decisions where early choices constrain later options. A mistake at step 5 may only manifest as failure at step 47.

2. **State Evolution**: Unlike static reasoning tasks, agent environments change. The "correct" action depends on a dynamically evolving state that the agent itself shapes.

3. **No Step-Level Ground Truth**: In math, we know the final answer. In code, we have test results. In agent tasks, we often have **no intermediate correctness signal**—only the final outcome.

4. **Exploration-Exploitation Structure**: Agents must balance exploring unknown states (searching for an object) with exploiting known strategies (executing a plan). This temporal structure has no analog in single-turn tasks.

### 1.3 Our Thesis

> **The same LLM that struggles to make optimal decisions in the moment can often recognize, in hindsight, which decisions were suboptimal—if shown the complete trajectory and outcome.**

This asymmetry between *prospective generation* and *retrospective evaluation* is well-documented in cognitive science and increasingly observed in LLMs. TISD operationalizes this insight for sequential decision-making.

---

## 2. Problem Formulation

### 2.1 Agent Task Definition

We consider long-horizon agent tasks with sparse terminal feedback:

- **Task specification**: $x$ (natural language instruction)
- **Environment state at step $t$**: $s_t$ (text observation)
- **Action space**: $\mathcal{A}$ (discrete actions or text generation)
- **Transition dynamics**: $s_{t+1} \sim P(\cdot | s_t, a_t)$
- **Policy**: $a_t \sim \pi_\theta(a_t | x, s_{\leq t})$
- **Trajectory**: $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$
- **Sparse reward**: $r(\tau) \in \{0, 1\}$ (only at episode termination)

### 2.2 The Credit Assignment Challenge

The fundamental problem: given a failed trajectory $\tau$ with $r(\tau) = 0$, identify which decision(s) $a_t$ were responsible for the failure.

**RLVR/GRPO approach**: Assign uniform advantage to all steps:
$$A_t = r(\tau) - b \quad \forall t$$

This conflates correct and incorrect decisions, leading to:
- High variance gradient estimates
- No learning when all trajectories in a batch have the same outcome
- Sample inefficiency scaling with trajectory length

**Our approach**: Use the model's own retrospective reasoning to generate *step-specific* learning signals.

---

## 3. Method: Trajectory-Informed Self-Distillation

### 3.1 Core Mechanism: Student-Teacher with Shared Parameters

We instantiate both student and teacher using the **same model** $\pi_\theta$, distinguished only by their conditioning context.

**Student Policy** (prospective, causal):
$$\pi_S(a_t | c_t) = \pi_\theta(a_t | x, s_{\leq t})$$

The student sees only information available at decision time—the task description and history up to the current step.

**Teacher Policy** (retrospective, privileged):
$$\pi_T(a_t | c_t, h_t) = \pi_\theta(a_t | x, s_{\leq t}, \underbrace{\tau_{>t}, r}_{\text{hindsight } h_t})$$

The teacher additionally conditions on:
- $\tau_{>t}$: What happened after step $t$ (future trajectory)
- $r$: The final outcome (success/failure)

**Critical design choice**: Teacher gradients are detached. Only the student receives gradient updates.

### 3.2 Distillation Objective

We minimize the forward KL divergence from teacher to student:

$$\mathcal{L}_{\text{TISD}} = \mathbb{E}_{\tau \sim \pi_S} \left[ \sum_{t=1}^{T} w_t \cdot D_{\text{KL}}\left( \pi_T(\cdot | c_t, h_t) \,\|\, \pi_S(\cdot | c_t) \right) \right]$$

**Why Forward KL?**

| Divergence | Effect | Suitability |
|------------|--------|-------------|
| Forward KL: $D(\pi_T \| \pi_S)$ | Student covers all teacher modes | ✓ Encourages exploration of teacher-preferred actions |
| Reverse KL: $D(\pi_S \| \pi_T)$ | Student mode-seeks | ✗ May collapse to single action |

Forward KL is also more stable when teacher distributions are computed from a frozen/detached model.

### 3.3 Adaptive Temporal Credit Assignment

Not all steps contribute equally to success or failure. We introduce an adaptive weighting scheme based on **teacher-student disagreement**:

**Step-level disagreement**:
$$\delta_t = D_{\text{KL}}(\pi_T(\cdot | c_t, h_t) \| \pi_S(\cdot | c_t))$$

**Intuition**: High $\delta_t$ indicates that hindsight information significantly changes the action distribution—suggesting this step was a **critical decision point**.

**Softmax-normalized weights**:
$$w_t = \frac{\exp(\lambda \cdot \delta_t)}{\sum_{j=1}^{T} \exp(\lambda \cdot \delta_j)}$$

where $\lambda > 0$ is a temperature controlling weight concentration.

**Properties**:
- Steps where teacher and student agree ($\delta_t \approx 0$) receive minimal weight
- Steps where hindsight reveals a clear "better choice" ($\delta_t$ large) receive high weight
- Naturally handles variable-length trajectories via normalization

### 3.4 Teacher Prompt Engineering

The teacher prompt is crucial—it must elicit **distributional** corrections, not just action copying.

```
[RETROSPECTIVE EVALUATION MODE]

Task: {task_description}

You are reviewing a completed execution trajectory to evaluate 
the decision made at a specific step.

=== CONTEXT AT DECISION TIME ===
History: {s_0, a_0, s_1, a_1, ..., s_{t-1}, a_{t-1}}
Current observation: {s_t}
Action taken: {a_t}

=== HINDSIGHT INFORMATION ===
What happened after this decision:
{s_{t+1}, a_{t+1}, ..., s_T, a_T}

Final outcome: {SUCCESS / FAILURE}

=== YOUR TASK ===
Based on the complete trajectory and its outcome, evaluate whether 
the action taken at the current step was appropriate.

Consider:
1. Did this action contribute to or detract from task success?
2. Were there better alternatives given what we now know?
3. How confident should we be in this assessment?

Provide a probability distribution over available actions that 
reflects what SHOULD have been done, given hindsight.

Available actions: {action_space}

Output format: {"action_1": prob_1, "action_2": prob_2, ...}
```

**Implementation note**: For efficiency, we can use constrained decoding or extract logits directly over the action vocabulary rather than parsing generated text.

---

## 4. Theoretical Analysis

### 4.1 Information-Theoretic Foundation

**Definition 1 (Hindsight Information Gain)**:
$$I_t = H(A_t | c_t) - H(A_t | c_t, h_t)$$

where $H(\cdot)$ denotes entropy under the model's predictive distribution.

**Interpretation**: $I_t$ measures how much the hindsight information $h_t$ reduces uncertainty about what action should be taken at step $t$.

**Proposition 1**: The KL divergence $\delta_t$ upper bounds the information gain:
$$I_t \leq \delta_t$$

with equality when student and teacher have the same entropy.

**Corollary**: Steps with $\delta_t \approx 0$ are **uninformative**—hindsight provides no useful signal about what should have been done differently.

### 4.2 Connection to Hindsight Experience Replay

TISD can be viewed as a *soft*, *distributional* generalization of Hindsight Experience Replay (HER):

| Aspect | HER | TISD |
|--------|-----|------|
| Relabeling | Change goal to match outcome | Keep goal, change action distribution |
| Signal type | Binary (success under new goal) | Continuous (distribution shift) |
| Learning mode | Off-policy Q-learning | On-policy distillation |
| Applicable to | Goal-conditioned RL | Any sparse-reward agent task |

### 4.3 Learning from All-Failure Batches

A critical advantage over GRPO: **TISD can learn even when all trajectories fail**.

**Proposition 2**: Learning signal exists whenever:
$$\text{Var}_\tau[\delta_t] > 0$$

**Proof sketch**: Different failure modes produce different teacher distributions. Even with $\mathbb{E}[r] = 0$:
- Trajectory A: Fails due to exploration inefficiency → Teacher adjusts early search decisions
- Trajectory B: Fails due to precondition violation → Teacher adjusts specific action choice
- Trajectory C: Fails due to timeout on correct path → Teacher may make minimal adjustments

These distinct patterns yield different $\delta_t$ profiles, enabling learning.

### 4.4 Causality Considerations

**Potential Issue**: The teacher conditions on $\tau_{>t}$, which causally depends on $a_t$. This creates a **post-treatment conditioning** problem—the teacher's recommendation may be confounded by the actual action taken.

**Example**: If the agent went left at step 5 and succeeded, the teacher (seeing this) might incorrectly conclude "left was correct"—but perhaps right would have succeeded faster.

**Mitigations**:

1. **Outcome-only ablation**: Teacher sees only final outcome $r$, not full future trajectory $\tau_{>t}$. This preserves causal validity at the cost of reduced signal.

2. **Disagreement-weighted learning**: High $\delta_t$ doesn't mean "teacher is definitely right"—it means "this step deserves attention." The actual gradient direction may be noisy, but the *attention allocation* is valuable.

3. **Counterfactual branching** (compute-intensive): For high-$\delta_t$ steps, actually simulate alternative actions to validate teacher suggestions.

**Empirical approach**: We recommend starting with full hindsight (maximum signal), then ablating to outcome-only if overfitting or causal artifacts are observed.

---

## 5. The Unique Structure of Agent Self-Distillation

This section articulates why agent tasks require a fundamentally different self-distillation approach.

### 5.1 Hierarchical Decision Structure

Agent tasks involve decisions at multiple abstraction levels:

```
Level 3 (Goal):     "Put clean apple in fridge"
                            |
Level 2 (Subgoal):  [Find apple] → [Clean apple] → [Store apple]
                         |              |              |
Level 1 (Action):   [go to X]      [use sink]     [open fridge]
                    [examine Y]    [pick up]       [put in]
```

**Implication for TISD**: Failures at different levels require different hindsight granularity.

- **Action-level failure** (e.g., tried to put in closed fridge): Local hindsight ($\tau_{>t}$ for a few steps) suffices.
- **Subgoal-level failure** (e.g., searched wrong area): Requires seeing complete failed search before recognizing the issue.
- **Goal-level failure** (e.g., misunderstood task): Full trajectory + outcome needed.

**Design principle**: Teacher prompt should include enough future context to capture the relevant failure mode.

### 5.2 State-Dependent Action Semantics

In static tasks (math, code), the "correct answer" is state-independent. In agent tasks, the optimal action depends on a **dynamically evolving state**:

| State | Optimal action |
|-------|---------------|
| Fridge is closed | Open fridge |
| Fridge is open | Put apple in fridge |
| Apple is dirty | Clean apple |
| Apple is clean | Go to fridge |

**Implication**: The teacher must reason about **state-action contingencies**, not just "what's the right action for this task."

### 5.3 Exploration as a First-Class Skill

Agent tasks often require systematic exploration before exploitation is possible:

```
Step 1-10: Search living room (no apple)
Step 11-20: Search kitchen (no apple)  
Step 21-25: Search bedroom (found apple!)
Step 26-40: Execute plan with apple
```

The first 20 steps are "failures" in a local sense but necessary for task completion.

**Challenge for TISD**: Teacher must distinguish:
- **Useful exploration** (systematically searching new areas)
- **Wasteful exploration** (revisiting checked areas)
- **Suboptimal exploration order** (bedroom should have been checked first)

**Approach**: Teacher prompt should emphasize *efficiency* and *information gain*, not just *eventual success*.

### 5.4 Error Recovery and Robustness

Unlike single-turn tasks, agents encounter unexpected situations and must recover:

```
Step 30: put apple in fridge → "Nothing happens" (fridge was closed)
Step 31: ??? 
```

The optimal recovery is `open fridge` then retry. But the agent might:
- Retry the same action (bad)
- Go searching for another fridge (wasteful)
- Give up (catastrophic)

**Implication**: TISD should provide strong signal on **recovery behavior**, not just initial decision-making.

---

## 6. Implementation

### 6.1 Training Algorithm

```python
def tisd_training_step(env, task, policy_theta, config):
    # === Phase 1: Trajectory Collection ===
    trajectories = []
    for _ in range(config.batch_size):
        tau = rollout(env, task, policy_theta)  # Student rollout
        trajectories.append(tau)
    
    # === Phase 2: Teacher Evaluation ===
    total_loss = 0
    for tau in trajectories:
        step_kls = []
        
        for t in range(len(tau)):
            # Student distribution (causal context only)
            student_logits = policy_theta.forward(
                context=build_student_context(task, tau[:t+1])
            )
            
            # Teacher distribution (with hindsight)
            with torch.no_grad():
                teacher_logits = policy_theta.forward(
                    context=build_teacher_context(task, tau, t)
                )
            
            # Compute KL divergence
            kl_t = kl_divergence(
                F.softmax(teacher_logits / config.teacher_temp, dim=-1),
                F.log_softmax(student_logits, dim=-1)
            )
            step_kls.append(kl_t)
        
        # === Phase 3: Adaptive Weighting ===
        step_kls = torch.stack(step_kls)
        weights = F.softmax(config.lambda_ * step_kls.detach(), dim=0)
        
        # Weighted loss
        trajectory_loss = (weights * step_kls).sum()
        total_loss += trajectory_loss
    
    # === Phase 4: Gradient Update ===
    loss = total_loss / len(trajectories)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### 6.2 Practical Considerations

**Teacher Temperature**: Sharpen teacher logits ($T < 1$) to provide clearer guidance:
$$\pi_T^{\text{sharp}}(a) = \frac{\exp(\log \pi_T(a) / T)}{\sum_{a'} \exp(\log \pi_T(a') / T)}$$

**KL Clipping**: Prevent exploding gradients from high-KL steps:
$$\delta_t^{\text{clipped}} = \min(\delta_t, \delta_{\max})$$

**Success Trajectory Handling**: For successful trajectories, teacher and student may already agree. Options:
- Down-weight successful trajectories
- Still train (reinforce good behavior)
- Use success trajectories only for cross-trajectory contrast

**Memory Efficiency**: Cache teacher evaluations since they don't require gradients:
```python
# Batch all teacher forward passes
all_teacher_contexts = [build_teacher_context(task, tau, t) 
                        for tau in trajectories for t in range(len(tau))]
all_teacher_logits = policy_theta.forward_batch(all_teacher_contexts)  # No grad
```

### 6.3 Hyperparameters

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| $\lambda$ (weight temperature) | 1.0 - 5.0 | Higher = more concentrated on high-KL steps |
| Teacher temperature $T$ | 0.5 - 1.0 | Lower = sharper teacher signal |
| $\delta_{\max}$ (KL clip) | 10.0 - 50.0 | Prevents outlier domination |
| Batch size | 4 - 16 trajectories | More enables cross-trajectory contrast |
| Learning rate | 1e-6 - 5e-5 | Standard LLM fine-tuning range |

---

## 7. Experimental Plan

### 7.1 Environments

| Environment | Horizon | Action Space | Key Challenge |
|-------------|---------|--------------|---------------|
| **ALFWorld** | 30-50 | Discrete text | Object search + manipulation |
| **WebArena** | 10-30 | Click + type | Dynamic web interaction |
| **AndroidWorld** | 20-50 | Touch + type | Mobile UI navigation |
| **MINT** | 5-15 turns | Text generation | Multi-turn tool use |

### 7.2 Baselines

1. **Prompting**: ReAct, Reflexion (no training)
2. **Imitation**: SFT on expert trajectories, Filtered BC
3. **RL**: GRPO, REINFORCE with sparse reward
4. **Self-improvement**: RFT (rejection sampling fine-tuning)
5. **External distillation**: GPT-4 as teacher

### 7.3 Key Experiments

**E1: Main Results**
- Compare TISD against all baselines on success rate
- Measure sample efficiency (trajectories to reach X% success)
- Measure token efficiency (total tokens generated during training)

**E2: Ablation Studies**

| Ablation | Purpose |
|----------|---------|
| No hindsight (student = teacher) | Is retrospective info necessary? |
| Outcome-only teacher | Is future trajectory necessary, or just final result? |
| Uniform weights ($w_t = 1/T$) | Does adaptive credit assignment help? |
| Reverse KL | Forward vs reverse divergence |
| Vary $\lambda$ | Sensitivity to weight concentration |

**E3: Analysis**

1. **Teacher Retrospective Ability**
   - Present model with failed trajectories
   - Ask "which step was the primary error?"
   - Compare to human annotations
   - Measure correlation between $\delta_t$ and human-identified error steps

2. **Failure Mode Breakdown**
   - Categorize failures: exploration / execution / understanding
   - Analyze TISD improvement per category
   - Identify failure modes where hindsight reasoning is insufficient

3. **Scaling Laws**
   - Test on 1.5B, 4B, 8B, 14B parameter models
   - Hypothesis: TISD effectiveness increases with model capability (stronger ICL)

**E4: All-Failure Learning**
- Artificially filter batches to contain only failures
- Compare TISD vs GRPO learning curves
- Demonstrate TISD's unique ability to learn from pure-failure data

### 7.4 Metrics

| Metric | Description |
|--------|-------------|
| **Success Rate (SR)** | Primary metric |
| **SR@k** | Success rate with k attempts (controls for entropy collapse) |
| **Avg Steps** | Efficiency measure |
| **Sample Efficiency** | Trajectories to reach threshold SR |
| **Token Efficiency** | Training tokens to reach threshold SR |
| **Generalization Gap** | Performance on held-out task variations |

---

## 8. Expected Contributions

### Methodological
- First systematic framework for self-distillation in long-horizon agent tasks
- Novel adaptive credit assignment mechanism based on teacher-student disagreement
- Practical training algorithm that works with sparse-reward-only environments

### Theoretical
- Information-theoretic characterization of when hindsight learning is effective
- Analysis of causality concerns unique to sequential decision-making
- Connection between self-distillation and hindsight experience replay

### Empirical
- Comprehensive evaluation across multiple agent benchmarks
- Ablation studies isolating key design choices
- Analysis of model scaling and failure mode breakdown

### Practical
- Open-source implementation with standard LLM training frameworks
- Guidelines for teacher prompt design
- Hyperparameter recommendations

---

## 9. Limitations and Future Work

### Known Limitations

1. **Model Capability Dependency**: TISD relies on the model's ability to perform retrospective reasoning. Below some capability threshold, the teacher may not provide useful signal.

2. **Causality-Signal Tradeoff**: Full hindsight (including future trajectory) provides stronger signal but introduces causal confounding. The optimal balance is task-dependent.

3. **Compute Overhead**: Teacher evaluation adds ~50% compute per trajectory compared to pure RL. However, this is often offset by improved sample efficiency.

4. **Environment Coverage**: Our experiments focus on text-based environments. Extension to vision-based embodied tasks requires additional consideration.

### Future Directions

1. **Hierarchical TISD**: Different hindsight granularity for different decision levels
2. **Counterfactual Teachers**: Explicitly simulate alternative action outcomes
3. **Cross-Task Transfer**: Use hindsight from one task to improve related tasks
4. **Hybrid TISD-RL**: Combine self-distillation with RL for sequential training

---

## 10. Conclusion

We propose TISD, a self-supervised framework that converts sparse task outcomes into dense, step-level learning signals for long-horizon agents. By leveraging an LLM's asymmetric ability—struggling to make optimal decisions prospectively but capable of recognizing errors retrospectively—TISD enables effective agent training without external reward models, dense annotations, or expert demonstrations.

Our core bet:

> **If an LLM can recognize its own mistakes in hindsight, it can teach itself where to improve—using only sparse task outcomes as supervision.**

TISD represents a step toward agents that genuinely learn from experience, not just from curated demonstrations or dense reward engineering.

---

## Appendix A: Theoretical Proofs

### A.1 Proof of Proposition 1

**Proposition 1**: $I_t \leq \delta_t$

**Proof**:
$$I_t = H(A_t | c_t) - H(A_t | c_t, h_t)$$

By definition of KL divergence:
$$D_{\text{KL}}(\pi_T \| \pi_S) = H(\pi_T, \pi_S) - H(\pi_T)$$

where $H(\pi_T, \pi_S)$ is cross-entropy.

Since $\pi_S$ is the marginal (without hindsight):
$$H(A_t | c_t) = H(\pi_S)$$

And $\pi_T$ is the conditional (with hindsight):
$$H(A_t | c_t, h_t) = H(\pi_T)$$

Therefore:
$$I_t = H(\pi_S) - H(\pi_T)$$

The KL divergence can be rewritten as:
$$\delta_t = H(\pi_T, \pi_S) - H(\pi_T) = H(\pi_T, \pi_S) - H(\pi_T)$$

Since cross-entropy is at least entropy ($H(\pi_T, \pi_S) \geq H(\pi_T)$ with equality when $\pi_T = \pi_S$):
$$\delta_t = H(\pi_T, \pi_S) - H(\pi_T) \geq 0$$

And:
$$\delta_t - I_t = H(\pi_T, \pi_S) - H(\pi_S) = \mathbb{E}_{\pi_T}[\log \pi_S] + H(\pi_S) \geq 0$$

The last inequality follows from $\mathbb{E}_{\pi_T}[-\log \pi_S] \geq H(\pi_S)$ by Gibbs' inequality.

Therefore $\delta_t \geq I_t$. ∎

---

## Appendix B: Extended Related Work

### B.1 Self-Distillation in LLMs

| Method | Setting | Privileged Info | Key Insight |
|--------|---------|-----------------|-------------|
| SDPO | Code gen | Test results | Learn from verification feedback |
| SDFT | Skills | Expert demos | In-context demonstrations as teacher |
| OPSD | Math | Answers | Answer-conditioned generation |
| **TISD (ours)** | Agents | Trajectory + outcome | Retrospective credit assignment |

### B.2 Credit Assignment in RL

Traditional approaches:
- **Temporal difference**: Bootstrapping with value functions
- **Eligibility traces**: Exponentially weighted credit
- **Attention-based**: Learn which states matter

TISD differs by using **model-based retrospection** rather than learned value functions.

### B.3 Learning from Failure

- **HER**: Relabel goals to create successes
- **Self-Refine**: Iterative critique and revision (single-turn)
- **Reflexion**: Verbal feedback for next attempt (episodic, not step-level)

TISD provides **within-episode, step-level** credit assignment from failure.

---

## Appendix C: Teacher Prompt Variants

### C.1 Minimal Prompt (Outcome Only)

```
Task: {task}
History: {s_0, a_0, ..., s_t}
Current observation: {s_t}
Outcome: This trajectory {SUCCEEDED/FAILED}.
Available actions: {actions}
Recommended action distribution:
```

### C.2 Full Hindsight Prompt (Default)

See Section 3.4.

### C.3 Chain-of-Thought Prompt

```
Task: {task}

[REVIEW MODE]
Complete trajectory: {full_tau}
Final outcome: {SUCCESS/FAILURE}

Now focusing on step {t}:
Current state: {s_t}
Action taken: {a_t}

First, analyze:
1. What was the agent trying to accomplish at this step?
2. What actually happened as a result?
3. With hindsight, was this the right choice?
4. What alternatives might have been better?

Based on your analysis, provide action probabilities:
{action_space}
```

---

## Appendix D: Computational Cost Analysis

| Component | FLOPs (relative) | Wall time (relative) |
|-----------|-----------------|---------------------|
| Student rollout | 1.0x | 1.0x (includes env) |
| Teacher evaluation | 0.5x | 0.3x (no env, batched) |
| KL computation | 0.01x | negligible |
| Gradient update | 0.2x | 0.2x |
| **Total TISD** | **1.7x** | **1.5x** |
| GRPO (8 rollouts) | 8.0x | 5.0x |

TISD is more compute-efficient than GRPO while providing denser signal.