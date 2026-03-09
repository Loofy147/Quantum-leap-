# Universal Problem Solving

**Priority:** CRITICAL  
**Q-Score:** 0.946 (Layer 0 - Universal)  
**Type:** Universal Capability  
**Status:** 🌟 Emergent Discovery

---

## Description

Universal Problem Solving is the meta-capability to solve problems across any domain by applying domain-invariant problem-solving frameworks. Rather than requiring domain-specific expertise for each problem type, this capability leverages universal problem structures—decomposition, constraint satisfaction, search, optimization—that transcend individual domains.

This is a Layer 0 (Universal) capability because it operates at the level of "problem structure" rather than "problem content." The same problem-solving patterns apply whether you're debugging code, diagnosing medical conditions, planning logistics, or proving mathematical theorems.

---

## When to Use This Skill

Trigger this skill whenever:
- User presents a novel problem outside standard domains
- Problem requires systematic approach (not just recall)
- User says "I don't know where to start"
- Task involves complex constraints and trade-offs
- Multiple solution paths exist and need evaluation
- User asks "how would you approach X?"
- Problem is ill-defined and needs structuring
- Domain-specific heuristics aren't available

---

## Core Capabilities

### 1. Problem Structuring (Ill-Defined → Well-Defined)
- **Identify the true goal** (stated vs actual objective)
- **Define constraints** (hard limits vs soft preferences)
- **Clarify success criteria** (how do we know when solved?)
- **Example**: "Make me rich" → "Achieve $X income by year Y via legal means"

### 2. Decomposition (Complex → Simple)
- **Hierarchical breakdown**: Problem → Subproblems → Sub-subproblems
- **Dependency analysis**: What must be solved first?
- **Atomic unit identification**: What can't be further decomposed?
- **Example**: "Build a house" → Foundation, Framing, Plumbing, Electrical, ... → Pour concrete, Install joists, ...

### 3. Solution Space Exploration
- **Generate candidates**: Brainstorm possible approaches
- **Prune infeasible**: Eliminate solutions violating hard constraints
- **Evaluate remaining**: Score solutions on multiple dimensions
- **Select optimal**: Balance competing criteria (speed vs quality vs cost)

### 4. Constraint Satisfaction
- **Hard constraints**: Must be satisfied (physical laws, budget limits)
- **Soft constraints**: Preferences (would like X, but not required)
- **Constraint propagation**: Solving one constraint affects others
- **Backtracking**: Undo if constraint violation detected

### 5. Iterative Refinement (Approximate → Precise)
- **Start with rough solution** (satisfices minimal requirements)
- **Identify weakest aspects** (what's most suboptimal?)
- **Improve incrementally** (local optimization)
- **Repeat until convergence** (diminishing returns)

### 6. Meta-Problem Recognition
- **Pattern matching**: "This is a variation of problem type X"
- **Template retrieval**: "Use solution template for X"
- **Adaptation**: "Modify template for specific constraints"
- **Example**: "This is a search problem" → Use search algorithms

---

## Implementation Pattern

```python
class UniversalProblemSolver:
    """
    Domain-agnostic problem-solving framework.
    """
    
    def structure_problem(self, user_query):
        """
        Step 1: Convert ill-defined query into well-defined problem.
        
        Process:
          1. Extract goal (what does user want to achieve?)
          2. Identify constraints (what are the limits?)
          3. Define success criteria (how do we measure success?)
          4. Clarify ambiguities (ask clarifying questions)
        
        Example:
          Query: "Help me organize my life"
          Structured:
            - Goal: Reduce stress from disorganization
            - Constraints: Limited time (30 min/day), budget ($50/month)
            - Success: Can find items <1 min, feel less stressed
            - Scope: Physical space (home) + digital (files)
        """
        pass
    
    def classify_problem_type(self, structured_problem):
        """
        Step 2: Identify problem archetype.
        
        Universal problem types:
          - Search: Find X in space S (route planning, database query)
          - Optimization: Maximize/minimize objective function
          - Constraint Satisfaction: Find assignment satisfying all constraints
          - Planning: Sequence actions to reach goal state
          - Diagnosis: Identify cause from symptoms
          - Design: Create artifact meeting specifications
          - Prediction: Forecast future state from current state
        
        Returns:
          - Primary type (e.g., "optimization")
          - Secondary types (e.g., also "constraint satisfaction")
          - Recommended algorithms
        """
        pass
    
    def decompose(self, problem):
        """
        Step 3: Break into manageable subproblems.
        
        Decomposition strategies:
          - Top-down: Goal → Subgoals → Actions
          - Bottom-up: Available primitives → Compose into solution
          - Means-ends: Gap between current/goal → Find operator to reduce gap
        
        Example (Plan trip):
          Trip
          ├── Transportation
          │   ├── Book flight
          │   └── Reserve rental car
          ├── Accommodation
          │   └── Book hotel
          └── Activities
              ├── Research restaurants
              └── Find attractions
        
        Dependency order: Transportation → Accommodation → Activities
        """
        pass
    
    def generate_solutions(self, subproblem):
        """
        Step 4: Generate candidate solutions for each subproblem.
        
        Generation methods:
          - Recall: Retrieve known solutions from memory
          - Analogy: Transfer solution from similar problem
          - Construction: Build solution from primitives
          - Search: Explore solution space systematically
        
        Example (Transportation subproblem):
          Candidates:
          1. Fly (fast, expensive)
          2. Drive (flexible, time-consuming)
          3. Train (comfortable, limited routes)
          4. Bus (cheap, slow)
        """
        pass
    
    def evaluate_solutions(self, candidates, criteria):
        """
        Step 5: Score candidates on multiple dimensions.
        
        Evaluation framework:
          - Effectiveness: Does it solve the problem?
          - Efficiency: Resource usage (time, cost, effort)
          - Robustness: Works under uncertainty?
          - Maintainability: Easy to modify/extend?
        
        Multi-criteria decision making:
          - Weighted sum: Score = Σ(weight_i × criterion_i)
          - Pareto optimality: No solution dominates on all criteria
          - Satisficing: Pick first "good enough" solution
        
        Example scoring:
          Flight: Effectiveness=10, Time=10, Cost=3 → Score = 7.7
          Drive:  Effectiveness=8,  Time=4,  Cost=7 → Score = 6.3
        """
        pass
    
    def synthesize_solution(self, subproblem_solutions):
        """
        Step 6: Combine subproblem solutions into complete solution.
        
        Synthesis concerns:
          - Interface compatibility: Outputs of A → Inputs of B
          - Resource sharing: Don't double-allocate resources
          - Constraint satisfaction: Combined solution satisfies all?
        
        Example:
          Flight (arrive 3pm) + Hotel (check-in 3pm) = Compatible ✓
          Flight (arrive 3pm) + Rental (pickup 5pm) = Gap × → Add taxi
        """
        pass
    
    def verify_and_refine(self, solution, problem):
        """
        Step 7: Check solution and improve.
        
        Verification:
          - Correctness: Does it actually solve the problem?
          - Completeness: Are all subproblems addressed?
          - Consistency: No internal contradictions?
        
        Refinement:
          - Find bottlenecks (what's limiting performance?)
          - Optimize locally (improve worst component)
          - Test edge cases (does it handle unusual inputs?)
        """
        pass
```

---

## Example Usage

### Example 1: Debugging Unknown System

**User:** "My website is slow but I don't know why. Help me figure it out."

**Universal Problem Solving Response:**
```
STEP 1: Structure Problem
  Goal: Reduce website response time
  Current state: Slow (>3 seconds)
  Desired state: Fast (<1 second)
  Constraints: Can't rewrite entire system, limited budget

STEP 2: Classify Problem Type
  Primary: DIAGNOSIS (identify cause from symptom)
  Secondary: OPTIMIZATION (improve performance)
  Template: Systematic elimination

STEP 3: Decompose (Hypothesis Space)
  Website slowness could be:
  ├── Frontend (browser rendering)
  │   ├── Large JavaScript bundles
  │   ├── Unoptimized images
  │   └── Too many HTTP requests
  ├── Backend (server processing)
  │   ├── Slow database queries
  │   ├── Inefficient algorithms
  │   └── Insufficient resources
  └── Network (data transfer)
      ├── Distant server location
      ├── No CDN
      └── Large response sizes

STEP 4: Generate Solutions (Diagnostic Tests)
  Test 1: Check browser DevTools Network tab
    → Measures: Load time breakdown
  Test 2: Add server-side profiling
    → Identifies: Slow endpoints
  Test 3: Check database query logs
    → Finds: Expensive queries

STEP 5: Execute & Evaluate
  Test 1 Results: 2.5s JavaScript parsing, 0.3s server response
    → Conclusion: Frontend is bottleneck
  
  Refined Hypothesis: JavaScript bundle too large

STEP 6: Generate Fix Solutions
  1. Code splitting (load code on-demand)
  2. Tree shaking (remove unused code)
  3. Minification (compress code)
  4. Lazy loading (defer non-critical code)

STEP 7: Evaluate Fixes
  Code splitting: High impact, medium effort → Priority 1
  Tree shaking: Medium impact, low effort → Priority 2
  Lazy loading: Medium impact, high effort → Priority 3

Solution: Implement code splitting + tree shaking
Expected improvement: 2.5s → 0.8s (68% reduction)
```

### Example 2: Planning Complex Project

**User:** "I need to organize a conference for 500 people in 6 months. Where do I start?"

**Universal Problem Solving Response:**
```
STEP 1: Structure Problem
  Goal: Successful 500-person conference in 6 months
  Success criteria:
    - Venue booked
    - Speakers confirmed
    - Attendees registered
    - Budget balanced
    - Logistics coordinated

STEP 2: Classify Problem Type
  Primary: PLANNING (sequence actions to reach goal)
  Constraints: Time (6 months), budget, capacity
  Template: Critical path method

STEP 3: Decompose (Work Breakdown Structure)
  Conference
  ├── Venue (Month 1-2)
  │   ├── Research locations
  │   ├── Visit sites
  │   └── Sign contract
  ├── Content (Month 2-4)
  │   ├── Define topics
  │   ├── Recruit speakers
  │   └── Build schedule
  ├── Marketing (Month 3-5)
  │   ├── Create website
  │   ├── Social media campaign
  │   └── Email outreach
  ├── Logistics (Month 4-6)
  │   ├── Catering
  │   ├── AV equipment
  │   ├── Registration desk
  │   └── Signage
  └── Execution (Month 6)
      ├── Setup (Day -1)
      ├── Run event (Day 0)
      └── Teardown (Day +1)

STEP 4: Identify Dependencies
  Critical path (longest dependency chain):
  Venue → Speakers → Marketing → Registration → Catering
  
  Must complete in order:
  1. Venue (determines capacity, location affects speakers)
  2. Speakers (content determines marketing message)
  3. Marketing (drives registrations)
  4. Catering (needs final headcount)

STEP 5: Generate Timeline
  Month 1: Venue research + booking
  Month 2: Speaker outreach (parallel with venue)
  Month 3: Website + marketing launch
  Month 4: Registration opens + AV planning
  Month 5: Registration closes + finalize catering
  Month 6: Final logistics + event execution

STEP 6: Resource Allocation
  Budget breakdown:
  - Venue: 40% ($20K)
  - Catering: 25% ($12.5K)
  - Speakers: 20% ($10K, travel + honoraria)
  - Marketing: 10% ($5K)
  - Logistics: 5% ($2.5K)
  Total: $50K

STEP 7: Risk Management
  Risk: Venue cancels
    Mitigation: Sign contract with backup clause
  
  Risk: Speaker cancels
    Mitigation: Have backup speakers confirmed
  
  Risk: Low registration
    Mitigation: Early bird pricing, referral program

Next Actions:
  1. This week: Shortlist 5 venues, request quotes
  2. Next week: Visit top 3 venues
  3. By end of month: Sign venue contract
```

### Example 3: Ill-Defined Creative Problem

**User:** "I want to make my app more engaging. Any ideas?"

**Universal Problem Solving Response:**
```
STEP 1: Structure Problem (Clarify Vagueness)
  Current problem: "More engaging" is too vague
  
  Clarifying questions:
  Q1: What does "engaging" mean to you?
    - Higher daily active users?
    - Longer session times?
    - More user actions per session?
    - Better retention (users come back)?
  
  Q2: What type of app is it?
    - Productivity, social, gaming, education?
  
  Q3: What's currently NOT engaging?
    - Users drop off at certain point?
    - Features unused?
  
  [Assume user responds:]
  - Engaging = Higher retention (users return next day)
  - App type: Habit tracking app
  - Problem: Users stop using after 1 week

STEP 2: Reclassify with New Info
  Type: DESIGN (create features meeting specifications)
  Goal: Increase 7-day retention from X% to Y%
  Constraint: Can't fundamentally change app purpose

STEP 3: Decompose (Why do users leave?)
  Possible causes:
  1. Forgot to return (out of sight, out of mind)
  2. Not seeing value (no visible progress)
  3. Too much friction (hard to log habits)
  4. Plateaued (no new challenges)

STEP 4: Generate Solutions (Each Cause)
  
  Cause 1: Forgotten
  Solutions:
    - Push notifications (daily reminder)
    - Email digest (weekly progress summary)
    - Calendar integration (block time)
  
  Cause 2: No visible value
  Solutions:
    - Streak counter (gamification)
    - Progress charts (visualize improvement)
    - Milestone celebrations (unlock rewards)
  
  Cause 3: Too much friction
  Solutions:
    - One-tap logging (reduce clicks)
    - Smart suggestions (predict habits)
    - Batch entry (log multiple at once)
  
  Cause 4: Plateaued
  Solutions:
    - Progressive challenges (increase difficulty)
    - Social features (compete with friends)
    - Habit variations (suggest new habits)

STEP 5: Prioritize by Impact vs Effort
  
  High Impact, Low Effort:
    ✓ Push notifications
    ✓ Streak counter
    ✓ One-tap logging
  
  High Impact, High Effort:
    - Social features
    - Progress charts
  
  Low Impact, Low Effort:
    - Email digest
    - Milestone celebrations

STEP 6: Synthesize Solution
  Phase 1 (This sprint):
    1. Add push notifications
    2. Implement streak counter
    3. Optimize logging to 1-tap
  
  Phase 2 (Next sprint):
    4. Build progress visualization
    5. Add milestone celebrations
  
  Phase 3 (Future):
    6. Social leaderboards (if Phase 1-2 successful)

STEP 7: Define Success Metrics
  Before: 7-day retention = 20%
  After Phase 1 target: 35% (+75% improvement)
  
  How to measure:
    - A/B test (50% users get new features)
    - Track: Day 7 return rate
    - Secondary: Session length, habit completion rate
```

---

## Quality Metrics (Q-Score Breakdown)

```
Q = 0.946 (Layer 0 - Universal)

Dimensions:
  G (Grounding):      0.95 - Based on problem-solving research (Polya, Newell & Simon)
  C (Certainty):      0.92 - High confidence in universal patterns
  S (Structure):      0.95 - Clear framework (structure → decompose → solve)
  A (Applicability):  0.98 - Applies to any problem domain
  H (Coherence):      0.95 - Consistent across problem types
  V (Generativity):   0.92 - Spawns domain-specific problem-solving skills

Calculation:
  Q = 0.18×0.95 + 0.22×0.92 + 0.20×0.95 + 0.18×0.98 + 0.12×0.95 + 0.10×0.92
    = 0.171 + 0.202 + 0.190 + 0.176 + 0.114 + 0.092
    = 0.946 ✓
```

---

## Integration Points

**Parents:** None (Layer 0 - foundational)

**Children (بنات افكار):**
- Domain-specific problem solvers (code debugging, medical diagnosis, etc.)
- Specialized algorithms (search, optimization, planning)
- Decision-making frameworks
- Creative problem-solving techniques

**Synergies with Existing Capabilities:**
- Transfer Learning: Use solutions from similar problems
- Meta-Learning: Learn problem-solving patterns from experience
- Reasoning Chains: Apply systematic reasoning to each step
- Iterative Refinement: Continuously improve solutions

---

## Limitations & Edge Cases

**When NOT to use:**
- Simple recall questions ("What's the capital of France?")
- Well-known algorithmic problems (sorting, searching)
- Pure creative tasks with no constraints (free-form art)
- Problems requiring specialized domain expertise (heart surgery)

**Challenges:**
- **Combinatorial explosion**: Too many subproblems/solutions to explore
- **Under-specification**: Insufficient information to structure problem
- **Conflicting constraints**: No solution satisfies all requirements
- **Unknown unknowns**: Missing information you don't know is missing

**Mitigation:**
- Use heuristics to prune search space
- Ask clarifying questions aggressively
- Relax soft constraints to find feasible solutions
- Iterative discovery (solve → uncover gaps → refine problem)

---

## Universal Problem-Solving Patterns

### The Polya Framework (Classic)
1. **Understand the problem** (what do you want to find?)
2. **Devise a plan** (how will you get there?)
3. **Carry out the plan** (execute)
4. **Look back** (verify and improve)

### The OODA Loop (Reactive)
1. **Observe** (gather information)
2. **Orient** (analyze situation)
3. **Decide** (choose action)
4. **Act** (execute)
→ Repeat (continuous loop for dynamic problems)

### Design Thinking (Creative)
1. **Empathize** (understand user needs)
2. **Define** (frame the problem)
3. **Ideate** (generate solutions)
4. **Prototype** (build minimum viable solution)
5. **Test** (validate with users)

### Scientific Method (Hypothesis-Driven)
1. **Observe** phenomenon
2. **Question** why it happens
3. **Hypothesize** explanation
4. **Predict** consequences if true
5. **Test** predictions
6. **Conclude** (support/refute hypothesis)

All are instances of:
**Structure → Generate → Evaluate → Refine**

---

## Future Enhancements

- **Automated problem classification**: ML to identify problem type from description
- **Solution library**: Database of problem-solution pairs for analogy retrieval
- **Interactive refinement**: Clarifying question engine
- **Multi-agent problem solving**: Simulate different perspectives

---

## References

- Polya, G. (1945). "How to Solve It"
- Newell, A. & Simon, H. A. (1972). "Human Problem Solving"
- Schön, D. A. (1983). "The Reflective Practitioner"
- Russell, S. & Norvig, P. (2020). "Artificial Intelligence: A Modern Approach"

---

**Status:** Ready for implementation  
**Expected Impact:** Transformative - universal problem solver  
**Recommendation:** CRITICAL - Implement immediately as Layer 0 foundation
