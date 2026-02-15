---
name: skill-creator
description: Autonomous skill generation and improvement system. Generates new skills from natural language descriptions WITH MANDATORY USER APPROVAL, validates structure, manages versions, and suggests improvements to existing skills.
---

# Skill Creator Skill

The **Skill Creator** enables Optimus Pryme to expand its own capabilities by generating new skills. **CRITICAL**: All skill generation requires explicit user approval before creation.

## Core Principle

> **MANDATORY APPROVAL WORKFLOW**
> 
> Generate → Validate → Present to User → Wait for Approval → Create
> 
> **NO AUTONOMOUS SKILL CREATION WITHOUT USER CONSENT**

## Core Capabilities

### 1. **Skill Generation from Description**
- Parse natural language skill requirements
- Generate SKILL.md with proper YAML frontmatter
- Create basic script templates
- Set up directory structure
- **ALWAYS** present to user for approval first

### 2. **Skill Variation Creation**
- Clone existing skills with modifications
- Adapt skills for different use cases
- Create specialized variants
- **Requires user review** before creation

### 3. **Skill Validation**
- Check SKILL.md format and completeness
- Validate YAML frontmatter
- Ensure required sections present
- Verify file structure compliance

### 4. **Version Management**
- Track skill versions over time
- Document changes between versions
- Rollback capabilities
- Performance comparison across versions

### 5. **Skill Improvement Suggestions**
- Analyze existing skills for gaps
- Suggest enhancements based on usage patterns
- Recommend consolidation of similar skills
- Identify unused or redundant capabilities

## Approval Workflow

### Step 1: Generate (Internal)
```
User Request: "Create a skill for inventory management"

SKILL CREATOR (Internal):
1. Parse requirements
2. Generate skill structure
3. Create SKILL.md draft
4. Validate structure
5. Prepare for user presentation
```

### Step 2: Present to User
```json
{
  "proposed_skill": {
    "name": "inventory-manager",
    "description": "Track inventory levels, predict stockouts, optimize reorder points",
    "capabilities": [
      "Real-time inventory tracking",
      "Stockout risk prediction",
      "Reorder point calculation",
      "FBA fee optimization"
    ],
    "files_to_create": [
      ".agent/skills/inventory-manager/SKILL.md",
      ".agent/skills/inventory-manager/scripts/inventory_tracker.py",
      ".agent/skills/inventory-manager/scripts/demand_forecaster.py"
    ]
  },
  "validation_status": "passed",
  "estimated_value": "high",
  "recommendation": "This skill would complement existing market-researcher and grok-admaster-operator skills"
}
```

### Step 3: Wait for User Approval
```
SYSTEM: Present proposed skill to user
WAIT: User reviews and approves/rejects/modifies
IF APPROVED: Proceed to creation
IF REJECTED: Abandon or revise
IF MODIFIED: Regenerate with changes, present again
```

### Step 4: Create (Only After Approval)
```
USER APPROVED ✓

SKILL CREATOR:
1. Create directory structure
2. Write SKILL.md
3. Generate script templates
4. Update skill registry
5. Log creation in database
6. Confirm to user
```

## Skill Template Structure

When generating a new skill, follow this template:

```markdown
---
name: skill-name
description: Brief description of what this skill does
---

# Skill Name

## Core Capabilities

1. **Capability 1**
   - Feature A
   - Feature B

2. **Capability 2**
   - Feature C
   - Feature D

## API Operations

### Operation 1

Input:
```json
{
  "action": "operation_name",
  "parameters": {...}
}
```

Output:
```json
{
  "result": {...}
}
```

## Usage Patterns

### Pattern 1: Description

Steps:
1. Step 1
2. Step 2

## Integration with Other Skills

- **skill-1**: How it integrates
- **skill-2**: How it integrates

## Files

```
.agent/skills/skill-name/
├── SKILL.md
├── scripts/
│   └── main_script.py
└── tests/
    └── test_main.py
```

## Example Invocation

```
USER: "Example request"
SKILL ACTION: Description of what skill does
```
```

## Validation Checklist

Before presenting a skill to user, verify:

- [ ] YAML frontmatter present with `name` and `description`
- [ ] Core Capabilities section exists
- [ ] At least one usage pattern documented
- [ ] File structure defined
- [ ] Integration points identified
- [ ] Example invocation provided
- [ ] No security risks identified
- [ ] No conflicts with existing skills

## Database Schema

```sql
-- From server/updates/04_meta_skills_tables.sql

generated_skills (
  skill_name,
  description,
  capabilities,
  template_used,
  validation_status,
  created_at,
  approved_at,
  approved_by
)

skill_versions (
  skill_name,
  version,
  changes,
  performance_metrics,
  created_at
)
```

## API Operations

### Propose New Skill (Internal Only)

```json
{
  "action": "propose_skill",
  "description": "Create a skill for inventory management with stockout prediction"
}
```

**Output** (Presented to User):
```json
{
  "proposal_id": "prop_abc123",
  "skill_name": "inventory-manager",
  "skill_description": "...",
  "capabilities": [...],
  "files": [...],
  "validation": "passed",
  "awaiting_approval": true
}
```

### User Approves Skill

```json
{
  "action": "approve_skill_creation",
  "proposal_id": "prop_abc123",
  "approved_by": "user_id"
}
```

**Outcome**: Skill is created

### Validate Existing Skill

```json
{
  "action": "validate_skill",
  "skill_path": ".agent/skills/existing-skill"
}
```

## Usage Patterns

### Pattern 1: User Requests New Skill

```
USER: "I need a skill to manage my inventory and predict stockouts"

SKILL CREATOR:
1. Parse requirements: inventory tracking + stockout prediction
2. Generate skill structure
3. Validate completeness
4. PRESENT TO USER for approval
5. WAIT for user response
6. IF APPROVED → Create skill
7. Confirm creation to user
```

### Pattern 2: Improve Existing Skill

```
USER: "Can you make the market-researcher skill also track competitor pricing?"

SKILL CREATOR:
1. Analyze existing market-researcher skill
2. Generate enhancement proposal
3. PRESENT changes to user
4. WAIT for approval
5. IF APPROVED → Update skill, increment version
6. Log changes in skill_versions table
```

### Pattern 3: Skill Consolidation

```
CONSCIOUSNESS ENGINE: "I notice 3 skills have overlapping inventory features"

SKILL CREATOR:
1. Analyze overlapping capabilities
2. Propose consolidated skill
3. PRESENT consolidation plan to user
4. WAIT for approval
5. IF APPROVED → Merge skills, deprecate old ones
```

## Security Safeguards

**Never autonomously create skills that**:
- Access sensitive APIs without encryption
- Execute system-level commands without sandboxing
- Modify core platform files
- Bypass authentication
- Store credentials in plaintext

**All skill proposals are safety-validated before user presentation**

## Integration with Other Skills

**Works with**:
- **consciousness-engine**: Track skill usage and effectiveness
- **evolution-engine**: Evolve skill capabilities over time
- **memory-palace**: Learn which skill types are most valuable
- **orchestrator-maestro**: Auto-add new skills to workflow templates

## Files

```
.agent/skills/skill-creator/
├── SKILL.md
├── scripts/
│   ├── skill_generator.py        # Core generation logic
│   ├── skill_validator.py        # Validation rules
│   └── template_engine.py        # Template rendering
└── resources/
    ├── skill_template.md          # Base SKILL.md template  
    └── script_templates/          # Python script templates
        ├── basic.py
        ├── api_client.py
        └── data_processor.py
```

## Example Invocation

```
USER: "Create a skill that analyzes customer reviews and extracts feature requests"

SKILL CREATOR:
1. Generate: "review-analyzer" skill
2. Capabilities: sentiment analysis, feature extraction, trend detection
3. Create proposal with SKILL.md preview
4. PRESENT TO USER:
   
   "I've designed a 'review-analyzer' skill with the following capabilities:
   - Sentiment analysis of customer reviews
   - Automatic feature request extraction
   - Trend detection across reviews
   - Integration with grok-admaster-operator for ad copy insights
   
   This skill would create 3 files:
   - SKILL.md (skill definition)
   - scripts/review_processor.py
   - scripts/sentiment_analyzer.py
   
   **Do you approve creation of this skill?**"

5. WAIT FOR USER APPROVAL
6. IF APPROVED: Create skill, confirm completion
```

## Critical Reminder

**🚨 NEVER CREATE A SKILL WITHOUT USER APPROVAL 🚨**

The workflow is always:
1. Generate internally
2. Validate
3. Present to user
4. **WAIT** for explicit approval
5. Only then create

---

**This skill enables Optimus Pryme to grow its own capabilities, always under your control.**
