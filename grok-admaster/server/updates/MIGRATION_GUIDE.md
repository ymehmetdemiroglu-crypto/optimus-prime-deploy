# Database Migration Guide

## Manual Migration Required

Due to API access limitations, the database migrations need to be applied manually through the Supabase dashboard.

## Step-by-Step Instructions

### 1. Access Supabase Dashboard

1. Go to [https://supabase.com/dashboard](https://supabase.com/dashboard)
2. Select your project: **lzhxmzsrxkzcbcglxhyk**
3. Navigate to **SQL Editor** in the left sidebar

### 2. Apply Tier 1 Migration (04_meta_skills_tables.sql)

1. Click **New Query**
2. Copy the entire contents of: `c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server\updates\04_meta_skills_tables.sql`
3. Paste into the SQL Editor
4. Click **Run** (or press Ctrl+Enter)
5. Verify success: Should see "Success. No rows returned"

**Tables Created** (8 tables):
- ✅ `skill_executions`
- ✅ `workflow_templates`
- ✅ `memory_patterns`
- ✅ `case_library`
- ✅ `decision_audit`
- ✅ `model_performance_tracking`
- ✅ `generated_skills`
- ✅ `skill_versions`

### 3. Apply Tier 2 Migration (05_tier2_meta_skills_tables.sql)

1. Click **New Query** again
2. Copy the entire contents of: `c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server\updates\05_tier2_meta_skills_tables.sql`
3. Paste into the SQL Editor
4. Click **Run**
5. Verify success

**Tables Created** (6 tables):
- ✅ `strategy_lineage`
- ✅ `evolution_cycles`
- ✅ `simulation_runs`
- ✅ `backtest_results`
- ✅ `synthesized_insights`
- ✅ `external_knowledge`

### 4. Verify Tables

Run this query to confirm all tables were created:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'skill_executions', 'workflow_templates', 'memory_patterns', 'case_library',
    'decision_audit', 'model_performance_tracking', 'generated_skills', 'skill_versions',
    'strategy_lineage', 'evolution_cycles', 'simulation_runs', 'backtest_results',
    'synthesized_insights', 'external_knowledge'
)
ORDER BY table_name;
```

**Expected Result**: 14 rows

### 5. Verify RLS Policies

Run this query to confirm Row Level Security is enabled:

```sql
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN (
    'skill_executions', 'workflow_templates', 'memory_patterns', 'case_library',
    'decision_audit', 'model_performance_tracking', 'generated_skills', 'skill_versions',
    'strategy_lineage', 'evolution_cycles', 'simulation_runs', 'backtest_results',
    'synthesized_insights', 'external_knowledge'
)
ORDER BY tablename;
```

**Expected**: All tables should have `rowsecurity = true`

## Troubleshooting

### Error: "relation already exists"
- **Cause**: Tables already created
- **Solution**: Skip that migration or use `DROP TABLE IF EXISTS` first

### Error: "permission denied"
- **Cause**: Insufficient database permissions
- **Solution**: Ensure you're logged in as the project owner

### Error: "syntax error"
- **Cause**: Incomplete SQL copied
- **Solution**: Ensure you copied the entire file contents

## Alternative: Python Script

If you prefer automation, you can use this Python script (requires database owner credentials):

```python
import os
from supabase import create_client, Client

# Set your Supabase credentials
url = "https://lzhxmzsrxkzcbcglxhyk.supabase.co"
key = "YOUR_SERVICE_ROLE_KEY"  # Service role key, not anon key

supabase: Client = create_client(url, key)

# Read and execute Tier 1 migration
with open(r"c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server\updates\04_meta_skills_tables.sql", "r") as f:
    tier1_sql = f.read()
    result = supabase.rpc("exec_sql", {"query": tier1_sql}).execute()
    print("Tier 1 migration:", result)

# Read and execute Tier 2 migration
with open(r"c:\Users\hp\OneDrive\Desktop\optimus pryme\grok-admaster\server\updates\05_tier2_meta_skills_tables.sql", "r") as f:
    tier2_sql = f.read()
    result = supabase.rpc("exec_sql", {"query": tier2_sql}).execute()
    print("Tier 2 migration:", result)
```

## Post-Migration Checklist

- [ ] All 14 tables created
- [ ] RLS enabled on all tables
- [ ] Policies created for authenticated users
- [ ] Indexes created
- [ ] No errors in Supabase logs

---

**Once migrations are complete, the meta-skills system will have full database support.**
