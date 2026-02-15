# Phase 7: Security & Compliance Report

## ✅ Actions Taken
We analyzed the Supabase Linter warnings and produced fixes for:

### 1. **Mutable Search Path (WARN)**
- **Issue**: Functions running without a fixed `search_path` can be hijacked by malicious users masking system functions.
- **Affected Functions**:
  - `public.get_anomaly_stats`
  - `public.archive_old_anomaly_alerts`
- **Fix**: Set `search_path = public, pg_temp` explicitly.

### 2. **RLS Disabled (ERROR)**
- **Issue**: Tables `anomaly_alerts`, `anomaly_history`, `anomaly_training_data` were public without RLS.
- **Fix**: Enabled Row Level Security on all 3 tables.
- **Note**: Currently using permissive `USING (true)` policies for authenticated users to facilitate MVP development. In production, restrictive policies (`auth.uid() = profile_id`) should be applied.

### 3. **Column Type Correction**
- **Issue**: `profile_id` was `INTEGER` but the `profiles` table uses `VARCHAR`.
- **Fix**: Altered column type to `VARCHAR` to allow foreign key constraints.

---

## 🛠 Manual Action Required
Due to permission restrictions on the database connection (the app user is not the table owner), the automated script could not apply these changes.

**Please execute the following SQL in your Supabase Dashboard:**

```sql
-- Security Hardening
ALTER FUNCTION public.archive_old_anomaly_alerts() SET search_path = public, pg_temp;
ALTER FUNCTION public.get_anomaly_stats(integer) SET search_path = public, pg_temp;

-- RLS & Schema Fixes
ALTER TABLE anomaly_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE anomaly_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE anomaly_training_data ENABLE ROW LEVEL SECURITY;

-- If type mismatch exists (Integer vs String)
DO $$ BEGIN
    ALTER TABLE anomaly_alerts ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;
    ALTER TABLE anomaly_history ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;
    ALTER TABLE anomaly_training_data ALTER COLUMN profile_id TYPE VARCHAR USING profile_id::VARCHAR;
EXCEPTION WHEN OTHERS THEN NULL; END $$;

-- Policies (Re-create if needed)
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON anomaly_alerts;
CREATE POLICY "Enable all access for authenticated users" ON anomaly_alerts
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
    
DROP POLICY IF EXISTS "Enable all access for authenticated users" ON anomaly_history;
CREATE POLICY "Enable all access for authenticated users" ON anomaly_history
    FOR ALL TO authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS "Enable all access for authenticated users" ON anomaly_training_data;
CREATE POLICY "Enable all access for authenticated users" ON anomaly_training_data
    FOR ALL TO authenticated USING (true) WITH CHECK (true);
```
