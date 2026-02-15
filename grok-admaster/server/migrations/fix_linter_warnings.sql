-- Fix "Function Search Path Mutable" warnings
-- Security Best Practice: Set a fixed search_path for functions to prevent hijacking

ALTER FUNCTION public.archive_old_anomaly_alerts() SET search_path = public, pg_temp;

ALTER FUNCTION public.get_anomaly_stats(integer) SET search_path = public, pg_temp;

-- Note on RLS Policies:
-- The warnings for 'rl_budget_actions' and 'rl_portfolio_state' indicate permissive policies.
-- If you want to secure them, you should restrict them to 'authenticated' or specific roles.
-- Example fix (uncomment to apply):
-- ALTER TABLE rl_budget_actions ENABLE ROW LEVEL SECURITY;
-- DROP POLICY IF EXISTS "Service role full access on rl_budget_actions" ON rl_budget_actions;
-- CREATE POLICY "Authenticated full access" ON rl_budget_actions FOR ALL TO authenticated USING (true) WITH CHECK (true);
