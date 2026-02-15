-- 1. Drop both potential variations of the function to ensure a clean slate
DROP FUNCTION IF EXISTS public.get_anomaly_stats(integer);
DROP FUNCTION IF EXISTS public.get_anomaly_stats(varchar);

-- 2. Recreate the function safely with the security fix
-- We accept INTEGER or VARCHAR input for convenience, casting internally
CREATE OR REPLACE FUNCTION public.get_anomaly_stats(p_profile_id INTEGER)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER -- Optional: Run as creator (usually postgres) to ensure permissions
SET search_path = public, pg_temp -- <--- SECURITY FIX: Immutable search path
AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_alerts', COUNT(*),
        'unresolved', COUNT(*) FILTER (WHERE is_resolved = FALSE),
        'critical', COUNT(*) FILTER (WHERE severity = 'critical' AND is_resolved = FALSE),
        'high', COUNT(*) FILTER (WHERE severity = 'high' AND is_resolved = FALSE),
        'medium', COUNT(*) FILTER (WHERE severity = 'medium' AND is_resolved = FALSE),
        'low', COUNT(*) FILTER (WHERE severity = 'low' AND is_resolved = FALSE),
        'last_24h', COUNT(*) FILTER (WHERE detection_timestamp >= NOW() - INTERVAL '24 hours')
    )
    INTO result
    FROM anomaly_alerts
    WHERE profile_id = p_profile_id::VARCHAR; -- Handle type conversion safely
    
    RETURN result;
END;
$$;
