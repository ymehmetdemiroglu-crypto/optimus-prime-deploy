-- Final Fix for "Function Search Path Mutable"
-- Run this in Supabase SQL Editor

-- 1. Drop the function first to ensure we can recreate it with the correct properties
DROP FUNCTION IF EXISTS public.get_anomaly_stats(integer);

-- 2. Recreate it with 'SET search_path' included in the definition
CREATE OR REPLACE FUNCTION public.get_anomaly_stats(p_profile_id INTEGER)
RETURNS JSONB
LANGUAGE plpgsql
-- SECURITY DEFINER -- Optional: Use only if you need the function to run with owner privileges
SET search_path = public, pg_temp -- <--- THIS IS THE FIX
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
    WHERE profile_id = p_profile_id::VARCHAR; -- Cast to VARCHAR to match new schema type
    
    RETURN result;
END;
$$;
