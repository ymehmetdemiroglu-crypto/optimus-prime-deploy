"""
Verify Domain Skills and Automation Database Tables
Run this script to check if the new tables have been created and contain the required 'profile_id' column.
"""

def get_verification_sql():
    tables_to_check = [
        "competitor_prices",
        "price_changes",
        "share_of_voice",
        "product_reviews_analysis",
        "ml_models",
        "detected_anomalies",
        "customer_segments",
        "alert_configs",
        "alert_history",
        "scheduled_reports",
        "scheduled_tasks",
        "task_executions"
    ]
    
    print("\n=== Expected Tables & Columns ===")
    print("Checking for tables and 'profile_id' column in each...")

    print("\n=== Verification SQL Query ===")
    print("Run this to verify structure:")
    print("-" * 50)
    
    query = """
    SELECT table_name, column_name
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND column_name = 'profile_id'
    AND table_name IN (
        'competitor_prices',
        'price_changes',
        'share_of_voice',
        'product_reviews_analysis',
        'ml_models',
        'detected_anomalies',
        'customer_segments',
        'alert_configs',
        'alert_history',
        'scheduled_reports',
        'scheduled_tasks',
        'task_executions'
    );
    """
    print(query)
    print("-" * 50)
    print(f"You should see {len(tables_to_check)} rows (one for each table having profile_id).")

if __name__ == "__main__":
    get_verification_sql()
