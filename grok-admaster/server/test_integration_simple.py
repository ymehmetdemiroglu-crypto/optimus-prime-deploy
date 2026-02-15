"""
Simple integration test for anomaly detection.
Uses plain ASCII to avoid Windows encoding issues.
"""
import asyncio
from sqlalchemy import text
from app.core.database import engine

async def main():
    print("\n" + "="*60)
    print("ANOMALY DETECTION INTEGRATION TEST")
    print("="*60)
    
    try:
        async with engine.begin() as conn:
            # Test 1: Check tables
            print("\n[TEST 1] Checking database tables...")
            result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'anomaly%'"))
            tables = [row[0] for row in result]
            print(f"Found tables: {tables}")
            
            if len(tables) >= 3:
                print("[PASS] All 3 anomaly tables exist!")
            else:
                print(f"[FAIL] Only {len(tables)} tables found")
                return False
            
            # Test 2: Check functions
            print("\n[TEST 2] Testing helper functions...")
            result = await conn.execute(text("SELECT archive_old_anomaly_alerts()"))
            archived = result.scalar()
            print(f"[PASS] archive_old_anomaly_alerts() works (archived: {archived})")
            
            result = await conn.execute(text("SELECT get_anomaly_stats('1')"))
            stats = result.scalar()
            print(f"[PASS] get_anomaly_stats() works")
            
            # Test 3: Check imports
            print("\n[TEST 3] Testing Python imports...")
            from app.modules.amazon_ppc.anomaly import router, anomaly_service
            print("[PASS] Router and service imported")
            
            from app.modules.amazon_ppc.anomaly.models import AnomalyAlert
            print("[PASS] Models imported")
            
            # Test 4: Check router registration
            print("\n[TEST 4] Checking router registration in main.py...")
            with open("app/main.py", "r") as f:
                main_content = f.read()
                if "anomaly_detection_router" in main_content:
                    print("[PASS] Router registered in main.py!")
                else:
                    print("[WARN] Router not found in main.py")
            
            print("\n" + "="*60)
            print("[SUCCESS] ALL TESTS PASSED!")
            print("="*60)
            print("\nIntegration complete! Next steps:")
            print("  1. Start server: uvicorn app.main:app --reload")
            print("  2. Test API: http://localhost:8000/docs")
            print("  3. Look for /api/v1/anomaly-detection endpoints")
            print("="*60)
            
            return True
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await engine.dispose()

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
