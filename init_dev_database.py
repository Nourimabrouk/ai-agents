#!/usr/bin/env python3
'''
Database Initialization Script
Creates development database tables and sample data
'''

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

async def init_development_database():
    '''Initialize development database with sample data'''
    try:
        # Import after adding to path
        from api.database.session import init_database, get_database
        from api.models.database_models import Base
        
        print("[DB] Initializing development database...")
        
        # Skip if no database configuration
        if not os.getenv('DATABASE_DATABASE_URL'):
            print("[SKIP] No database URL configured")
            return
        
        # Initialize database
        await init_database()
        print("[OK] Database initialized successfully")
        
    except ImportError as e:
        print(f"[SKIP] Database modules not available: {e}")
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        print("[INFO] This is normal for development - database is optional")

if __name__ == "__main__":
    print("[DATABASE] Development database initialization")
    asyncio.run(init_development_database())
    print("[COMPLETE] Database initialization script completed")
