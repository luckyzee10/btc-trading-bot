#!/usr/bin/env python3
"""
Database Migration Script for BTC Trading Bot
Migrates from old schema to new portfolio tracking schema
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_db_connection():
    """Get database connection."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logging.error("DATABASE_URL environment variable not found!")
        return None
    return psycopg2.connect(db_url)

def check_current_schema():
    """Check current database schema."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check bot_state columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'bot_state'
                    ORDER BY ordinal_position
                """)
                bot_state_columns = cur.fetchall()
                
                # Check trades columns  
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'trades'
                    ORDER BY ordinal_position
                """)
                trades_columns = cur.fetchall()
                
                print("🔍 Current Database Schema:")
                print("=" * 50)
                print("bot_state table columns:")
                for col in bot_state_columns:
                    print(f"  - {col['column_name']} ({col['data_type']})")
                
                print("\ntrades table columns:")
                for col in trades_columns:
                    print(f"  - {col['column_name']} ({col['data_type']})")
                
                return bot_state_columns, trades_columns
                
    except Exception as e:
        logging.error(f"Error checking schema: {e}")
        return None, None

def migrate_bot_state_table():
    """Add missing portfolio tracking columns to bot_state table."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                print("\n🔧 Migrating bot_state table...")
                
                # Add missing columns if they don't exist
                migrations = [
                    "ALTER TABLE bot_state ADD COLUMN IF NOT EXISTS current_balance DECIMAL(15, 2) DEFAULT 100.0",
                    "ALTER TABLE bot_state ADD COLUMN IF NOT EXISTS btc_holdings DECIMAL(15, 8) DEFAULT 0.0",
                    "ALTER TABLE bot_state ADD COLUMN IF NOT EXISTS trade_count INTEGER DEFAULT 0",
                    "ALTER TABLE bot_state ADD COLUMN IF NOT EXISTS winning_trades INTEGER DEFAULT 0", 
                    "ALTER TABLE bot_state ADD COLUMN IF NOT EXISTS losing_trades INTEGER DEFAULT 0"
                ]
                
                for migration in migrations:
                    cur.execute(migration)
                    print(f"  ✅ {migration}")
                
                conn.commit()
                print("✅ bot_state table migration completed!")
                
    except Exception as e:
        logging.error(f"Error migrating bot_state table: {e}")
        raise

def migrate_trades_table():
    """Update trades table schema for portfolio tracking."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                print("\n🔧 Migrating trades table...")
                
                # Check if old columns exist (reason, rsi, etc.)
                cur.execute("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'trades' AND column_name = 'reason'
                """)
                has_old_schema = cur.fetchone() is not None
                
                if has_old_schema:
                    print("  📊 Found old trades schema - creating new table...")
                    
                    # Backup existing trades data
                    cur.execute("CREATE TABLE trades_backup AS SELECT * FROM trades")
                    print("  💾 Backed up existing trades to trades_backup")
                    
                    # Drop old table
                    cur.execute("DROP TABLE trades")
                    print("  🗑️ Dropped old trades table")
                    
                    # Create new trades table with portfolio tracking
                    cur.execute("""
                        CREATE TABLE trades (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            price DECIMAL(15, 2) NOT NULL,
                            signal VARCHAR(10) NOT NULL,
                            btc_amount DECIMAL(15, 8),
                            usd_amount DECIMAL(15, 2),
                            portfolio_value DECIMAL(15, 2),
                            balance DECIMAL(15, 2),
                            btc_holdings DECIMAL(15, 8)
                        )
                    """)
                    print("  ✅ Created new trades table with portfolio tracking")
                    
                    # Try to migrate old data (basic mapping)
                    cur.execute("""
                        INSERT INTO trades (timestamp, price, signal)
                        SELECT timestamp, price, signal FROM trades_backup
                    """)
                    migrated_count = cur.rowcount
                    print(f"  📈 Migrated {migrated_count} existing trades")
                    
                else:
                    # Add missing columns to existing table
                    migrations = [
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS btc_amount DECIMAL(15, 8)",
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS usd_amount DECIMAL(15, 2)",
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS portfolio_value DECIMAL(15, 2)",
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS balance DECIMAL(15, 2)",
                        "ALTER TABLE trades ADD COLUMN IF NOT EXISTS btc_holdings DECIMAL(15, 8)"
                    ]
                    
                    for migration in migrations:
                        cur.execute(migration)
                        print(f"  ✅ {migration}")
                
                conn.commit()
                print("✅ trades table migration completed!")
                
    except Exception as e:
        logging.error(f"Error migrating trades table: {e}")
        raise

def verify_migration():
    """Verify that migration was successful."""
    try:
        print("\n🔍 Verifying migration...")
        bot_state_cols, trades_cols = check_current_schema()
        
        # Check required bot_state columns
        required_bot_state = ['current_balance', 'btc_holdings', 'trade_count', 'winning_trades', 'losing_trades']
        bot_state_names = [col['column_name'] for col in bot_state_cols]
        
        missing_bot_state = [col for col in required_bot_state if col not in bot_state_names]
        
        # Check required trades columns  
        required_trades = ['btc_amount', 'usd_amount', 'portfolio_value', 'balance', 'btc_holdings']
        trades_names = [col['column_name'] for col in trades_cols]
        
        missing_trades = [col for col in required_trades if col not in trades_names]
        
        if not missing_bot_state and not missing_trades:
            print("✅ Migration verification successful!")
            print("🎯 Database is ready for portfolio tracking!")
            return True
        else:
            print("❌ Migration verification failed!")
            if missing_bot_state:
                print(f"   Missing bot_state columns: {missing_bot_state}")
            if missing_trades:
                print(f"   Missing trades columns: {missing_trades}")
            return False
            
    except Exception as e:
        logging.error(f"Error verifying migration: {e}")
        return False

def main():
    """Run the database migration."""
    print("🚀 BTC Trading Bot Database Migration")
    print("=" * 50)
    
    # Check if DATABASE_URL exists
    if not os.getenv('DATABASE_URL'):
        print("❌ DATABASE_URL environment variable not found!")
        print("   Set it to your Railway PostgreSQL connection string")
        return
    
    try:
        # Show current schema
        check_current_schema()
        
        # Confirm migration
        print("\n⚠️  This will modify your database schema.")
        response = input("Continue with migration? (y/N): ").lower().strip()
        
        if response != 'y':
            print("Migration cancelled.")
            return
        
        # Run migrations
        migrate_bot_state_table()
        migrate_trades_table()
        
        # Verify success
        if verify_migration():
            print("\n🎉 Migration completed successfully!")
            print("Your bot should now start tracking portfolio performance!")
        else:
            print("\n❌ Migration failed - please check the errors above")
        
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        print(f"\n❌ Migration failed: {e}")

if __name__ == "__main__":
    main() 