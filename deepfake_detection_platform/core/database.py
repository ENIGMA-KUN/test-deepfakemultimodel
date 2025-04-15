import os
import sqlite3
import aiosqlite
import logging
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

# Database connection
DB_PATH = settings.DATABASE_PATH

async def init_db():
    """
    Initialize the database with required tables.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            
            # Create uploads table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            
            # Create results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id TEXT PRIMARY KEY,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    completion_time TEXT NOT NULL,
                    duration REAL NOT NULL,
                    details TEXT,
                    FOREIGN KEY (id) REFERENCES uploads (id)
                )
            """)
            
            await db.commit()
            
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise

async def get_db_connection():
    """
    Get a database connection.
    
    Returns:
        An aiosqlite connection object
    """
    try:
        db = await aiosqlite.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        return db
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}", exc_info=True)
        raise

async def close_db():
    """
    Close all database connections.
    This is a placeholder as aiosqlite automatically closes connections.
    """
    # The connections are closed automatically when the context managers exit
    logger.info("Database connections closed")

# Helper functions for database operations

async def add_upload_record(upload_id: str, filename: str, media_type: str, status: str = "pending"):
    """
    Add a new upload record to the database.
    
    Args:
        upload_id: Unique ID for the upload
        filename: Original filename
        media_type: Type of media (image, audio, video)
        status: Upload status
    """
    from datetime import datetime
    
    try:
        db = await get_db_connection()
        await db.execute(
            """
            INSERT INTO uploads (id, filename, media_type, upload_time, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                upload_id,
                filename,
                media_type,
                datetime.now().isoformat(),
                status
            )
        )
        await db.commit()
        await db.close()
    except Exception as e:
        logger.error(f"Error adding upload record: {str(e)}", exc_info=True)
        raise

async def update_upload_status(upload_id: str, status: str):
    """
    Update the status of an upload.
    
    Args:
        upload_id: ID of the upload
        status: New status
    """
    try:
        db = await get_db_connection()
        await db.execute(
            "UPDATE uploads SET status = ? WHERE id = ?",
            (status, upload_id)
        )
        await db.commit()
        await db.close()
    except Exception as e:
        logger.error(f"Error updating upload status: {str(e)}", exc_info=True)
        raise

async def add_result_record(result_id: str, prediction: str, confidence: float, duration: float, details: Optional[str] = None):
    """
    Add a new result record to the database.
    
    Args:
        result_id: Unique ID for the result (same as upload_id)
        prediction: Prediction result (Real/Fake)
        confidence: Confidence score (0-1)
        duration: Analysis duration in seconds
        details: JSON string with detection details
    """
    from datetime import datetime
    
    try:
        db = await get_db_connection()
        await db.execute(
            """
            INSERT INTO results (id, prediction, confidence, completion_time, duration, details)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                prediction,
                confidence,
                datetime.now().isoformat(),
                duration,
                details
            )
        )
        await db.commit()
        await db.close()
    except Exception as e:
        logger.error(f"Error adding result record: {str(e)}", exc_info=True)
        raise
