"""
Script to backfill keyword embeddings into the vector table.
"""
import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.core.config import settings
from app.modules.amazon_ppc.models.ppc_data import PPCKeyword, KeywordVector
from app.services.ml.embedding_service import embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def backfill_embeddings():
    """
    Find keywords without embeddings and generate them.
    Store in keyword_vectors table.
    """
    logger.info("Connecting to DB...")
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        # Find keywords missing from keyword_vectors
        # Left join or NOT EXISTS
        query = select(PPCKeyword).outerjoin(KeywordVector).where(KeywordVector.keyword_id == None)
        result = await db.execute(query)
        keywords_to_process = result.scalars().all()
        
        logger.info(f"Found {len(keywords_to_process)} keywords needing embeddings.")
        
        batch_size = 50
        processed = 0
        
        for i in range(0, len(keywords_to_process), batch_size):
            batch = keywords_to_process[i:i+batch_size]
            
            # Generate vectors
            texts = [kw.keyword_text for kw in batch]
            vectors = embedding_service.encode_batch(texts)
            
            # Create KeywordVector objects
            for j, kw in enumerate(batch):
                vector = vectors[j]
                if vector is not None:
                    # Convert to list for pgvector (it handles numpy conversion usually, but list is safer for some drivers)
                    # pgvector-python handles numpy arrays fine.
                    kv = KeywordVector(
                        keyword_id=kw.id,
                        embedding=vector
                    )
                    db.add(kv)
            
            await db.commit()
            processed += len(batch)
            logger.info(f"Processed {processed}/{len(keywords_to_process)} keywords")
            
    logger.info("Backfill complete.")

if __name__ == "__main__":
    asyncio.run(backfill_embeddings())
