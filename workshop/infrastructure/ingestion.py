import json
import logging

from workshop import config
from workshop.infrastructure.search_engine import QdrantSearchEngine

logger = logging.getLogger(__name__)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ingest_data(recreate: bool = False, only_new: bool = False) -> None:
    """Ingest data from a pubmed dataset into the Qdrant vector search engine.

    Args:
        recreate: If True, recreate the collection (deletes all existing data).
        only_new: If True, only ingest papers not already in the collection.
    """
    logger.info("Starting Qdrant data ingestion...")

    search_engine = QdrantSearchEngine()
    try:
        if search_engine.client.collection_exists(search_engine.collection_name):
            if recreate:
                logger.info("Recreating collection...")
                search_engine.delete_collection()
                search_engine.create_collection()
        else:
            logger.info(f"Collection '{search_engine.collection_name}' does not exist. Creating...")
            search_engine.create_collection()

        # Load datasets
        logger.info("Loading datasets...")
        pubmed_data = _load_json(config.PUBMED_JSON_PATH)
        logger.info(f"Loaded {len(pubmed_data.get('papers', []))} papers")

        # Upsert papers into Qdrant
        search_engine.upsert_points(pubmed_data, only_new=only_new)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        search_engine.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_data(recreate=True, only_new=False)
