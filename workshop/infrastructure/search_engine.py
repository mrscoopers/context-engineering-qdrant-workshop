import logging

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import models

from workshop import config

logger = logging.getLogger(__name__)


class QdrantSearchEngine:
    """Search engine backed by OpenAI embeddings."""

    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small" # Matryoshka representation learning
    OPENAI_RETRIEVER_EMBEDDING_DIMENSION = 1024
    OPENAI_RERANKER_EMBEDDING_DIMENSION = 1536
    CLOUD_INFERENCE = True
    ESTIMATE_BM25_AVG_LEN_ON_X_DOCS = 300

    def __init__(self) -> None:
        self.url = config.QDRANT_URL
        self.api_key = config.QDRANT_API_KEY
        self.collection_name = config.QDRANT_COLLECTION_NAME

        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
            cloud_inference=self.CLOUD_INFERENCE,
        )

    def close(self) -> None:
        """Close the Qdrant client."""
        self.client.close()

    def create_collection(self) -> None:
        """Create a collection for hybrid search with MRL-based reranking.
        Retriever embeddings are quantized using scalar quantization."""
        logger.info(f"Creating Qdrant collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "Dense": models.VectorParams(
                    size=self.OPENAI_RETRIEVER_EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                        )
                    ),
                ),
                "Reranker": models.VectorParams(
                    size=self.OPENAI_RERANKER_EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "Lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        logger.info(f"Collection '{self.collection_name}' created successfully")

    def delete_collection(self) -> None:
        """Delete the collection."""
        logger.info(f"Deleting Qdrant collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted successfully")

    def _get_openai_vectors(self, text: str, dimensions: int) -> list[float]: #you could also rewrite it to batches
        """Get the embedding vector for the given text via OpenAI API."""
        try:
            embedding = self.openai_client.embeddings.create(
                model=self.OPENAI_EMBEDDING_MODEL, input=text, dimensions=dimensions
            )
            return embedding.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    def _define_openai_vectors(self, text: str, mrl_dimensions: int = 1536) -> models.Document:
        """Wrap text in models.Document for Qdrant Cloud Inference."""
        return models.Document(
            text=text,
            model=f"openai/{self.OPENAI_EMBEDDING_MODEL}",
            options={
                "openai-api-key": config.OPENAI_API_KEY,
                "mrl": mrl_dimensions,
            },
        )
    
    def _estimate_avg_abstract_len(self, papers: list[dict]) -> int:
        """Estimate average abstract length from the first 
        ESTIMATE_BM25_AVG_LEN_ON_X_DOCS for BM25 formula-based sparse vectors."""
        total_words = 0
        sampled_count = 0
        for paper in papers[: self.ESTIMATE_BM25_AVG_LEN_ON_X_DOCS]:
            abstract = paper.get("abstract")
            if abstract:
                total_words += len(abstract.split())
                sampled_count += 1
        if sampled_count > 0:
            avg_len = total_words // sampled_count
            logger.info(f"Estimated average abstract length for BM25 formula: {avg_len} words")
            return avg_len
        logger.info("No abstracts found to estimate average length, defaulting to 256 words")
        return 256

    def _define_bm25_vectors(self, text: str, avg_len: int = 256) -> models.Document:
        """Wrap text in models.Document for BM25 sparse vectors."""
        return models.Document(
            text=text, model="qdrant/bm25", options={"avg_len": avg_len, "language": "english"}
        )

    def _points_generator(self, papers: list[dict], avg_abstracts_len: int):
        """Yield PointStruct objects."""
        for paper in papers:
            pmid = paper.get("pmid")
            abstract = paper.get("abstract")

            if not abstract or not pmid:
                logger.info("Skipping paper: abstract or pmid is missing")
                continue

            try:
                point_id = int(pmid)

                payload = {
                    "paper": {
                        "pmid": pmid,
                        "title": paper.get("title", ""),
                        "abstract": abstract,
                        "authors": paper.get("authors", []),
                        "mesh_terms": paper.get("mesh_terms", []),
                        "publication_date": paper.get("publication_date", ""),
                        "journal": paper.get("journal", ""),
                        "doi": paper.get("doi", ""),
                    },
                }

                if self.CLOUD_INFERENCE:
                    retriever_vector = self._define_openai_vectors(
                        abstract, mrl_dimensions=self.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
                    )
                    reranker_vector = self._define_openai_vectors(
                        abstract, mrl_dimensions=self.OPENAI_RERANKER_EMBEDDING_DIMENSION  # Qdrant deduplicates into one OpenAI API call
                    )
                else:
                    openai_vector = self._get_openai_vectors(
                        abstract, dimensions=self.OPENAI_RERANKER_EMBEDDING_DIMENSION
                    )
                    retriever_vector = openai_vector[: self.OPENAI_RETRIEVER_EMBEDDING_DIMENSION]  # MRL truncation; Qdrant normalizes vectors for you
                    reranker_vector = openai_vector

                sparse_vector = self._define_bm25_vectors(abstract, avg_len=avg_abstracts_len)

                yield models.PointStruct(
                    id=point_id,
                    vector={
                        "Dense": retriever_vector,
                        "Reranker": reranker_vector,
                        "Lexical": sparse_vector,
                    },
                    payload=payload,
                )
            except Exception as e:
                logger.error(f"Failed to process paper {pmid}: {e}")
                continue

    def upsert_points(
        self,
        pubmed_data: dict,
        only_new: bool = False,
        batch_size: int = 32,
        parallel: int = 2,
        max_retries: int = 5
    ) -> None:
        """Upsert papers into Qdrant.

        Args:
            pubmed_data: Parsed JSON from pubmed_dataset.json.
            only_new: If True, only ingest papers not already in the collection.
            batch_size: Number of points per batch.
        """
        papers = pubmed_data.get("papers", [])

        logger.info(f"Starting ingestion of {len(papers)} papers in batches of {batch_size}")

        avg_abstracts_len = self._estimate_avg_abstract_len(papers)

        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_generator(papers, avg_abstracts_len),
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
            update_mode=models.UpdateMode.INSERT_ONLY if only_new else models.UpdateMode.UPSERT
        )

        logger.info(f"Ingestion complete!")
