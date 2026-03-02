import logging

from qdrant_client.models import models

from workshop.infrastructure.search_engine import QdrantSearchEngine

logger = logging.getLogger(__name__)


class QdrantQuery:
    """Handles querying Qdrant vector search engine."""

    def __init__(self) -> None:
        self.search_engine = QdrantSearchEngine()

    def close(self) -> None:
        self.search_engine.close()

    def retrieve_papers_based_on_query(self, query: str, limit: int = 5) -> list[dict]:
        """Search for papers using dense + lexical retrieval, fused by reranking."""
        if self.search_engine.CLOUD_INFERENCE:
            retriever_vector = self.search_engine._define_openai_vectors(
                query, mrl_dimensions=self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
            )
            reranker_vector = self.search_engine._define_openai_vectors(
                query, mrl_dimensions=self.search_engine.OPENAI_RERANKER_EMBEDDING_DIMENSION
            )
        else:
            openai_vector = self.search_engine._get_openai_vectors(
                query, dimensions=self.search_engine.OPENAI_RERANKER_EMBEDDING_DIMENSION
            )
            retriever_vector = openai_vector[: self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION]
            reranker_vector = openai_vector

        sparse_vector = self.search_engine._define_bm25_vectors(query)

        search_result = self.search_engine.client.query_points(
            collection_name=self.search_engine.collection_name,
            prefetch=[
                models.Prefetch(
                    query=retriever_vector,
                    using="Dense",
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            oversampling=3.0,
                            rescore=True,
                        )
                    ),
                    limit=limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="Lexical",
                    limit=limit,
                ),
            ],
            query=reranker_vector,
            using="Reranker",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in search_result.points
        ]

    def recommend_papers_based_on_constraints(
        self,
        positive_examples: list[str] | None,
        negative_examples: list[str] | None,
        limit: int = 5,
    ) -> list[dict]:
        """Recommend papers based on positive and negative examples."""
        if self.search_engine.CLOUD_INFERENCE:
            positive_vectors = (
                [
                    self.search_engine._define_openai_vectors(
                        ex, mrl_dimensions=self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
                    )
                    for ex in positive_examples
                ]
                if positive_examples
                else None
            )
            negative_vectors = (
                [
                    self.search_engine._define_openai_vectors(
                        ex, mrl_dimensions=self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
                    )
                    for ex in negative_examples
                ]
                if negative_examples
                else None
            )
        else:
            positive_vectors = (
                [
                    self.search_engine._get_openai_vectors(
                        ex, dimensions=self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
                    )
                    for ex in positive_examples
                ]
                if positive_examples
                else None
            )
            negative_vectors = (
                [
                    self.search_engine._get_openai_vectors(
                        ex, dimensions=self.search_engine.OPENAI_RETRIEVER_EMBEDDING_DIMENSION
                    )
                    for ex in negative_examples
                ]
                if negative_examples
                else None
            )

        recommendation_result = self.search_engine.client.query_points(
            collection_name=self.search_engine.collection_name,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=positive_vectors,
                    negative=negative_vectors,
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                )
            ),
            using="Dense",
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": point.id, "score": point.score, "payload": point.payload}
            for point in recommendation_result.points
        ]
