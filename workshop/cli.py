import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_create_collection() -> None:
    from workshop.infrastructure.search_engine import QdrantSearchEngine

    engine = QdrantSearchEngine()
    try:
        engine.create_collection()
    finally:
        engine.close()


def cmd_delete_collection() -> None:
    from workshop.infrastructure.search_engine import QdrantSearchEngine

    engine = QdrantSearchEngine()
    try:
        engine.delete_collection()
    finally:
        engine.close()


def cmd_ingest(args: argparse.Namespace) -> None:
    from workshop.infrastructure.ingestion import ingest_data

    ingest_data(recreate=args.recreate, only_new=args.only_new)


def cmd_context_engineering(args: argparse.Namespace) -> None:
    from workshop.context_engineering.context import get_context

    question = " ".join(args.question)
    if not question.strip():
        logger.error("Please provide a question.")
        sys.exit(1)

    result = get_context(question, limit=args.limit)

    print(f"\n{'='*60}")
    print(f"Tool used: {result.tool_used}")
    print(f"Results: {len(result.qdrant_results)}")
    print(f"{'='*60}\n")

    for i, r in enumerate(result.qdrant_results, 1):
        paper = r.get("payload", {}).get("paper", {})
        print(f"--- Result {i} (score: {r.get('score', 'N/A'):.4f}) ---")
        print(f"  PMID:    {paper.get('pmid', 'N/A')}")
        print(f"  Title:   {paper.get('title', 'N/A')}")
        print(f"  Journal: {paper.get('journal', 'N/A')}")
        print(f"  Date:    {paper.get('publication_date', 'N/A')}")
        print()

    if result.agent_summary:
        print(f"{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}\n")
        print(result.agent_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="PubMed Navigator Workshop CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("create-qdrant-collection", help="Create the Qdrant collection")

    subparsers.add_parser("delete-qdrant-collection", help="Delete the Qdrant collection")

    ingest_parser = subparsers.add_parser("ingest-data-to-qdrant", help="Ingest data into Qdrant")
    ingest_parser.add_argument(
        "--recreate", action="store_true", help="Recreate collection before ingesting"
    )
    ingest_parser.add_argument(
        "--only-new", action="store_true", help="Only ingest papers not already in the collection"
    )

    context_parser = subparsers.add_parser(
        "context-engineering-qdrant", help="Context engineering with Qdrant vector search"
    )
    context_parser.add_argument("question", nargs="+", help="Your question")
    context_parser.add_argument("--limit", type=int, default=5, help="Number of top results Qdrant will return (default: 5)")

    args = parser.parse_args()

    if args.command == "create-qdrant-collection":
        cmd_create_collection()
    elif args.command == "delete-qdrant-collection":
        cmd_delete_collection()
    elif args.command == "ingest-data-to-qdrant":
        cmd_ingest(args)
    elif args.command == "context-engineering-qdrant":
        cmd_context_engineering(args)


if __name__ == "__main__":
    main()
