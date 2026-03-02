create-qdrant-collection:
	uv run python -m workshop.cli create-qdrant-collection

delete-qdrant-collection:
	uv run python -m workshop.cli delete-qdrant-collection

ingest-data-to-qdrant:
	uv run python -m workshop.cli ingest-data-to-qdrant $(if $(RECREATE),--recreate) $(if $(ONLY_NEW),--only-new)

context-engineering-qdrant:
	uv run python -m workshop.cli context-engineering-qdrant $(if $(LIMIT),--limit $(LIMIT)) "$(QUESTION)"
