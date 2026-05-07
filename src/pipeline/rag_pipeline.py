"""
RAG Pipeline Module
Orchestrates the complete Retrieval-Augmented Generation pipeline
using LlamaIndex with custom components for production-grade performance.
"""

import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    source_documents: list[dict] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """
    End-to-end RAG pipeline that orchestrates:
    1. Document ingestion and preprocessing
    2. Semantic/hybrid chunking
    3. Embedding generation with optimized models
    4. Vector store indexing (ChromaDB/FAISS)
    5. Hybrid retrieval with Reciprocal Rank Fusion
    6. Cross-encoder reranking
    7. LLM-based answer generation with citation

    Built on LlamaIndex for flexible component composition.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self._components = {}
        self._index = None
        self._is_initialized = False
        logger.info("RAGPipeline created. Call initialize() to set up components.")

    def _load_config(self, config_path: str) -> dict:
        """Load pipeline configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing RAG pipeline components...")

        # Initialize embedding manager
        from src.embedding.embedding_manager import EmbeddingManager

        self._components["embedding_manager"] = EmbeddingManager(self.config)

        # Initialize vector store
        backend = self.config.get("vectorstore", {}).get("backend", "chroma")
        if backend == "chroma":
            from src.vectorstore.chroma_store import ChromaVectorStore

            self._components["vector_store"] = ChromaVectorStore(self.config)
        else:
            from src.vectorstore.faiss_store import FAISSVectorStore

            self._components["vector_store"] = FAISSVectorStore(self.config)

        # Initialize chunker
        strategy = self.config.get("chunking", {}).get("strategy", "semantic")
        if strategy == "semantic":
            from src.chunking.semantic_chunker import SemanticChunker

            self._components["chunker"] = SemanticChunker(self.config)
        elif strategy == "recursive":
            from src.chunking.recursive_chunker import RecursiveChunker

            self._components["chunker"] = RecursiveChunker(self.config)
        else:
            from src.chunking.hybrid_chunker import HybridChunker

            self._components["chunker"] = HybridChunker(self.config)

        # Initialize retriever
        from src.retrieval.retriever import HybridRetriever

        self._components["retriever"] = HybridRetriever(
            self.config,
            self._components["vector_store"],
            self._components["embedding_manager"],
        )

        # Initialize reranker
        from src.retrieval.reranker import Reranker

        self._components["reranker"] = Reranker(self.config)

        # Initialize LlamaIndex components
        self._setup_llama_index()

        self._is_initialized = True
        logger.info("RAG pipeline fully initialized")

    def _setup_llama_index(self):
        """Configure LlamaIndex with custom components."""
        from llama_index.core import Settings, VectorStoreIndex, StorageContext
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Configure embedding model for LlamaIndex
        embed_model_name = self.config.get("embedding", {}).get(
            "model_name", "BAAI/bge-large-en-v1.5"
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Configure LLM
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama")

        if provider == "groq":
            import os
            from llama_index.llms.groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", llm_config.get("api_key", ""))
            Settings.llm = Groq(
                model=llm_config.get("model_name", "llama-3.1-8b-instant"),
                api_key=api_key,
                temperature=llm_config.get("temperature", 0.1),
                max_tokens=llm_config.get("max_tokens", 2048),
            )
        elif provider == "huggingface":
            from llama_index.llms.huggingface import HuggingFaceLLM

            Settings.llm = HuggingFaceLLM(
                model_name=llm_config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2"),
                tokenizer_name=llm_config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2"),
                context_window=llm_config.get("context_window", 4096),
                max_new_tokens=llm_config.get("max_tokens", 2048),
                generate_kwargs={"temperature": llm_config.get("temperature", 0.1)},
                device_map="auto",
            )
        elif provider == "ollama":
            from llama_index.llms.ollama import Ollama

            Settings.llm = Ollama(
                model=llm_config.get("model_name", "mistral"),
                temperature=llm_config.get("temperature", 0.1),
                request_timeout=300.0,
            )

        # Configure chunking for LlamaIndex
        chunk_config = self.config.get("chunking", {}).get("recursive", {})
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_config.get("chunk_size", 512),
            chunk_overlap=chunk_config.get("chunk_overlap", 50),
        )

        logger.info("LlamaIndex settings configured")

    def ingest_documents(self, source_path: str) -> int:
        """
        Ingest documents from a directory into the pipeline.

        Process:
        1. Load documents (PDF, text, etc.)
        2. Chunk using configured strategy
        3. Generate embeddings
        4. Store in vector database
        5. Build BM25 index for sparse retrieval

        Returns:
            Number of chunks indexed.
        """
        if not self._is_initialized:
            self.initialize()

        from src.ingestion.document_loader import DocumentLoader

        # Load documents
        loader = DocumentLoader(self.config)
        documents = loader.load_directory(source_path)
        logger.info(f"Loaded {len(documents)} documents from {source_path}")

        # Chunk documents
        chunker = self._components["chunker"]
        all_chunks = []

        for doc in documents:
            metadata = {
                "source": doc.metadata.source,
                "page_number": doc.metadata.page_number or 0,
                "section": doc.metadata.section or "",
                "filing_type": doc.metadata.filing_type or "",
                "has_tables": doc.metadata.has_tables,
            }
            chunks = chunker.chunk(doc.content, metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        if not all_chunks:
            logger.warning("No chunks generated. Check document content.")
            return 0

        # Generate embeddings
        embedding_manager = self._components["embedding_manager"]
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = embedding_manager.encode_documents(texts)

        # Store in vector database
        vector_store = self._components["vector_store"]
        metadatas = [chunk.get("metadata", {}) for chunk in all_chunks]

        if hasattr(vector_store, "add_documents"):
            vector_store.add_documents(
                documents=texts,
                embeddings=embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else embeddings,
                metadatas=metadatas,
            )

        # Build BM25 index for sparse retrieval
        retriever = self._components["retriever"]
        retriever.build_sparse_index(texts, metadatas)

        logger.info(f"Successfully indexed {len(all_chunks)} chunks")
        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """
        Query the RAG pipeline with a natural language question.

        Process:
        1. Hybrid retrieval (dense + sparse with RRF)
        2. Cross-encoder reranking
        3. Context construction
        4. LLM generation with citations

        Args:
            question: Natural language question.
            top_k: Override number of retrieved chunks.
            metadata_filter: Filter results by metadata.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        if not self._is_initialized:
            self.initialize()

        # Step 1: Hybrid retrieval
        retriever = self._components["retriever"]
        retrieval_results = retriever.retrieve(
            query=question,
            top_k=top_k or self.config.get("retrieval", {}).get("top_k", 10),
            metadata_filter=metadata_filter,
        )

        if not retrieval_results:
            return RAGResponse(
                answer="I could not find relevant information to answer this question.",
                source_documents=[],
                retrieval_scores=[],
                metadata={"query": question, "num_results": 0},
            )

        # Step 2: Reranking
        if use_reranker:
            reranker = self._components["reranker"]
            retrieval_results = reranker.rerank(question, retrieval_results)

        # Step 3: Construct context
        context = self._build_context(retrieval_results)

        # Step 4: Generate answer
        answer = self._generate_answer(question, context)

        return RAGResponse(
            answer=answer,
            source_documents=retrieval_results,
            retrieval_scores=[r.get("score", 0) for r in retrieval_results],
            metadata={
                "query": question,
                "num_results": len(retrieval_results),
                "fusion_method": self.config.get("retrieval", {})
                .get("fusion", {})
                .get("method", "reciprocal_rank"),
                "reranker_used": use_reranker,
            },
        )

    def _build_context(self, results: list[dict]) -> str:
        """Build a structured context string from retrieval results."""
        context_parts = []

        for i, result in enumerate(results):
            source = result.get("metadata", {}).get("source", "Unknown")
            page = result.get("metadata", {}).get("page_number", "")
            section = result.get("metadata", {}).get("section", "")

            header = f"[Source {i + 1}: {Path(source).name}"
            if page:
                header += f", Page {page}"
            if section:
                header += f", Section: {section}"
            header += "]"

            context_parts.append(f"{header}\n{result['content']}")

        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LlamaIndex LLM with structured prompt."""
        from llama_index.core import Settings
        from llama_index.core.llms import ChatMessage, MessageRole

        system_prompt = self.config.get("llm", {}).get(
            "system_prompt",
            "Answer questions based on the provided context. Cite your sources.",
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Please provide a detailed answer based strictly on the context above. "
            f"Cite the specific source numbers [Source X] that support your answer."
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            response = Settings.llm.chat(messages)
            return str(response.message.content)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {str(e)}"

    def get_pipeline_stats(self) -> dict:
        """Get statistics about the pipeline state."""
        stats = {
            "initialized": self._is_initialized,
            "config": {
                "chunking_strategy": self.config.get("chunking", {}).get("strategy"),
                "embedding_model": self.config.get("embedding", {}).get("model_name"),
                "vector_backend": self.config.get("vectorstore", {}).get("backend"),
                "fusion_method": self.config.get("retrieval", {})
                .get("fusion", {})
                .get("method"),
            },
        }

        if self._is_initialized:
            vector_store = self._components.get("vector_store")
            if vector_store and hasattr(vector_store, "get_stats"):
                stats["vector_store"] = vector_store.get_stats()

        return stats
