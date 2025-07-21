import os
from dotenv import load_dotenv
load_dotenv()

import requests
from typing import List, Dict, Any, Union
from urllib.parse import urlparse
import tempfile
import logging
from pathlib import Path

# Core dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader
)
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# YouTube transcript extraction
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    print("Warning: youtube-transcript-api not installed. YouTube support disabled.")

# PDF extraction from web
try:
    import PyPDF2
    import io
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDF2 not installed. Web PDF support may be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSourceTextExtractor:
    """Extract text from multiple sources and create a searchable vector index."""
    
    def __init__(self, embedding_model=None, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text extractor.
        
        Args:
            embedding_model: Embedding model (defaults to OpenAI)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vectorstore = None
        self.documents = []
        self.sources = []
    
    def extract_youtube_transcript(self, url: str) -> List[Document]:
        """Extract transcript from YouTube video."""
        if not YOUTUBE_AVAILABLE:
            raise ImportError("youtube-transcript-api required for YouTube support")
        
        try:
            # Extract video ID from URL
            video_id = self._extract_youtube_id(url)
            if not video_id:
                raise ValueError(f"Could not extract video ID from URL: {url}")
            
            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine transcript text
            full_text = " ".join([entry['text'] for entry in transcript])
            
            # Create document
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": url,
                    "type": "youtube",
                    "video_id": video_id,
                    "title": f"YouTube Video {video_id}"
                }
            )
            
            logger.info(f"Extracted transcript from YouTube: {url}")
            return [doc]
            
        except Exception as e:
            logger.error(f"Error extracting YouTube transcript from {url}: {str(e)}")
            return []
    
    def _extract_youtube_id(self, url: str) -> str:
        """Extract YouTube video ID from URL."""
        import re
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def extract_web_content(self, url: str) -> List[Document]:
        """Extract content from web pages and web-hosted PDFs."""
        try:
            # Check if URL points to a PDF
            if url.lower().endswith('.pdf') or 'pdf' in url.lower():
                return self._extract_web_pdf(url)
            else:
                # Regular web page
                loader = WebBaseLoader([url])
                docs = loader.load()
                
                # Update metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": url,
                        "type": "webpage"
                    })
                
                logger.info(f"Extracted content from webpage: {url}")
                return docs
                
        except Exception as e:
            logger.error(f"Error extracting web content from {url}: {str(e)}")
            return []
    
    def _extract_web_pdf(self, url: str) -> List[Document]:
        """Extract text from PDF hosted on the web."""
        try:
            # Download PDF content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            try:
                # Use PyPDFLoader to extract text
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Update metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": url,
                        "type": "web_pdf"
                    })
                
                logger.info(f"Extracted content from web PDF: {url}")
                return docs
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Error extracting web PDF from {url}: {str(e)}")
            return []
    
    def extract_local_document(self, file_path: str) -> List[Document]:
        """Extract text from local documents (.pdf, .docx)."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type and use appropriate loader
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            docs = loader.load()
            
            # Update metadata
            for doc in docs:
                doc.metadata.update({
                    "source": str(file_path),
                    "type": f"local_{file_path.suffix[1:]}",
                    "filename": file_path.name
                })
            
            logger.info(f"Extracted content from local document: {file_path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error extracting local document {file_path}: {str(e)}")
            return []
        
    def extract_web_search(self, query: str, num_results: int = 5) -> List[Document]:
        """
        Perform web search and extract content from search results.
        
        Args:
            query: Search query string
            num_results: Number of search results to process
            
        Returns:
            List of Document objects from search results
        """
        try:
            
            # Initialize search wrapper
            search = DuckDuckGoSearchAPIWrapper(max_results=num_results)
            
            # Perform search
            search_results = search.results(query, max_results=num_results)
            
            documents = []
            
            for result in search_results:
                try:
                    # Extract URL from search result
                    url = result.get('link', '')
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    
                    if url:
                        # Try to extract full content from the page
                        web_docs = self.extract_web_content(url)
                        
                        # If web extraction failed, use snippet
                        if not web_docs and snippet:
                            doc = Document(
                                page_content=snippet,
                                metadata={
                                    "source": url,
                                    "type": "web_search_snippet",
                                    "title": title,
                                    "search_query": query
                                }
                            )
                            documents.append(doc)
                        else:
                            # Update metadata for successfully extracted docs
                            for doc in web_docs:
                                doc.metadata.update({
                                    "search_query": query,
                                    "search_title": title,
                                    "type": "web_search_page"
                                })
                            documents.extend(web_docs)
                    
                except Exception as e:
                    logger.warning(f"Error processing search result {url}: {str(e)}")
                    continue
            
            logger.info(f"Extracted content from {len(documents)} web search results for query: '{query}'")
            return documents
        
        except Exception as e:
            logger.error(f"Error performing web search for '{query}': {str(e)}")
            return []
    
    def process_sources(self, sources: List[str], **kwargs) -> List[Document]:
        """
        Process multiple sources and extract text from all.
        
        Args:
            sources: List of URLs, file paths, or YouTube URLs
            
        Returns:
            List of Document objects
        """
        all_documents = []
        
        for source in sources:

            if source not in self.sources:
                logger.info(f"Processing source: {source}")
                
                # Determine source type and process accordingly
                if self._is_youtube_url(source):
                    docs = self.extract_youtube_transcript(source)
                elif self._is_web_url(source):
                    docs = self.extract_web_content(source)
                else:
                    # Assume local file
                    docs = self.extract_local_document(source)
                
                all_documents.extend(docs)

                self.sources.append(source)

            else:
                logging.info(f"Source {source} already indexed")
        
        return all_documents
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube URL."""
        return 'youtube.com' in url.lower() or 'youtu.be' in url.lower()
    
    def _is_web_url(self, url: str) -> bool:
        """Check if string is a web URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def create_vectorstore_index(self, sources: List[str], **kwargs) -> InMemoryVectorStore:
        """
        Create a searchable vector index from multiple sources.
        
        Args:
            sources: List of URLs, file paths, or YouTube URLs
            
        Returns:
            InMemoryVectorStore with indexed documents
        """
        logger.info("Starting text extraction and indexing process")
        
        # Extract documents from all sources
        documents = self.process_sources(sources, **kwargs)
        
        if not documents:
            logger.warning("No documents were successfully extracted")
            return InMemoryVectorStore.from_documents([], self.embedding_model)
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunked_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunked_docs)} text chunks")
        
        # Create vector store
        logger.info("Creating vector store index")
        self.vectorstore = InMemoryVectorStore.from_documents(
            chunked_docs, 
            self.embedding_model
        )
        
        # Store documents for reference
        self.documents = chunked_docs
        
        logger.info("Vector store index created successfully")
        return self.vectorstore
    
    def search(self, query: str, k: int = 5, web_search=False) -> List[Document]:
        """
        Search the vector store for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            raise ValueError("No vector store index created. Call create_vectorstore_index first.")
        
        indexed_documents = self.vectorstore.similarity_search(query, k=k)

        return indexed_documents
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("No vector store index created. Call create_vectorstore_index first.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if not self.documents:
            return {"total_chunks": 0, "sources": []}
        
        sources = set(doc.metadata.get("source", "unknown") for doc in self.documents)
        source_types = set(doc.metadata.get("type", "unknown") for doc in self.documents)
        
        return {
            "total_chunks": len(self.documents),
            "unique_sources": len(sources),
            "source_types": list(source_types),
            "sources": list(sources)
        }