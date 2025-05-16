# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import json
import datetime
import os
import re
import hashlib
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Callable
from urllib.parse import urlparse
import threading
import time

logger = logging.getLogger(__name__)

class ExternalLearner:
    """
    Implements internet-sourced self-education capabilities,
    allowing the AI to download, verify, and incorporate external knowledge
    from articles, social media, PDFs, etc. into its learning model autonomously.
    """

    def __init__(self, config_path: Path = Path("config/external_learning.yaml")):
        self.config_path = config_path
        self.knowledge_base_path = Path("knowledge_base")
        self.max_articles_per_search = 10
        self.max_concurrent_downloads = 5
        self.trusted_domains = set()
        self.topic_experts = {}
        self.blacklisted_domains = set()
        self.download_history = []
        self.knowledge_index = {}
        self.verification_threshold = 0.7
        self.last_sync_time = None
        self.validation_metrics = {}
        self.content_processors = {}
        self.update_frequency = 24  # hours
        self.source_credibility = {}

        # Thread management
        self.active_downloads = 0
        self.download_lock = threading.Lock()

        # Search API keys and endpoints
        self.search_apis = {}

        # Load configuration if it exists
        self._load_config()

        # Ensure knowledge base directory exists
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        logger.info("External Learning Engine initialized")

    def _load_config(self):
        """Load configuration settings if available"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        self.knowledge_base_path = Path(config.get('knowledge_base_path', 'knowledge_base'))
                        self.max_articles_per_search = config.get('max_articles_per_search', 10)
                        self.max_concurrent_downloads = config.get('max_concurrent_downloads', 5)
                        self.trusted_domains = set(config.get('trusted_domains', []))
                        self.topic_experts = config.get('topic_experts', {})
                        self.blacklisted_domains = set(config.get('blacklisted_domains', []))
                        self.verification_threshold = config.get('verification_threshold', 0.7)
                        self.update_frequency = config.get('update_frequency', 24)

                        # Load API keys and endpoints
                        self.search_apis = config.get('search_apis', {})

                        logger.info(f"Loaded external learning configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def search_and_acquire_knowledge(self,
                                    topic: str,
                                    max_sources: int = None,
                                    domains: List[str] = None) -> Dict[str, Any]:
        """
        Search for and download knowledge about a specific topic

        Parameters:
        - topic: The topic to research
        - max_sources: Maximum number of sources to download (defaults to config)
        - domains: Specific domains to search within

        Returns:
        - Results of the search and download operation
        """
        try:
            # Set max sources from parameter or config
            max_sources = max_sources or self.max_articles_per_search

            # Search for relevant sources
            search_results = self._search_for_sources(topic, max_sources, domains)

            if not search_results.get("urls", []):
                logger.warning(f"No sources found for topic: {topic}")
                return {
                    "status": "no_sources",
                    "topic": topic,
                    "message": "No relevant sources found"
                }

            # Download and process each source
            urls = search_results.get("urls", [])
            logger.info(f"Found {len(urls)} sources for topic '{topic}'")

            # Process the sources
            download_results = self._download_and_process_sources(urls, topic)

            # Process successful downloads
            successful_downloads = [r for r in download_results if r.get("status") == "success"]

            # Index the new knowledge
            if successful_downloads:
                self._index_new_knowledge(successful_downloads, topic)

            # Return results
            return {
                "status": "completed",
                "topic": topic,
                "total_sources": len(urls),
                "successful_downloads": len(successful_downloads),
                "failed_downloads": len(download_results) - len(successful_downloads),
                "acquisition_time": datetime.datetime.now().isoformat(),
                "knowledge_items": [
                    {
                        "title": item.get("title", "Untitled"),
                        "url": item.get("url", ""),
                        "source": item.get("domain", "unknown"),
                        "credibility": item.get("credibility", 0)
                    }
                    for item in successful_downloads
                ]
            }

        except Exception as e:
            logger.error(f"Error acquiring knowledge on {topic}: {str(e)}")
            return {
                "status": "error",
                "topic": topic,
                "message": str(e)
            }

    def _search_for_sources(self,
                           topic: str,
                           max_sources: int,
                           domains: List[str] = None) -> Dict[str, Any]:
        """Search for sources on a topic using configured search APIs"""
        all_urls = set()

        # Use multiple search APIs for broader coverage
        for api_name, api_config in self.search_apis.items():
            try:
                urls = self._search_with_api(api_name, api_config, topic, domains)
                all_urls.update(urls)

                # Break if we have enough sources
                if len(all_urls) >= max_sources:
                    break
            except Exception as e:
                logger.error(f"Error using search API {api_name}: {str(e)}")

        # Filter for allowed domains
        filtered_urls = self._filter_urls(all_urls, domains)

        # Limit to max sources
        result_urls = list(filtered_urls)[:max_sources]

        return {
            "query": topic,
            "urls": result_urls,
            "count": len(result_urls)
        }

    def _search_with_api(self,
                        api_name: str,
                        api_config: Dict[str, Any],
                        topic: str,
                        domains: List[str] = None) -> List[str]:
        """Use a specific search API to find sources"""
        # This is a simplified version that would be expanded with actual API calls
        # In a real system, this would use the appropriate API client for each search engine

        # For demonstration, simulate search results
        urls = []

        # In a real implementation, these would be actual API calls
        if api_name == "google":
            # Simulate Google search results
            base_urls = [
                "https://en.wikipedia.org/wiki/",
                "https://www.sciencedirect.com/science/article/",
                "https://www.ncbi.nlm.nih.gov/pmc/articles/",
                "https://arxiv.org/abs/",
                "https://www.researchgate.net/publication/"
            ]

            # Generate simulated URLs based on topic
            topic_slug = topic.lower().replace(" ", "_")
            for i, base in enumerate(base_urls):
                urls.append(f"{base}{topic_slug}_{i}")

        elif api_name == "academic":
            # Simulate academic search results
            base_urls = [
                "https://scholar.google.com/scholar?q=",
                "https://www.semanticscholar.org/paper/",
                "https://link.springer.com/article/"
            ]

            # Generate simulated URLs
            topic_slug = topic.lower().replace(" ", "+")
            for i, base in enumerate(base_urls):
                urls.append(f"{base}{topic_slug}_{i}")

        # Filter by domains if specified
        if domains:
            domain_set = set(domains)
            urls = [url for url in urls if self._get_domain(url) in domain_set]

        return urls

    def _filter_urls(self, urls: Set[str], allowed_domains: List[str] = None) -> Set[str]:
        """Filter URLs based on trusted and blacklisted domains"""
        filtered_urls = set()

        for url in urls:
            domain = self._get_domain(url)

            # Skip blacklisted domains
            if domain in self.blacklisted_domains:
                logger.debug(f"Skipping blacklisted domain: {domain}")
                continue

            # Only include trusted domains or explicitly allowed domains
            if (not self.trusted_domains or domain in self.trusted_domains or
                (allowed_domains and domain in allowed_domains)):
                filtered_urls.add(url)

        return filtered_urls

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Remove www. prefix if present
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except:
            return ""

    def _download_and_process_sources(self, urls: List[str], topic: str) -> List[Dict[str, Any]]:
        """Download and process multiple sources concurrently"""
        download_results = []
        threads = []

        # Reset active downloads counter
        with self.download_lock:
            self.active_downloads = 0

        # Create a thread for each download
        for url in urls:
            # Wait if we've reached the concurrent download limit
            self._wait_for_download_slot()

            # Create and start thread
            thread = threading.Thread(
                target=self._download_source_thread,
                args=(url, topic, download_results)
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        return download_results

    def _wait_for_download_slot(self):
        """Wait until a download slot is available"""
        while True:
            with self.download_lock:
                if self.active_downloads < self.max_concurrent_downloads:
                    self.active_downloads += 1
                    return
            # Wait a bit before checking again
            time.sleep(0.1)

    def _download_source_thread(self, url: str, topic: str, results: List[Dict[str, Any]]):
        """Thread function to download and process a source"""
        try:
            result = self._download_and_process_source(url, topic)
            with self.download_lock:
                results.append(result)
                self.active_downloads -= 1
        except Exception as e:
            logger.error(f"Error in download thread for {url}: {str(e)}")
            with self.download_lock:
                results.append({
                    "status": "error",
                    "url": url,
                    "message": str(e)
                })
                self.active_downloads -= 1

    def _download_and_process_source(self, url: str, topic: str) -> Dict[str, Any]:
        """Download and process a single source"""
        try:
            # Get domain for logging
            domain = self._get_domain(url)
            logger.info(f"Downloading from {domain}: {url}")

            # Download the content
            # In a real implementation, this would use proper HTTP requests with headers, etc.
            # For now, simulate with a simplified request
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content = response.text
            except Exception as e:
                logger.error(f"Failed to download {url}: {str(e)}")
                return {
                    "status": "download_failed",
                    "url": url,
                    "message": str(e)
                }

            # Hash the content for deduplication
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            # Check if we already have this content
            for entry in self.download_history:
                if entry.get("content_hash") == content_hash:
                    logger.info(f"Duplicate content found: {url}")
                    return {
                        "status": "duplicate",
                        "url": url,
                        "original_url": entry.get("url"),
                        "message": "Duplicate content"
                    }

            # Process content based on type
            content_type = self._determine_content_type(url, content)
            extracted_data = self._extract_content(content, content_type)

            if not extracted_data.get("text"):
                logger.warning(f"No meaningful content extracted from {url}")
                return {
                    "status": "extraction_failed",
                    "url": url,
                    "message": "Failed to extract meaningful content"
                }

            # Verify content quality and relevance
            verification_result = self._verify_content_quality(
                extracted_data, topic
            )

            if not verification_result.get("is_valid", False):
                logger.warning(f"Content verification failed for {url}: {verification_result.get('reason')}")
                return {
                    "status": "verification_failed",
                    "url": url,
                    "message": verification_result.get("reason", "Content verification failed")
                }

            # Calculate source credibility
            credibility = self._calculate_source_credibility(url, domain, extracted_data)

            # Save the content
            file_path = self._save_content(extracted_data, url, topic, content_type)

            # Record in download history
            download_record = {
                "url": url,
                "domain": domain,
                "topic": topic,
                "title": extracted_data.get("title", "Untitled"),
                "content_hash": content_hash,
                "content_type": content_type,
                "download_time": datetime.datetime.now().isoformat(),
                "file_path": str(file_path),
                "credibility": credibility
            }

            self.download_history.append(download_record)

            # Return success result
            return {
                "status": "success",
                "url": url,
                "domain": domain,
                "title": extracted_data.get("title", "Untitled"),
                "content_type": content_type,
                "file_path": str(file_path),
                "credibility": credibility,
                "word_count": len(extracted_data.get("text", "").split())
            }

        except Exception as e:
            logger.error(f"Error processing source {url}: {str(e)}")
            return {
                "status": "processing_error",
                "url": url,
                "message": str(e)
            }

    def _determine_content_type(self, url: str, content: str) -> str:
        """Determine the type of content (HTML, PDF, etc.)"""
        # Check URL extension
        if url.lower().endswith(".pdf"):
            return "pdf"
        elif url.lower().endswith((".doc", ".docx")):
            return "document"
        elif url.lower().endswith((".ppt", ".pptx")):
            return "presentation"
        elif url.lower().endswith((".csv", ".xls", ".xlsx")):
            return "data"

        # Check for HTML content
        if "<html" in content.lower() or "<body" in content.lower():
            return "html"

        # Default to text
        return "text"

    def _extract_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Extract structured content based on content type"""
        # Get the appropriate processor for this content type
        processor = self.content_processors.get(content_type)

        # If a processor is registered, use it
        if processor:
            return processor(content)

        # Otherwise, use basic extraction methods
        if content_type == "html":
            return self._extract_from_html(content)
        elif content_type == "pdf":
            # In a real implementation, this would use a PDF parser
            return {"text": "PDF content placeholder", "title": "PDF Document"}
        elif content_type in ["document", "presentation", "data"]:
            # In a real implementation, this would use appropriate parsers
            return {"text": f"{content_type.capitalize()} content placeholder", "title": f"{content_type.capitalize()} File"}
        else:
            # Plain text
            lines = content.split("\n")
            title = lines[0] if lines else "Untitled"
            return {"text": content, "title": title}

    def _extract_from_html(self, html_content: str) -> Dict[str, Any]:
        """Extract readable content from HTML"""
        # In a real implementation, this would use a proper HTML parser
        # For simplicity, use regex to extract title and basic content

        # Extract title
        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Untitled"

        # Extract body
        body_match = re.search(r"<body.*?>(.*?)</body>", html_content, re.IGNORECASE | re.DOTALL)
        body = body_match.group(1) if body_match else html_content

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", body)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "title": title,
            "text": text
        }

    def _verify_content_quality(self, content: Dict[str, Any], topic: str) -> Dict[str, Any]:
        """Verify content quality and relevance to topic"""
        text = content.get("text", "")

        # Check content length
        if len(text) < 100:
            return {
                "is_valid": False,
                "reason": "Content too short"
            }

        # In a real implementation, this would apply more sophisticated checks:
        # - Relevance scoring using NLP
        # - Sentiment analysis
        # - Factual verification against trusted sources
        # - Plagiarism detection

        # For this example, use a simple keyword check
        topic_keywords = set(topic.lower().split())

        # Count keyword occurrences
        keyword_count = sum(1 for keyword in topic_keywords if keyword.lower() in text.lower())

        # Calculate coverage ratio
        if topic_keywords:
            coverage = keyword_count / len(topic_keywords)
        else:
            coverage = 0

        if coverage < 0.5:  # Require at least 50% keyword coverage
            return {
                "is_valid": False,
                "reason": f"Low relevance to topic ({coverage:.1%} keyword coverage)"
            }

        return {
            "is_valid": True,
            "relevance_score": coverage
        }

    def _calculate_source_credibility(self, url: str, domain: str, content: Dict[str, Any]) -> float:
        """Calculate credibility score for a source"""
        # Base credibility starts from domain reputation
        if domain in self.source_credibility:
            base_credibility = self.source_credibility[domain]
        else:
            # Default for unknown domains
            base_credibility = 0.5

        # Check if domain is in trusted expert domains for this topic
        for topic, experts in self.topic_experts.items():
            if domain in experts:
                base_credibility += 0.2
                break

        # In a real implementation, additional factors would be considered:
        # - Link analysis
        # - Publication date
        # - Author credentials
        # - Citation analysis
        # - Content quality metrics

        # Cap at 0.95 - even trusted sources aren't perfect
        return min(0.95, max(0.1, base_credibility))

    def _save_content(self, content: Dict[str, Any], url: str, topic: str, content_type: str) -> Path:
        """Save extracted content to the knowledge base"""
        # Create topic directory if it doesn't exist
        topic_slug = topic.lower().replace(" ", "_")
        topic_dir = self.knowledge_base_path / topic_slug
        topic_dir.mkdir(exist_ok=True)

        # Create a filename based on content title
        title = content.get("title", "untitled")
        safe_title = re.sub(r"[^\w\s-]", "", title).strip().lower()
        safe_title = re.sub(r"[-\s]+", "-", safe_title)

        # Add timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{safe_title}_{timestamp}.json"

        # Create full file path
        file_path = topic_dir / filename

        # Save content as JSON with metadata
        data = {
            "title": title,
            "url": url,
            "topic": topic,
            "content_type": content_type,
            "extraction_time": datetime.datetime.now().isoformat(),
            "text": content.get("text", ""),
            "metadata": {
                "domain": self._get_domain(url)
            }
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return file_path

    def _index_new_knowledge(self, successful_downloads: List[Dict[str, Any]], topic: str):
        """Index newly downloaded knowledge for easy retrieval"""
        # In a real implementation, this would use a proper vector database
        # or search index. For simplicity, use a basic dictionary index.

        topic_slug = topic.lower().replace(" ", "_")

        if topic_slug not in self.knowledge_index:
            self.knowledge_index[topic_slug] = []

        for download in successful_downloads:
            self.knowledge_index[topic_slug].append({
                "title": download.get("title", "Untitled"),
                "url": download.get("url", ""),
                "file_path": download.get("file_path", ""),
                "credibility": download.get("credibility", 0),
                "domain": download.get("domain", "unknown"),
                "word_count": download.get("word_count", 0),
                "indexed_time": datetime.datetime.now().isoformat()
            })

    def retrieve_knowledge(self,
                          topic: str,
                          query: str = None,
                          min_credibility: float = 0.0) -> Dict[str, Any]:
        """
        Retrieve knowledge about a specific topic

        Parameters:
        - topic: The topic to retrieve
        - query: Optional search query within the topic
        - min_credibility: Minimum credibility score for sources

        Returns:
        - Retrieved knowledge
        """
        try:
            topic_slug = topic.lower().replace(" ", "_")

            # Check if we have knowledge on this topic
            if topic_slug not in self.knowledge_index or not self.knowledge_index[topic_slug]:
                logger.warning(f"No knowledge available for topic: {topic}")
                return {
                    "status": "no_knowledge",
                    "topic": topic,
                    "message": "No knowledge available on this topic"
                }

            # Get all knowledge entries for this topic
            entries = self.knowledge_index[topic_slug]

            # Filter by credibility
            if min_credibility > 0:
                entries = [e for e in entries if e.get("credibility", 0) >= min_credibility]

            # Search within entries if query provided
            if query:
                matched_entries = []
                query_terms = query.lower().split()

                for entry in entries:
                    file_path = entry.get("file_path", "")
                    if file_path and Path(file_path).exists():
                        # Load content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            text = content.get("text", "").lower()

                            # Simple keyword matching
                            match_count = sum(1 for term in query_terms if term in text)

                            if match_count > 0:
                                # Copy entry and add relevance score
                                matched_entry = entry.copy()
                                matched_entry["relevance"] = match_count / len(query_terms)
                                matched_entries.append(matched_entry)

                # Sort by relevance
                entries = sorted(matched_entries, key=lambda x: x.get("relevance", 0), reverse=True)

            # Return results
            return {
                "status": "success",
                "topic": topic,
                "query": query,
                "sources_count": len(entries),
                "sources": entries,
                "retrieved_time": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error retrieving knowledge on {topic}: {str(e)}")
            return {
                "status": "error",
                "topic": topic,
                "message": str(e)
            }

    def extract_facts(self, topic: str, query: str = None) -> Dict[str, Any]:
        """
        Extract factual statements about a topic from the knowledge base

        Parameters:
        - topic: The topic to extract facts about
        - query: Optional search query to narrow down facts

        Returns:
        - Extracted facts with source citations
        """
        try:
            # Retrieve knowledge
            knowledge = self.retrieve_knowledge(topic, query, min_credibility=0.5)

            if knowledge.get("status") != "success":
                return {
                    "status": knowledge.get("status"),
                    "topic": topic,
                    "message": knowledge.get("message", "Failed to retrieve knowledge")
                }

            sources = knowledge.get("sources", [])
            if not sources:
                return {
                    "status": "no_sources",
                    "topic": topic,
                    "message": "No reliable sources found for fact extraction"
                }

            # In a real implementation, this would use advanced NLP to extract facts
            # For demonstration, create simulated facts from the sources
            facts = []

            for source in sources[:5]:  # Limit to top 5 sources
                file_path = source.get("file_path", "")
                if file_path and Path(file_path).exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        text = content.get("text", "")

                        # Simulate fact extraction with sentences containing key topic terms
                        # In a real implementation, this would use proper NLP
                        sentences = re.split(r'[.!?]', text)
                        topic_terms = set(topic.lower().split())

                        for sentence in sentences:
                            sentence = sentence.strip()
                            if len(sentence) > 20:  # Skip short fragments
                                # Check if sentence contains topic terms
                                sentence_lower = sentence.lower()
                                has_topic_terms = any(term in sentence_lower for term in topic_terms)

                                if has_topic_terms:
                                    facts.append({
                                        "fact": sentence,
                                        "source": {
                                            "title": source.get("title", "Untitled"),
                                            "url": source.get("url", ""),
                                            "credibility": source.get("credibility", 0)
                                        }
                                    })

            # Deduplicate facts (simplistic approach)
            unique_facts = []
            seen_facts = set()

            for fact in facts:
                # Create a simplified representation for deduplication
                simplified = ' '.join(fact["fact"].lower().split())
                if simplified not in seen_facts and len(simplified) > 20:
                    seen_facts.add(simplified)
                    unique_facts.append(fact)

            # Sort by source credibility
            sorted_facts = sorted(
                unique_facts,
                key=lambda x: x.get("source", {}).get("credibility", 0),
                reverse=True
            )

            return {
                "status": "success",
                "topic": topic,
                "query": query,
                "facts_count": len(sorted_facts),
                "facts": sorted_facts,
                "extraction_time": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error extracting facts for {topic}: {str(e)}")
            return {
                "status": "error",
                "topic": topic,
                "message": str(e)
            }

    def learn_from_topic(self, topic: str, deep_learning: bool = False) -> Dict[str, Any]:
        """
        Comprehensive learning process for a topic:
        1. Search and acquire knowledge
        2. Extract facts and concepts
        3. Integrate with existing knowledge

        Parameters:
        - topic: Topic to learn about
        - deep_learning: Whether to conduct deep learning (more sources, relations)

        Returns:
        - Results of the learning process
        """
        try:
            learning_stats = {
                "topic": topic,
                "start_time": datetime.datetime.now().isoformat(),
                "stages": {}
            }

            # Stage 1: Search and acquire knowledge
            max_sources = 20 if deep_learning else 10
            acquisition_result = self.search_and_acquire_knowledge(topic, max_sources)
            learning_stats["stages"]["acquisition"] = {
                "status": acquisition_result.get("status"),
                "sources_found": acquisition_result.get("total_sources", 0),
                "successful_downloads": acquisition_result.get("successful_downloads", 0)
            }

            if acquisition_result.get("status") not in ["completed", "partial"]:
                # If acquisition failed completely, abort
                learning_stats["status"] = "failed_acquisition"
                learning_stats["message"] = acquisition_result.get("message", "Knowledge acquisition failed")
                return learning_stats

            # Stage 2: Extract facts and concepts
            facts_result = self.extract_facts(topic)
            learning_stats["stages"]["facts_extraction"] = {
                "status": facts_result.get("status"),
                "facts_extracted": facts_result.get("facts_count", 0)
            }

            # Stage 3: Integrate with existing knowledge
            # In a real implementation, this would use knowledge graphs or other
            # knowledge representation structures
            integration_result = self._integrate_knowledge(topic, facts_result.get("facts", []))
            learning_stats["stages"]["integration"] = integration_result

            # Calculate overall success
            stages = learning_stats["stages"]
            if (stages["acquisition"]["status"] == "completed" and
                stages["facts_extraction"]["status"] == "success" and
                stages["integration"]["status"] == "success"):
                overall_status = "success"
            elif stages["acquisition"]["successful_downloads"] > 0:
                overall_status = "partial"
            else:
                overall_status = "failed"

            learning_stats["status"] = overall_status
            learning_stats["end_time"] = datetime.datetime.now().isoformat()

            logger.info(f"Learning process for '{topic}' completed with status: {overall_status}")
            return learning_stats

        except Exception as e:
            logger.error(f"Error in learning process for topic {topic}: {str(e)}")
            return {
                "status": "error",
                "topic": topic,
                "message": str(e),
                "start_time": learning_stats.get("start_time"),
                "end_time": datetime.datetime.now().isoformat()
            }

    def _integrate_knowledge(self, topic: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate extracted facts into the knowledge system"""
        # This is a simplified placeholder for knowledge integration
        # In a real implementation, this would:
        # 1. Update a knowledge graph
        # 2. Establish relationships between concepts
        # 3. Resolve conflicts with existing knowledge
        # 4. Update confidence scores for facts

        if not facts:
            return {
                "status": "no_facts",
                "relationships_created": 0
            }

        # Simulate knowledge integration
        relationships_created = min(len(facts), 10)  # Simulate creating relationships

        return {
            "status": "success",
            "facts_integrated": len(facts),
            "relationships_created": relationships_created,
            "integration_time": datetime.datetime.now().isoformat()
        }

    def update_knowledge(self, force: bool = False) -> Dict[str, Any]:
        """
        Update existing knowledge based on configured update frequency

        Parameters:
        - force: Force update regardless of schedule

        Returns:
        - Update results
        """
        try:
            # Check if update is needed based on schedule
            current_time = datetime.datetime.now()

            if not force and self.last_sync_time:
                last_sync = datetime.datetime.fromisoformat(self.last_sync_time)
                hours_since_update = (current_time - last_sync).total_seconds() / 3600

                if hours_since_update < self.update_frequency:
                    return {
                        "status": "skipped",
                        "reason": f"Last update was {hours_since_update:.1f} hours ago (update frequency: {self.update_frequency} hours)"
                    }

            # Get all topics in knowledge base
            topics = list(self.knowledge_index.keys())
            topics = [t.replace("_", " ") for t in topics]

            if not topics:
                return {
                    "status": "no_topics",
                    "message": "No topics found in knowledge base"
                }

            logger.info(f"Updating knowledge for {len(topics)} topics")

            # Update each topic
            update_results = {}
            for topic in topics:
                update_results[topic] = self.learn_from_topic(topic, deep_learning=False)

            # Update last sync time
            self.last_sync_time = current_time.isoformat()

            return {
                "status": "completed",
                "topics_updated": len(topics),
                "results": update_results,
                "update_time": current_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Error updating knowledge: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def register_content_processor(self, content_type: str, processor_func: Callable) -> bool:
        """Register a custom content processor for a specific content type"""
        try:
            self.content_processors[content_type] = processor_func
            logger.info(f"Registered content processor for type: {content_type}")
            return True
        except Exception as e:
            logger.error(f"Error registering content processor: {str(e)}")
            return False

    def set_source_credibility(self, domain: str, credibility: float) -> bool:
        """Set the credibility score for a specific source domain"""
        try:
            self.source_credibility[domain] = max(0.0, min(1.0, credibility))
            logger.info(f"Set credibility for domain '{domain}' to {credibility}")
            return True
        except Exception as e:
            logger.error(f"Error setting source credibility: {str(e)}")
            return False

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            # Count topics and sources
            topic_count = len(self.knowledge_index)
            total_sources = sum(len(sources) for sources in self.knowledge_index.values())

            # Calculate average credibility
            all_sources = []
            for sources in self.knowledge_index.values():
                all_sources.extend(sources)

            if all_sources:
                avg_credibility = sum(s.get("credibility", 0) for s in all_sources) / len(all_sources)
            else:
                avg_credibility = 0

            # Count by content type
            content_types = {}
            for download in self.download_history:
                content_type = download.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

            # Get top domains
            domain_counts = {}
            for download in self.download_history:
                domain = download.get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            return {
                "topics_count": topic_count,
                "sources_count": total_sources,
                "average_credibility": avg_credibility,
                "content_types": content_types,
                "top_domains": dict(top_domains),
                "last_update": self.last_sync_time,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting knowledge statistics: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
