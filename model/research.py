import os
import logging
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Changed from tqdm.notebook to regular tqdm
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# 1) EMBEDDING MODEL (SPECTER2)
# ---------------------------------------------------
class Specter2Encoder:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load Specter2 tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoModel.from_pretrained("allenai/specter2_base").to(self.device)
        logger.info("Loaded Specter2 model for scientific embeddings")

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text."""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512  # Limit input length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.model.config.hidden_size)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---------------------------------------------------
# 2) DATA SOURCES (ENHANCED)
# ---------------------------------------------------
class JournalFetcher:
    def __init__(self, email: str=None, springer_api_key: str=None):
        self.session = requests.Session()
        self.email = email
        self.springer_api_key = springer_api_key

    def fetch_openalex(self, min_works=500, limit=200) -> List[Dict]:
        """Fetch journals from OpenAlex API."""
        url = "https://api.openalex.org/sources"
        params = {
            "filter": f"type:journal,works_count:>{min_works}",
            "sort": "cited_by_count:desc",
            "per-page": limit
        }
        try:
            resp = self.session.get(url, params=params).json()
            return [{
                "id": j["id"],
                "name": j["display_name"],
                "publisher": j.get("host_organization", "Unknown"),
                "url": j.get("homepage_url", ""),
                "subjects": [c["display_name"] for c in j.get("concepts", [])],
                "description": f"{j['display_name']} ({j.get('host_organization', '')}) - " +
                              f"Publishes research in: {', '.join([c['display_name'] for c in j.get('concepts', [])[:5]])}"
            } for j in resp.get("results", [])]
        except Exception as e:
            logger.error(f"Error fetching OpenAlex data: {e}")
            return []

    def fetch_crossref(self, limit=50) -> List[Dict]:
        """Fetch journals from Crossref API."""
        url = "https://api.crossref.org/journals"
        params = {"rows": limit}
        if self.email:
            params["mailto"] = self.email
        try:
            resp = self.session.get(url, params=params).json()
            items = resp.get("message", {}).get("items", [])
            return [{
                "id": it.get("ISSN", [""])[0],
                "name": it.get("title", "Unknown"),
                "publisher": it.get("publisher", "Unknown"),
                "url": it.get("URL", ""),
                "subjects": it.get("subjects", []),
                "description": f"{it.get('title', 'Unknown')} ({it.get('publisher', 'Unknown')})"
            } for it in items]
        except Exception as e:
            logger.error(f"Error fetching Crossref data: {e}")
            return []

    def fetch_springer(self, limit=50) -> List[Dict]:
        """Fetch journals from Springer API."""
        if not self.springer_api_key:
            return []
        url = "https://api.springernature.com/meta/v2/json"
        params = {"api_key": self.springer_api_key, "q": "type:Journal", "p": limit}
        try:
            resp = self.session.get(url, params=params).json()
            recs = resp.get("records", [])
            return [{
                "id": r.get("issn", ""),
                "name": r.get("title", "Unknown"),
                "publisher": r.get("publisher", "Springer"),
                "url": r.get("url", ""),
                "subjects": r.get("subject", []),
                "description": f"{r.get('title', 'Unknown')} ({r.get('publisher', 'Springer')})"
            } for r in recs]
        except Exception as e:
            logger.error(f"Error fetching Springer data: {e}")
            return []

    def all_journals(self) -> List[Dict]:
        """Combine and deduplicate journals from all sources."""
        oa = self.fetch_openalex()
        cr = self.fetch_crossref()
        sp = self.fetch_springer()
        seen = set()
        out = []
        for lst in (oa, cr, sp):
            for j in lst:
                key = j["name"].lower()
                if key not in seen:
                    seen.add(key)
                    out.append(j)
        logger.info(f"Fetched {len(out)} unique journals")
        return out

# ---------------------------------------------------
# 3) RECOMMENDER SYSTEM
# ---------------------------------------------------
class JournalRecommender:
    def __init__(self):
        self.encoder = Specter2Encoder()

    def recommend(self, title: str, abstract: str, top_k: int=10) -> List[Dict]:
        """Recommend journals based on spectral similarity."""
        # Generate paper embedding
        paper_text = f"Title: {title}\nAbstract: {abstract}"
        paper_embedding = self.encoder.get_embedding(paper_text)

        # Fetch and prepare journals
        fetcher = JournalFetcher()
        journals = fetcher.all_journals()

        # Calculate similarities
        for journal in tqdm(journals, desc="Calculating similarities"):
            journal_text = f"Journal: {journal['name']}\nPublisher: {journal['publisher']}\n" + \
                          f"Subjects: {', '.join(journal['subjects'][:5])}"
            journal_embedding = self.encoder.get_embedding(journal_text)
            journal["score"] = self.encoder.cosine_similarity(paper_embedding, journal_embedding)

        # Sort and return top recommendations
        journals.sort(key=lambda x: x["score"], reverse=True)
        return journals[:top_k]

# ---------------------------------------------------
# 4) USAGE EXAMPLE
# ---------------------------------------------------
if __name__ == "__main__":
    # Example paper details
    title = "Efficient Text Summarization Using Transformer-Based Models"
    abstract = (
        "Transformer architectures have revolutionized natural language processing (NLP) tasks, particularly in text summarization. "
        "This study proposes an optimized transformer-based model for abstractive text summarization that achieves state-of-the-art performance "
        "while reducing computational overhead. Our approach leverages a lightweight variant of the BERT encoder and a novel attention mechanism "
        "to generate concise and coherent summaries from long documents. We evaluate our model on benchmark datasets such as CNN/DailyMail and XSum, "
        "achieving ROUGE scores of 47.2 (ROUGE-1), 23.8 (ROUGE-2), and 40.5 (ROUGE-L). Additionally, we demonstrate a 30% reduction in inference time "
        "compared to standard transformer models like T5 and BART. The proposed method is particularly effective for summarizing technical and scientific texts, "
        "making it suitable for applications in automated report generation and document analysis."
    )

    # Initialize recommender and generate recommendations
    recommender = JournalRecommender()
    top10 = recommender.recommend(title, abstract, top_k=10)

    # Print results
    print("\n=== Top 10 Journal Recommendations ===")
    for idx, j in enumerate(top10, 1):
        print(f"{idx}. {j['name']} (Score: {j['score']:.4f})")
        print(f"   Publisher: {j['publisher']}")
        print(f"   URL: {j['url']}")
        print(f"   Subjects: {', '.join(j['subjects'][:5])}\n")