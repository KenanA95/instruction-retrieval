import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BaseRetriever:
    """Base class for retrieval components."""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the base retriever.
        
        Args:
            embedding_model: Sentence transformer model to use for embeddings
        """
        self.model = SentenceTransformer(embedding_model)
    
    def _get_embeddings(self, texts):
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, convert_to_tensor=False)
    
    def _retrieve_top_k(self, query_embedding, corpus_embeddings, corpus_texts, top_k=3):
        """
        Retrieve top-k most similar texts based on embeddings.
        
        Args:
            query_embedding: Embedding of the query
            corpus_embeddings: Embeddings of the corpus
            corpus_texts: Corpus text strings
            top_k: Number of results to return
            
        Returns:
            List of top-k most similar texts
        """
        # Reshape query embedding for similarity calculation
        query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        
        # Get indices of top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return top-k texts and their scores
        return [corpus_texts[i] for i in top_indices]

class TextbookRetriever(BaseRetriever):
    """Retriever for medical textbook passages."""
    
    def __init__(self, textbook_path, top_k=3, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the textbook retriever.
        
        Args:
            textbook_path: Path to the directory containing textbook passages
            top_k: Number of passages to retrieve
            embedding_model: Sentence transformer model to use
        """
        super().__init__(embedding_model)
        self.textbook_path = textbook_path
        self.top_k = top_k
        self.passages = []
        self.passage_embeddings = None
        
        # Load the passages
        self._load_passages()
    
    def _load_passages(self):
        """Load textbook passages from the textbook path."""
        passage_file = os.path.join(self.textbook_path, 'passages.json')
        
        if os.path.exists(passage_file):
            with open(passage_file, 'r') as f:
                self.passages = json.load(f)
            
            # Get passage texts for embedding
            passage_texts = [p['text'] for p in self.passages]
            
            # Generate embeddings
            self.passage_embeddings = self._get_embeddings(passage_texts)
            
            print(f"Loaded {len(self.passages)} textbook passages")
        else:
            print(f"No passage file found at {passage_file}")
    
    def retrieve(self, query):
        """
        Retrieve relevant textbook passages for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of relevant passages
        """
        if not self.passages:
            return []
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Get passage texts
        passage_texts = [p['text'] for p in self.passages]
        
        # Retrieve top-k passages
        top_passages = self._retrieve_top_k(
            query_embedding, 
            self.passage_embeddings, 
            self.passages, 
            self.top_k
        )
        
        return top_passages

class CaseLawRetriever(BaseRetriever):
    """Retriever for legal case law."""
    
    def __init__(self, case_law_path, top_k=3, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the case law retriever.
        
        Args:
            case_law_path: Path to the directory containing case law documents
            top_k: Number of cases to retrieve
            embedding_model: Sentence transformer model to use
        """
        super().__init__(embedding_model)
        self.case_law_path = case_law_path
        self.top_k = top_k
        self.cases = []
        self.case_embeddings = None
        
        # Load the cases
        self._load_cases()
    
    def _load_cases(self):
        """Load case law from the case law path."""
        case_file = os.path.join(self.case_law_path, 'cases.json')
        
        if os.path.exists(case_file):
            with open(case_file, 'r') as f:
                self.cases = json.load(f)
            
            # Get case texts for embedding
            case_texts = [c['text'] for c in self.cases]
            
            # Generate embeddings
            self.case_embeddings = self._get_embeddings(case_texts)
            
            print(f"Loaded {len(self.cases)} case law documents")
        else:
            print(f"No case file found at {case_file}")
    
    def retrieve(self, query):
        """
        Retrieve relevant case law for a query.
        
        Args:
            query: Query text
            
        Returns:
            List of relevant cases
        """
        if not self.cases:
            return []
        
        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]
        
        # Get case texts
        case_texts = [c['text'] for c in self.cases]
        
        # Retrieve top-k cases
        top_cases = self._retrieve_top_k(
            query_embedding, 
            self.case_embeddings, 
            self.cases, 
            self.top_k
        )
        
        return top_cases

if __name__ == "__main__":
    # Example usage for textbook retriever
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval components")
    parser.add_argument("--type", choices=['textbook', 'caselaw'], required=True, 
                        help="Type of retriever to test")
    parser.add_argument("--path", required=True, help="Path to corpus directory")
    parser.add_argument("--query", required=True, help="Query to test retrieval")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve")
    args = parser.parse_args()
    
    if args.type == 'textbook':
        retriever = TextbookRetriever(args.path, args.top_k)
        
        # Test retrieval
        results = retriever.retrieve(args.query)
        
        print(f"\nQuery: {args.query}")
        print(f"Top {args.top_k} results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Text: {result.get('text', '')[:200]}...")
    
    elif args.type == 'caselaw':
        retriever = CaseLawRetriever(args.path, args.top_k)
        
        # Test retrieval
        results = retriever.retrieve(args.query)
        
        print(f"\nQuery: {args.query}")
        print(f"Top {args.top_k} results:")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Case: {result.get('case_name', 'N/A')}")
            print(f"Citation: {result.get('citation', 'N/A')}")
            print(f"Text: {result.get('text', '')[:200]}...") 