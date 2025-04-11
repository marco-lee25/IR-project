import tkinter as tk
from tkinter import ttk
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer
from ranking_system.ranking_function import HybridRanker
from scibert.model import scibert_model
import torch

class SearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Paper Search")
        self.root.geometry("800x600")

        # Allow resizing
        self.root.grid_rowconfigure(9, weight=1)  # Results text row expands vertically
        self.root.grid_columnconfigure(1, weight=1)  # Main content column expands horizontally

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = scibert_model(self.device)

        print("Initializing preprocess system...")
        self.preprocess = preprocess_sys(self.model, self.device)

        print("Initializing search engine...")
        self.se = search_engine.engine(self.model)

        self.summarizer = None

        self.create_widgets()

    def create_widgets(self):
        # Query input
        tk.Label(self.root, text="Search Query:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.query_entry = tk.Entry(self.root)
        self.query_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        # Search options
        self.use_bm25_var = tk.BooleanVar(value=True)
        self.use_bert_var = tk.BooleanVar(value=False)
        self.use_expansion_var = tk.BooleanVar(value=False)
        self.exp_syn_var = tk.BooleanVar(value=False)
        self.exp_sem_var = tk.BooleanVar(value=False)
        self.use_summary_var = tk.BooleanVar(value=False)
        self.sem_method_var = tk.StringVar(value="GenAI (1)")  # Default to GenAI (1)

        tk.Checkbutton(self.root, text="Use BM25", variable=self.use_bm25_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Checkbutton(self.root, text="Use BERT", variable=self.use_bert_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        tk.Checkbutton(self.root, text="Use Expansion", variable=self.use_expansion_var, command=self.toggle_expansion).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.synonyms_check = tk.Checkbutton(self.root, text="Synonyms", variable=self.exp_syn_var, state="disabled")
        self.synonyms_check.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.semantic_check = tk.Checkbutton(self.root, text="Semantic", variable=self.exp_sem_var, state="disabled", command=self.toggle_semantic_method)
        self.semantic_check.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Semantic method selection (dropdown)
        tk.Label(self.root, text="Semantic Method:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.sem_method_menu = ttk.Combobox(self.root, textvariable=self.sem_method_var, state="disabled")
        self.sem_method_menu['values'] = ("Database-vector (0)", "glove-word2vec (1)", "GenAI (2)")
        self.sem_method_menu.current(0)  # Default to GenAI
        self.sem_method_menu.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        tk.Checkbutton(self.root, text="Use Summarization", variable=self.use_summary_var).grid(row=4, column=0, padx=5, pady=5, sticky="w")

        # Top N selection
        tk.Label(self.root, text="Top N Results:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.top_n_var = tk.IntVar(value=5)
        top_n_options = [1, 3, 5, 10, 20]
        self.top_n_menu = ttk.Combobox(self.root, textvariable=self.top_n_var, values=top_n_options, state="readonly")
        self.top_n_menu.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # BM25 and Vector weights
        tk.Label(self.root, text="BM25 Weight:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.bm25_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.bm25_weight_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.root, text="Vector Weight:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.vector_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.vector_weight_var, width=10).grid(row=7, column=1, padx=5, pady=5, sticky="ew")

        # Search button
        tk.Button(self.root, text="Search", command=self.perform_search).grid(row=8, column=0, columnspan=3, pady=10, sticky="ew")

        # Results display
        tk.Label(self.root, text="Results:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.results_text = tk.Text(self.root, wrap="word")  # Wrap text for readability
        self.results_text.grid(row=10, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Add scrollbar
        scrollbar = tk.Scrollbar(self.root, command=self.results_text.yview)
        scrollbar.grid(row=10, column=3, sticky="ns")
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Configure grid weights
        self.root.grid_rowconfigure(10, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

    def toggle_expansion(self):
        state = "normal" if self.use_expansion_var.get() else "disabled"
        self.synonyms_check.config(state=state)
        self.semantic_check.config(state=state)
        if not self.use_expansion_var.get():
            self.exp_syn_var.set(False)
            self.exp_sem_var.set(False)
            self.sem_method_menu.config(state="disabled")

    def toggle_semantic_method(self):
        state = "normal" if self.exp_sem_var.get() else "disabled"
        self.sem_method_menu.config(state=state)
        if not self.exp_sem_var.get():
            self.sem_method_var.set("GenAI (1)")  # Reset to default (GenAI)

    def compare_rankings(self, results_hybrid, results_bm25, results_bert, ranker):
        """Display ranking comparison in the UI."""
        output = ["=== RANKING COMPARISON ===\n"]
        output.append(f"{'BM25 Order':<40} | {'Vector Order':<40} | {'Hybrid Order':<40}\n")
        output.append("-" * 120 + "\n")

        # Sort each result set independently
        bm25_sorted = sorted(results_bm25, key=lambda x: x['score'], reverse=True)
        vector_sorted = sorted(results_bert, key=lambda x: x['score'], reverse=True)
        hybrid_sorted = ranker.rank_documents(results_bm25, results_bert)

        for i in range(min(5, len(bm25_sorted))):
            bm25_title = bm25_sorted[i]['title'][:35] + ('...' if len(bm25_sorted[i]['title']) > 35 else '')
            vector_title = vector_sorted[i]['title'][:35] + ('...' if len(vector_sorted[i]['title']) > 35 else '')
            hybrid_title = hybrid_sorted[i]['title'][:35] + ('...' if len(hybrid_sorted[i]['title']) > 35 else '')

            output.append(f"{bm25_title:<40} | {vector_title:<40} | {hybrid_title:<40}\n")
            output.append(f"BM25: {bm25_sorted[i]['score']:.2f} | "
                          f"Vector: {vector_sorted[i]['score']:.2f} | "
                          f"Combined: {hybrid_sorted[i].get('combined_score', 0):.2f}\n")
            output.append("-" * 120 + "\n")

        return "".join(output)

    def display_results(self, results, top_n, summarizer):
        """Display detailed results in the UI."""
        output = ["\n=== HYBRID RANKING RESULTS ===\n"]

        for i, doc in enumerate(results[:top_n], 1):
            output.append(f"Rank {i}: {doc['title']}\n")
            output.append(f"Abstract: {doc['abstract'][:150]}{'...' if len(doc['abstract']) > 150 else ''}\n")

            score_info = []
            if 'bm25_score' in doc:
                score_info.append(f"BM25: {doc['bm25_score']:.2f}")
            if 'vector_score' in doc:
                score_info.append(f"Vector: {doc['vector_score']:.2f}")
            if 'combined_score' in doc:
                score_info.append(f"Combined: {doc['combined_score']:.2f}")
                if 'normalized_bm25' in doc and 'normalized_vector' in doc:
                    score_info.append(f"(Norm: BM25={doc['normalized_bm25']:.2f}, Vector={doc['normalized_vector']:.2f})")

            if score_info:
                output.append("Scores: " + " | ".join(score_info) + "\n")

            if summarizer:
                output.append(f"Summary: {summarizer.summarize(doc['abstract'])}\n")

            output.append("=" * 80 + "\n")

        return "".join(output)

    def perform_search(self):
        self.results_text.delete(1.0, tk.END)

        query = self.query_entry.get().strip()
        if not query:
            self.results_text.insert(tk.END, "Please enter a query.\n")
            return

        use_bm25 = self.use_bm25_var.get()
        use_bert = self.use_bert_var.get()
        use_expansion = self.use_expansion_var.get()
        exp_syn = self.exp_syn_var.get()
        exp_sem = self.exp_sem_var.get()
        # Extract the integer from the selected semantic method string (e.g., "GenAI (1)" -> 1)
        sem_method_str = self.sem_method_var.get()
        sem_method = int(sem_method_str.split("(")[1].strip(")"))
        top_n = self.top_n_var.get()
        use_summary = self.use_summary_var.get()
        bm25_weight = self.bm25_weight_var.get()
        vector_weight = self.vector_weight_var.get()

        self.summarizer = BartSummarizer("cpu") if use_summary else None

        if use_expansion and not (exp_syn or exp_sem):
            self.results_text.insert(tk.END, "Please specify an expansion method (Synonyms or Semantic).\n")
            return

        try:
            # Process query with expansion if enabled
            if use_expansion:
                processed_query = self.preprocess.process_query(query, use_semantic=exp_sem, use_synonyms=exp_syn, sem_method=sem_method)
                self.results_text.insert(tk.END, f"Expanded Query: {processed_query}\n\n")
                if not processed_query:
                    self.results_text.insert(tk.END, "Query expansion returned no results.\n")
                    return
            else:
                processed_query = [query]

            # Perform search
            results_hybrid = self.se.search(processed_query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
            self.results_text.insert(tk.END, f"Query: {query}\nBM25: {use_bm25}, BERT: {use_bert}\n{'='*50}\n")

            if not results_hybrid:
                self.results_text.insert(tk.END, "No results found.\n")
                return

            # Compare rankings if both BM25 and BERT are enabled
            if use_bm25 and use_bert:
                results_bm25 = self.se.search(processed_query, use_bm25=True, use_bert=False, top_n=top_n)
                results_bert = self.se.search(processed_query, use_bm25=False, use_bert=True, top_n=top_n)
                ranker = HybridRanker(bm25_weight=bm25_weight, vector_weight=vector_weight)
                ranked_results = ranker.rank_documents(results_bm25, results_bert)

                comparison_text = self.compare_rankings(ranked_results, results_bm25, results_bert, ranker)
                self.results_text.insert(tk.END, comparison_text)

                # Display detailed hybrid results
                detailed_text = self.display_results(ranked_results, top_n, self.summarizer)
                self.results_text.insert(tk.END, detailed_text)
            else:
                # Display results without comparison
                detailed_text = self.display_results(results_hybrid, top_n, self.summarizer)
                self.results_text.insert(tk.END, detailed_text)

        except Exception as e:
            self.results_text.insert(tk.END, f"Error during search: {str(e)}\n")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchUI(root)
    root.mainloop()