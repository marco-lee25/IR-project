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

        # Allow the window to resize
        self.root.grid_rowconfigure(9, weight=1)  # Results text row expands vertically
        self.root.grid_columnconfigure(1, weight=1)  # Main content column expands horizontally

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = scibert_model(device)

        print("Initializing preprocess system...")
        self.preprocess = preprocess_sys(self.model, device)

        print("Initializing search engine...")
        self.se = search_engine.engine(self.model)

        self.summarizer = None

        self.create_widgets()

    def create_widgets(self):
        # Query input
        tk.Label(self.root, text="Search Query:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.query_entry = tk.Entry(self.root)
        self.query_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")  # Stretch horizontally

        # Search options
        self.use_bm25_var = tk.BooleanVar(value=True)
        self.use_bert_var = tk.BooleanVar(value=False)
        self.use_expansion_var = tk.BooleanVar(value=False)
        self.exp_syn_var = tk.BooleanVar(value=False)
        self.exp_sem_var = tk.BooleanVar(value=False)
        self.use_summary_var = tk.BooleanVar(value=False)

        tk.Checkbutton(self.root, text="Use BM25", variable=self.use_bm25_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Checkbutton(self.root, text="Use BERT", variable=self.use_bert_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        tk.Checkbutton(self.root, text="Use Expansion", variable=self.use_expansion_var, command=self.toggle_expansion).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.synonyms_check = tk.Checkbutton(self.root, text="Synonyms", variable=self.exp_syn_var, state="disabled")
        self.synonyms_check.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.semantic_check = tk.Checkbutton(self.root, text="Semantic", variable=self.exp_sem_var, state="disabled")
        self.semantic_check.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        tk.Checkbutton(self.root, text="Use Summarization", variable=self.use_summary_var).grid(row=3, column=0, padx=5, pady=5, sticky="w")

        # Top N selection
        tk.Label(self.root, text="Top N Results:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.top_n_var = tk.IntVar(value=5)
        top_n_options = [1, 3, 5, 10, 20]
        self.top_n_menu = ttk.Combobox(self.root, textvariable=self.top_n_var, values=top_n_options, state="readonly")
        self.top_n_menu.grid(row=4, column=1, padx=5, pady=5, sticky="ew")  # Stretch horizontally

        # BM25 and Vector weights
        tk.Label(self.root, text="BM25 Weight:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.bm25_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.bm25_weight_var, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.root, text="Vector Weight:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.vector_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.vector_weight_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        # Search button
        tk.Button(self.root, text="Search", command=self.perform_search).grid(row=7, column=0, columnspan=3, pady=10, sticky="ew")

        # Results display
        tk.Label(self.root, text="Results:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.results_text = tk.Text(self.root)
        self.results_text.grid(row=9, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")  # Stretch in all directions

        # Add scrollbar for results
        scrollbar = tk.Scrollbar(self.root, command=self.results_text.yview)
        scrollbar.grid(row=9, column=3, sticky="ns")
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(9, weight=1)  # Results text expands vertically
        self.root.grid_columnconfigure(1, weight=1)  # Main content column expands horizontally
        self.root.grid_columnconfigure(2, weight=1)  # Allow column 2 to expand too

    def toggle_expansion(self):
        state = "normal" if self.use_expansion_var.get() else "disabled"
        self.synonyms_check.config(state=state)
        self.semantic_check.config(state=state)
        if not self.use_expansion_var.get():
            self.exp_syn_var.set(False)
            self.exp_sem_var.set(False)

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
        top_n = self.top_n_var.get()
        use_summary = self.use_summary_var.get()
        bm25_weight = self.bm25_weight_var.get()
        vector_weight = self.vector_weight_var.get()

        self.summarizer = BartSummarizer() if use_summary else None

        if use_expansion and not (exp_syn or exp_sem):
            self.results_text.insert(tk.END, "Please specify an expansion method (Synonyms or Semantic).\n")
            return

        try:
            if use_expansion:
                processed_query = self.preprocess.process_query(query, use_semantic=exp_sem, use_synonyms=exp_syn)
                self.results_text.insert(tk.END, f"Expanded Query: {processed_query}\n\n")
                if not processed_query:
                    self.results_text.insert(tk.END, "Query expansion returned no results.\n")
                    return
                results = self.se.search(processed_query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
            else:
                processed_query = [query]
                results = self.se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)

            ranker = HybridRanker(bm25_weight=bm25_weight, vector_weight=vector_weight)
            if results and all('bm25_score' in doc and 'vector_score' in doc for doc in results):
                results = ranker.rank_documents(results)

            self.results_text.insert(tk.END, f"Query: {query}\nBM25: {use_bm25}, BERT: {use_bert}\n{'='*50}\n")
            if not results:
                self.results_text.insert(tk.END, "No results found.\n")
                return

            for i, doc in enumerate(results[:top_n], 1):
                output = [
                    f"RESULT {i}:",
                    f"Title: {doc['title']}",
                    f"Abstract: {doc['abstract'][:200]}..."
                ]
                if 'combined_score' in doc:
                    output.extend([
                        f"BM25: {doc['bm25_score']:.3f} (norm: {doc['normalized_bm25']:.3f})",
                        f"Vector: {doc['vector_score']:.3f} (norm: {doc['normalized_vector']:.3f})",
                        f"Combined: {doc['combined_score']:.3f}"
                    ])
                else:
                    output.append(f"BM25: {doc.get('bm25_score', 'N/A'):.3f}, Vector: {doc.get('vector_score', 'N/A'):.3f}")

                if self.summarizer:
                    summary = self.summarizer.summarize(doc['abstract'])
                    output.append(f"Summary: {summary}")

                self.results_text.insert(tk.END, "\n".join(output) + "\n" + "-"*50 + "\n")

        except Exception as e:
            self.results_text.insert(tk.END, f"Error during search: {str(e)}\n")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchUI(root)
    root.mainloop()