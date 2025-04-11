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
import textwrap

class SearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Paper Search")
        self.root.geometry("800x600")

        # Allow resizing
        self.root.grid_rowconfigure(9, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = scibert_model(self.device)

        print("Initializing preprocess system...")
        self.preprocess = preprocess_sys(self.model, self.device)

        print("Initializing search engine...")
        self.se = search_engine.engine(self.model)

        self.summarizer = None

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Search Query:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.query_entry = tk.Entry(self.root)
        self.query_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

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

        tk.Label(self.root, text="Top N Results:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.top_n_var = tk.IntVar(value=5)
        top_n_options = [1, 3, 5, 10, 20]
        self.top_n_menu = ttk.Combobox(self.root, textvariable=self.top_n_var, values=top_n_options, state="readonly")
        self.top_n_menu.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.root, text="BM25 Weight:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.bm25_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.bm25_weight_var, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(self.root, text="Vector Weight:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.vector_weight_var = tk.DoubleVar(value=0.5)
        tk.Entry(self.root, textvariable=self.vector_weight_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        tk.Button(self.root, text="Search", command=self.perform_search).grid(row=7, column=0, columnspan=3, pady=10, sticky="ew")

        tk.Label(self.root, text="Results:").grid(row=8, column=0, padx=5, pady=5, sticky="w")

        # Create a frame for the text widget and scrollbars
        text_frame = tk.Frame(self.root)
        text_frame.grid(row=9, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        # Text widget with wrapping and monospaced font
        self.results_text = tk.Text(text_frame, wrap="word", font=("Courier", 10))
        self.results_text.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar
        v_scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.config(yscrollcommand=v_scrollbar.set)

        # Horizontal scrollbar
        h_scrollbar = tk.Scrollbar(text_frame, orient="horizontal", command=self.results_text.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.results_text.config(xscrollcommand=h_scrollbar.set)

        self.root.grid_rowconfigure(9, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

    def toggle_expansion(self):
        state = "normal" if self.use_expansion_var.get() else "disabled"
        self.synonyms_check.config(state=state)
        self.semantic_check.config(state=state)
        if not self.use_expansion_var.get():
            self.exp_syn_var.set(False)
            self.exp_sem_var.set(False)

    def wrap_text(self, text, width=40):
        """Wrap text to a specified width, returning a list of lines."""
        return textwrap.wrap(text, width=width)

    def compare_rankings(self, results_hybrid, results_bm25, results_bert, ranker):
        """Display ranking comparison with wrapped titles and aligned scores."""
        output = ["=== RANKING COMPARISON ===\n"]
        column_width = 40
        separator_width = column_width * 3 + 10  # 3 columns + 2 separators (" | ")
        output.append(f"{'BM25 Order':<{column_width}} | {'Vector Order':<{column_width}} | {'Hybrid Order':<{column_width}}\n")
        output.append("-" * separator_width + "\n")

        bm25_sorted = sorted(results_bm25, key=lambda x: x['score'], reverse=True)
        vector_sorted = sorted(results_bert, key=lambda x: x['score'], reverse=True)
        hybrid_sorted = ranker.rank_documents(results_bm25, results_bert)

        for i in range(min(5, len(bm25_sorted))):
            # Wrap each title
            bm25_lines = self.wrap_text(bm25_sorted[i]['title'], width=column_width)
            vector_lines = self.wrap_text(vector_sorted[i]['title'], width=column_width)
            hybrid_lines = self.wrap_text(hybrid_sorted[i]['title'], width=column_width)

            # Determine the maximum number of lines needed
            max_lines = max(len(bm25_lines), len(vector_lines), len(hybrid_lines))

            # Pad shorter titles with empty lines to align scores
            bm25_lines.extend([''] * (max_lines - len(bm25_lines)))
            vector_lines.extend([''] * (max_lines - len(vector_lines)))
            hybrid_lines.extend([''] * (max_lines - len(hybrid_lines)))

            # Display each line of the titles
            for j in range(max_lines):
                output.append(f"{bm25_lines[j]:<{column_width}} | {vector_lines[j]:<{column_width}} | {hybrid_lines[j]:<{column_width}}\n")

            # Align scores under their respective columns
            bm25_score = f"BM25: {bm25_sorted[i]['score']:.2f}"
            vector_score = f"Vector: {vector_sorted[i]['score']:.2f}"
            combined_score = f"Combined: {hybrid_sorted[i].get('combined_score', 0):.2f}"
            output.append(f"{bm25_score:<{column_width}} | {vector_score:<{column_width}} | {combined_score:<{column_width}}\n")

            output.append("-" * separator_width + "\n")

        return "".join(output)

    def display_results(self, results, top_n, summarizer):
        """Mirror main.py's display_results with wrapped titles."""
        output = [f"\n=== HYBRID RANKING RESULTS ===\n"]

        for i, doc in enumerate(results[:top_n], 1):
            # Wrap the title
            wrapped_title = "\n".join(self.wrap_text(doc['title'], width=80))
            output.append(f"Rank {i}: {wrapped_title}\n")
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
        top_n = self.top_n_var.get()
        use_summary = self.use_summary_var.get()
        bm25_weight = self.bm25_weight_var.get()
        vector_weight = self.vector_weight_var.get()

        self.summarizer = BartSummarizer(self.device) if use_summary else None

        if use_expansion and not (exp_syn or exp_sem):
            self.results_text.insert(tk.END, "Please specify an expansion method (Synonyms or Semantic).\n")
            return

        try:
            self.results_text.insert(tk.END, f"Query: {query}\n")
            self.results_text.insert(tk.END, f"BM25: {use_bm25}, BERT: {use_bert}\n")

            if use_expansion:
                processed_query = self.preprocess.process_query(query, use_semantic=exp_sem, use_synonyms=exp_syn)
                self.results_text.insert(tk.END, f"Expanded Query: {processed_query}\n\n")
                if not processed_query:
                    self.results_text.insert(tk.END, "Query expansion returned no results.\n")
                    return
            else:
                processed_query = [query]

            results_hybrid = self.se.search(processed_query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)

            if not results_hybrid:
                self.results_text.insert(tk.END, "No results found.\n")
                return

            if use_bm25 and use_bert:
                results_bm25 = self.se.search(processed_query, use_bm25=True, use_bert=False, top_n=top_n)
                results_bert = self.se.search(processed_query, use_bm25=False, use_bert=True, top_n=top_n)
                ranker = HybridRanker(bm25_weight=bm25_weight, vector_weight=vector_weight)
                ranked_results = ranker.rank_documents(results_bm25, results_bert)

                comparison_text = self.compare_rankings(ranked_results, results_bm25, results_bert, ranker)
                self.results_text.insert(tk.END, comparison_text)

                detailed_text = self.display_results(ranked_results, top_n, self.summarizer)
                self.results_text.insert(tk.END, detailed_text)
            else:
                detailed_text = self.display_results(results_hybrid, top_n, self.summarizer)
                self.results_text.insert(tk.END, detailed_text)

        except Exception as e:
            self.results_text.insert(tk.END, f"Error during search: {str(e)}\n")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchUI(root)
    root.mainloop()