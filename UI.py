import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import textwrap
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer, deepseekSummarizer
from ranking_system.ranking_function import HybridRanker
from models.model import scibert_model, deepseek_model
import torch

class SearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Paper Search")
        self.root.geometry("1280x720")

        # Allow resizing
        self.root.grid_rowconfigure(11, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Initialize search systems
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = scibert_model(self.device)
        self.deepseek_model, self.deepseek_tokenizer = deepseek_model(self.device).get_model()
        self.summarizer = BartSummarizer("cpu")

        print("Initializing preprocess system...")
        self.preprocess = preprocess_sys(self.model, self.deepseek_model, self.deepseek_tokenizer, self.device)

        print("Initializing search engine...")
        self.se = search_engine.engine(self.model)

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("TCombobox", font=("Helvetica", 10))

        # Query input
        ttk.Label(main_frame, text="Search Query:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.query_entry = ttk.Entry(main_frame)
        self.query_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Search options
        self.use_bm25_var = tk.BooleanVar(value=True)
        self.use_bert_var = tk.BooleanVar(value=False)
        self.use_expansion_var = tk.BooleanVar(value=False)
        self.exp_syn_var = tk.BooleanVar(value=False)
        self.exp_sem_var = tk.BooleanVar(value=False)
        self.use_summary_var = tk.BooleanVar(value=False)
        self.sem_method_var = tk.StringVar(value="GenAI (2)")

        ttk.Checkbutton(main_frame, text="Use BM25", variable=self.use_bm25_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(main_frame, text="Use BERT", variable=self.use_bert_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Checkbutton(main_frame, text="Use Expansion", variable=self.use_expansion_var, command=self.toggle_expansion).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.synonyms_check = ttk.Checkbutton(main_frame, text="Synonyms", variable=self.exp_syn_var, state="disabled")
        self.synonyms_check.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.semantic_check = ttk.Checkbutton(main_frame, text="Semantic", variable=self.exp_sem_var, state="disabled", command=self.toggle_semantic_method)
        self.semantic_check.grid(row=2, column=2, padx=5, pady=5, sticky="w")

        # Semantic method dropdown
        ttk.Label(main_frame, text="Semantic Method:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.sem_method_menu = ttk.Combobox(main_frame, textvariable=self.sem_method_var, state="disabled")
        self.sem_method_menu['values'] = ("Database-vector (0)", "GoogleNews-word2vec (1)", "GenAI (2)")
        self.sem_method_menu.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        ttk.Checkbutton(main_frame, text="Use Summarization", variable=self.use_summary_var).grid(row=4, column=0, padx=5, pady=5, sticky="w")

        # Top N selection
        ttk.Label(main_frame, text="Top N Results:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.top_n_var = tk.IntVar(value=5)
        self.top_n_menu = ttk.Combobox(main_frame, textvariable=self.top_n_var, values=[1, 3, 5, 10, 20], state="readonly")
        self.top_n_menu.grid(row=5, column=1, padx=5, pady=5, sticky="ew")

        # Weights
        ttk.Label(main_frame, text="BM25 Weight:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.bm25_weight_var = tk.DoubleVar(value=0.5)
        ttk.Entry(main_frame, textvariable=self.bm25_weight_var, width=10).grid(row=6, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(main_frame, text="Vector Weight:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.vector_weight_var = tk.DoubleVar(value=0.5)
        ttk.Entry(main_frame, textvariable=self.vector_weight_var, width=10).grid(row=7, column=1, padx=5, pady=5, sticky="ew")

        # Font selection
        ttk.Label(main_frame, text="Font:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.font_var = tk.StringVar(value="Courier")
        font_menu = ttk.Combobox(main_frame, textvariable=self.font_var, values=["Courier", "Helvetica", "Times"], state="readonly")
        font_menu.grid(row=8, column=1, padx=5, pady=5, sticky="ew")
        font_menu.bind("<<ComboboxSelected>>", self.update_font)

        ttk.Label(main_frame, text="Font Size:").grid(row=8, column=2, padx=5, pady=5, sticky="w")
        self.font_size_var = tk.IntVar(value=12)
        size_menu = ttk.Combobox(main_frame, textvariable=self.font_size_var, values=[8, 10, 12, 14, 16], state="readonly")
        size_menu.grid(row=8, column=3, padx=5, pady=5, sticky="ew")
        size_menu.bind("<<ComboboxSelected>>", self.update_font)

        # Buttons
        ttk.Button(main_frame, text="Search", command=self.perform_search).grid(row=9, column=0, columnspan=4, pady=10, sticky="ew")
        ttk.Button(main_frame, text="Clear Display", command=self.clear_display).grid(row=10, column=0, columnspan=4, pady=5, sticky="ew")

        # Results display
        ttk.Label(main_frame, text="Results:").grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.result_font = Font(family="Courier", size=14)
        self.results_text = tk.Text(main_frame, wrap="none", font=self.result_font, height=20)
        self.results_text.grid(row=12, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")

        # Scrollbars
        y_scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.results_text.yview)
        y_scrollbar.grid(row=12, column=4, sticky="ns")
        self.results_text.config(yscrollcommand=y_scrollbar.set)

        x_scrollbar = ttk.Scrollbar(main_frame, orient="horizontal", command=self.results_text.xview)
        x_scrollbar.grid(row=13, column=0, columnspan=4, sticky="ew")
        self.results_text.config(xscrollcommand=x_scrollbar.set)

        # Bind events
        self.results_text.bind("<Configure>", self.on_resize)
        main_frame.grid_rowconfigure(12, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

    def update_font(self, event=None):
        """Update font and refresh display."""
        font_name = self.font_var.get()
        font_size = self.font_size_var.get()
        self.result_font.configure(family=font_name, size=font_size)
        self.perform_search()

    def clear_display(self):
        self.results_text.delete(1.0, tk.END)

    def on_resize(self, event=None):
        self.perform_search()

    def get_text_width_in_chars(self):
        """Calculate the width of the results_text widget in characters."""
        widget_width = self.results_text.winfo_width()
        char_width = self.result_font.measure("M") or 10  # Fallback
        num_chars = max(30, (widget_width - 20) // char_width)  # Padding
        return num_chars

    def wrap_text(self, text, width):
        """Wrap text to fit within the specified width."""
        if not text:
            return [""]
        wrapped = textwrap.wrap(text, width=max(1, width), break_long_words=True)
        return wrapped or [""]

    def compare_rankings(self, results_hybrid, results_bm25, results_bert, ranker):
        """Display ranking comparison with three columns, dynamically adjusted."""
        output = ["=== RANKING COMPARISON ===\n"]

        # Calculate column width
        total_width = self.get_text_width_in_chars()
        column_width = max(15, (total_width - 8) // 3)  # 3 columns, 2 separators (" | ")
        separator_width = column_width * 3 + 8  # Adjust for separators

        # Header
        output.append(
            f"{'BM25 Order':<{column_width}} | "
            f"{'Vector Order':<{column_width}} | "
            f"{'Hybrid Order':<{column_width}}\n"
        )
        output.append("-" * separator_width + "\n")

        # Sort results
        bm25_sorted = sorted(results_bm25, key=lambda x: x['score'], reverse=True)
        vector_sorted = sorted(results_bert, key=lambda x: x['score'], reverse=True)
        hybrid_sorted = ranker.rank_documents(results_bm25, results_bert)

        for i in range(min(5, len(bm25_sorted))):
            # Wrap titles
            bm25_lines = self.wrap_text(bm25_sorted[i]['title'], width=column_width - 2)
            vector_lines = self.wrap_text(vector_sorted[i]['title'], width=column_width - 2)
            hybrid_lines = self.wrap_text(hybrid_sorted[i]['title'], width=column_width - 2)

            # Pad to equal length
            max_lines = max(len(bm25_lines), len(vector_lines), len(hybrid_lines))
            bm25_lines.extend([''] * (max_lines - len(bm25_lines)))
            vector_lines.extend([''] * (max_lines - len(vector_lines)))
            hybrid_lines.extend([''] * (max_lines - len(hybrid_lines)))

            # Display titles
            for j in range(max_lines):
                output.append(
                    f"{bm25_lines[j]:<{column_width}} | "
                    f"{vector_lines[j]:<{column_width}} | "
                    f"{hybrid_lines[j]:<{column_width}}\n"
                )

            # Scores
            bm25_score = f"BM25: {bm25_sorted[i]['score']:.2f}"
            vector_score = f"Vector: {vector_sorted[i]['score']:.2f}"
            hybrid_score = f"Combined: {hybrid_sorted[i].get('combined_score', 0):.2f}"
            output.append(
                f"{bm25_score:<{column_width}} | "
                f"{vector_score:<{column_width}} | "
                f"{hybrid_score:<{column_width}}\n"
            )

            output.append("-" * separator_width + "\n")

        return "".join(output)

    def display_results(self, results, top_n, summarizer):
        """Display detailed results with dynamic wrapping."""
        output = ["=== FINAL RESULTS ===\n\n"]
        wrap_width = self.get_text_width_in_chars() - 4

        for i, doc in enumerate(results[:top_n], 1):
            title_lines = self.wrap_text(f"Rank {i}: {doc['title']}", width=wrap_width)
            output.append("  " + "\n  ".join(title_lines) + "\n")

            id = self.wrap_text(f"ID : {doc['id']}", width=wrap_width)
            output.append("  " + "\n  ".join(id) + "\n")
            
            abstract = doc.get('abstract', 'No abstract available')
            if len(abstract) > wrap_width * 3:
                abstract = abstract[:wrap_width * 3 - 3] + "..."
            abstract_lines = self.wrap_text(f"Abstract: {abstract}", width=wrap_width)
            output.append("  " + "\n  ".join(abstract_lines) + "\n")

            scores = []
            if 'bm25_score' in doc:
                scores.append(f"BM25: {doc['bm25_score']:.2f}")
            if 'vector_score' in doc:
                scores.append(f"Vector: {doc['vector_score']:.2f}")
            if 'combined_score' in doc:
                scores.append(f"Combined: {doc['combined_score']:.2f}")
                if 'normalized_bm25' in doc and 'normalized_vector' in doc:
                    scores.append(f"Norm: BM25={doc['normalized_bm25']:.2f}, Vector={doc['normalized_vector']:.2f}")
            if scores:
                score_lines = self.wrap_text("Scores: " + "; ".join(scores), width=wrap_width)
                output.append("  " + "\n  ".join(score_lines) + "\n")

            if summarizer and 'abstract' in doc:
                summary = summarizer.summarize(doc['abstract'])
                summary_lines = self.wrap_text(f"Summary: {summary}", width=wrap_width)
                output.append("  " + "\n  ".join(summary_lines) + "\n")

            output.append("=" * min(wrap_width, 80) + "\n\n")

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
        sem_method_str = self.sem_method_var.get()
        sem_method = int(sem_method_str.split("(")[1].strip(")"))
        top_n = self.top_n_var.get()
        use_summary = self.use_summary_var.get()
        bm25_weight = self.bm25_weight_var.get()
        vector_weight = self.vector_weight_var.get()

        summarizer = self.summarizer if use_summary else None
        if use_expansion and not (exp_syn or exp_sem):
            self.results_text.insert(tk.END, "Please specify an expansion method (Synonyms or Semantic).\n")
            return

        try:
            if use_expansion:
                processed_query = self.preprocess.process_query(query, use_semantic=exp_sem, use_synonyms=exp_syn, sem_method=sem_method)
                wrap_width = self.get_text_width_in_chars() - 4
                query_lines = self.wrap_text(f"Expanded Query: {processed_query}", width=wrap_width)
                self.results_text.insert(tk.END, "\n".join(query_lines) + "\n\n")
                if not processed_query:
                    self.results_text.insert(tk.END, "Query expansion returned no results.\n")
                    return
            else:
                processed_query = [query]

            results_hybrid = self.se.search(processed_query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
            wrap_width = self.get_text_width_in_chars() - 4
            info_lines = self.wrap_text(f"Query: {query}\nBM25: {use_bm25}, BERT: {use_bert}", width=wrap_width)
            self.results_text.insert(tk.END, "\n".join(info_lines) + "\n\n")

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

                detailed_text = self.display_results(ranked_results, top_n, summarizer)
                self.results_text.insert(tk.END, detailed_text)
            else:
                detailed_text = self.display_results(results_hybrid, top_n, summarizer)
                self.results_text.insert(tk.END, detailed_text)

        except Exception as e:
            self.results_text.insert(tk.END, f"Error during search: {str(e)}\n")
            raise

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
            self.sem_method_var.set("GenAI (2)")

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchUI(root)
    root.mainloop()