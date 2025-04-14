import tkinter as tk
from tkinter import ttk
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer, deepseekSummarizer
from ranking_system.ranking_function import HybridRanker
from models.model import scibert_model, deepseek_model
import torch
import textwrap
from tkinter.font import Font

class SearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Paper Search")
        self.root.geometry("800x600")

        # Allow resizing
        self.root.grid_rowconfigure(10, weight=1)  # Results text row expands vertically
        self.root.grid_columnconfigure(1, weight=1)  # Main content column expands horizontally

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
        self.sem_method_menu['values'] = ("Database-vector (0)", "GoogleNews-word2vec (1)", "GenAI (2)")
        self.sem_method_menu.current(0)  # Default to Database-vector
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
        tk.Button(self.root, text="Search", command=self.perform_search).grid(row=8, column=0, columnspan=3, pady=5, sticky="ew")

        # Clear Display button
        tk.Button(self.root, text="Clear Display", command=self.clear_display).grid(row=9, column=0, columnspan=3, pady=5, sticky="ew")

        # Results display
        tk.Label(self.root, text="Results:").grid(row=10, column=0, padx=5, pady=5, sticky="w")
        # Use a monospaced font for proper alignment
        self.result_font = Font(family="Courier New", size=10)
        self.results_text = tk.Text(self.root, wrap="word", font=self.result_font)
        self.results_text.grid(row=11, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Add scrollbar
        scrollbar = tk.Scrollbar(self.root, command=self.results_text.yview)
        scrollbar.grid(row=11, column=3, sticky="ns")
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Configure grid weights
        self.root.grid_rowconfigure(11, weight=1)  # Updated row for results_text
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # Bind resize event to update wrapping
        self.results_text.bind("<Configure>", self.on_resize)

    def clear_display(self):
        """Clear the contents of the results_text widget."""
        self.results_text.delete(1.0, tk.END)

    def on_resize(self, event):
        """Handle window resize by updating the display."""
        # Redisplay the results when the window is resized
        self.perform_search()

    def get_text_width_in_chars(self):
        """Calculate the width of the results_text widget in characters."""
        # Get the width of the widget in pixels
        widget_width = self.results_text.winfo_width()
        # Get the average character width in pixels for the font
        char_width = self.result_font.measure("M")  # 'M' is typically the widest character in monospaced fonts
        if char_width == 0:  # Avoid division by zero during initialization
            char_width = 10  # Fallback value
        # Calculate the number of characters that fit in the widget's width
        num_chars = max(20, (widget_width - 10) // char_width)  # Subtract some padding, ensure minimum width
        return num_chars

    def wrap_text(self, text, width):
        """Wrap text to a specified width, returning a list of lines."""
        return textwrap.wrap(text, width=width)

    def compare_rankings(self, results_hybrid, results_bm25, results_bert, ranker):
        """Display ranking comparison with dynamically wrapped titles and aligned scores."""
        output = ["=== RANKING COMPARISON ===\n"]
        
        # Dynamically calculate column width based on widget size
        total_width = self.get_text_width_in_chars()
        # Divide the total width by 3 columns, accounting for separators (" | ")
        column_width = max(20, (total_width - 6) // 3)  # Subtract 6 for " | " separators, ensure minimum width
        separator_width = column_width * 3 + 6  # 3 columns + 2 separators (" | ")

        output.append(f"{'BM25 Order':<{column_width}} | {'Vector Order':<{column_width}} | {'Hybrid Order':<{column_width}}\n")
        output.append("-" * separator_width + "\n")

        bm25_sorted = sorted(results_bm25, key=lambda x: x['score'], reverse=True)
        vector_sorted = sorted(results_bert, key=lambda x: x['score'], reverse=True)
        hybrid_sorted = ranker.rank_documents(results_bm25, results_bert)

        for i in range(min(5, len(bm25_sorted))):
            # Wrap each title based on dynamic column width
            bm25_lines = self.wrap_text(bm25_sorted[i]['title'], width=column_width - 3)  # Adjust for padding
            vector_lines = self.wrap_text(vector_sorted[i]['title'], width=column_width - 3)
            hybrid_lines = self.wrap_text(hybrid_sorted[i]['title'], width=column_width - 3)

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
        """Display detailed results with dynamic wrapping based on window size."""
        output = ["\n=== HYBRID RANKING RESULTS ===\n"]
        
        # Dynamically calculate wrapping width based on widget size
        wrap_width = self.get_text_width_in_chars() - 2  # Subtract some padding

        for i, doc in enumerate(results[:top_n], 1):
            # Wrap the title
            title_lines = self.wrap_text(f"Rank {i}: {doc['title']}", width=wrap_width)
            output.extend(line + "\n" for line in title_lines)

            # Wrap the abstract
            abstract_text = doc['abstract']
            if len(abstract_text) > wrap_width * 2:  # Limit to roughly 2 lines
                abstract_text = abstract_text[:wrap_width * 2 - 3] + "..."
            abstract_lines = self.wrap_text(f"Abstract: {abstract_text}", width=wrap_width)
            output.extend(line + "\n" for line in abstract_lines)

            # Wrap the scores
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
                score_text = "Scores: " + " | ".join(score_info)
                score_lines = self.wrap_text(score_text, width=wrap_width)
                output.extend(line + "\n" for line in score_lines)

            # Wrap the summary if enabled
            if summarizer:
                summary = summarizer.summarize(doc['abstract'])
                summary_lines = self.wrap_text(f"Summary: {summary}", width=wrap_width)
                output.extend(line + "\n" for line in summary_lines)

            output.append("=" * min(wrap_width, 80) + "\n")

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
                # Wrap the expanded query display
                wrap_width = self.get_text_width_in_chars() - 2
                query_lines = self.wrap_text(f"Expanded Query: {processed_query}", width=wrap_width)
                self.results_text.insert(tk.END, "\n".join(query_lines) + "\n\n")
                if not processed_query:
                    self.results_text.insert(tk.END, "Query expansion returned no results.\n")
                    return
            else:
                processed_query = [query]

            results_hybrid = self.se.search(processed_query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
            wrap_width = self.get_text_width_in_chars() - 2
            info_lines = self.wrap_text(f"Query: {query}\nBM25: {use_bm25}, BERT: {use_bert}", width=wrap_width)
            self.results_text.insert(tk.END, "\n".join(info_lines) + "\n" + "=" * min(wrap_width, 50) + "\n")

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
            self.sem_method_var.set("GenAI (1)")  # Reset to default (GenAI)

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchUI(root)
    root.mainloop()