import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import threading
import sys
import os
import re
import time
import datetime
import shutil
import webbrowser
import math

# Attempt to import necessary deep learning libraries (optional)
try:
    import torch 
    import sentencepiece as spm
    import numpy as np
except ImportError:
    pass


# ==========================================
# CORE UTILITIES & SETUP
# ==========================================
REQUIRED_PACKAGES = {
    "torch": "torch", 
    "numpy": "numpy", 
    "transformers": "transformers", 
    "omegaconf": "omegaconf", 
    "tensorboard": "tensorboard", 
    "sentencepiece": "sentencepiece"
}

def check_dependencies():
    missing = []
    for module, pkg_name in REQUIRED_PACKAGES.items():
        try:
            # Simplified check: just attempt import
            __import__(module)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        root = tk.Tk()
        root.withdraw()
        
        missing_list = ", ".join(missing)
        pip_command = f"pip install {' '.join(missing)}"
        uv_command = f"uv pip install {' '.join(missing)}"
        
        msg = (
            "🚨 Libraries Missing! 🚨\n\n"
            f"The following required packages were not found: \n{missing_list}\n\n"
            "The GUI will now close. Please run one of the following commands in "
            "the directory where this script is located to install the dependencies:\n\n"
            f"Using pip: \n  {pip_command}\n\n"
            f"Using uv: \n  {uv_command}\n"
        )
        messagebox.showerror("Dependency Error", msg)
        sys.exit(1)

def ensure_directories():
    dirs = [
        "1- datasets", "2- trained_models", "3- pruned_models", 
        "4- pretrained", "0- checkpoints"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ==========================================
# BASE GUI FRAMEWORK
# ==========================================
class BaseTab(ttk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.widgets = {}
        self.current_row = 0
        
        lbl = ttk.Label(self, text=title, font=("Segoe UI", 12, "bold"))
        lbl.grid(row=self.current_row, column=0, columnspan=3, pady=(15, 15), padx=10, sticky="w")
        self.current_row += 1

    def add_file(self, label, key, default="", dir=".", types=None, is_save=False):
        ttk.Label(self, text=label).grid(row=self.current_row, column=0, sticky="w", padx=10, pady=2)
        var = tk.StringVar(value=default)
        self.widgets[key] = var
        ttk.Entry(self, textvariable=var).grid(row=self.current_row, column=1, sticky="ew", padx=5)
        cmd = lambda: self._browse_file(var, dir, types, is_save)
        ttk.Button(self, text="Browse", command=cmd).grid(row=self.current_row, column=2, padx=10)
        self.current_row += 1
        return var

    def add_dir(self, label, key, default="", dir="."):
        ttk.Label(self, text=label).grid(row=self.current_row, column=0, sticky="w", padx=10, pady=2)
        var = tk.StringVar(value=default)
        self.widgets[key] = var
        ttk.Entry(self, textvariable=var).grid(row=self.current_row, column=1, sticky="ew", padx=5)
        cmd = lambda: self._browse_dir(var, dir)
        ttk.Button(self, text="Browse", command=cmd).grid(row=self.current_row, column=2, padx=10)
        self.current_row += 1
        return var

    def add_entry(self, label, key, default="", width=None):
        ttk.Label(self, text=label).grid(row=self.current_row, column=0, sticky="w", padx=10, pady=2)
        var = tk.StringVar(value=str(default))
        self.widgets[key] = var
        e = ttk.Entry(self, textvariable=var, width=width)
        e.grid(row=self.current_row, column=1, sticky="w" if width else "ew", padx=5)
        self.current_row += 1
        return var

    def add_check(self, label, key, default=False):
        var = tk.BooleanVar(value=default)
        self.widgets[key] = var
        ttk.Checkbutton(self, text=label, variable=var).grid(row=self.current_row, column=1, sticky="w", padx=5, pady=2)
        self.current_row += 1
        return var

    def add_separator(self):
        ttk.Separator(self, orient='horizontal').grid(row=self.current_row, column=0, columnspan=3, sticky="ew", pady=10)
        self.current_row += 1

    def add_log_box(self, height=10):
        ttk.Label(self, text="Log Output:").grid(row=self.current_row, column=0, sticky="w", padx=10, pady=(10,0))
        self.current_row += 1
        self.log_text = scrolledtext.ScrolledText(self, height=height, state='disabled', font=("Consolas", 9))
        self.log_text.grid(row=self.current_row, column=0, columnspan=3, sticky="nsew", padx=10, pady=5)
        # Custom tags for log output colors
        self.log_text.tag_config("err", foreground="red")
        self.log_text.tag_config("ok", foreground="green")
        self.log_text.tag_config("cmd", foreground="blue")
        self.log_text.tag_config("warn", foreground="#FF8C00")
        self.rowconfigure(self.current_row, weight=1)
        self.columnconfigure(1, weight=1)
        self.current_row += 1

    def _browse_file(self, var, start, types, is_save):
        if not os.path.exists(start): os.makedirs(start, exist_ok=True)
        ft = types or [("All Files", "*.*")]
        
        if is_save:
            # Get filename from user
            f = filedialog.asksaveasfilename(initialdir=start, filetypes=ft)
            
            # --- FORCE EXTENSION LOGIC ---
            if f and types:
                try:
                    # Extract extension from the first tuple, e.g., ("SentencePiece", "*.model") -> .model
                    # We remove the wildcard '*'
                    ext_pattern = types[0][1] 
                    expected_ext = ext_pattern.replace("*", "") # becomes .model or .jsonl
                    
                    # If the file path doesn't end with the expected extension, append it
                    if not f.lower().endswith(expected_ext.lower()):
                        f += expected_ext
                except IndexError:
                    pass
            # -----------------------------
        else:
            f = filedialog.askopenfilename(initialdir=start, filetypes=ft)
            
        if f: var.set(f)

    def _browse_dir(self, var, start):
        if not os.path.exists(start): os.makedirs(start, exist_ok=True)
        d = filedialog.askdirectory(initialdir=start)
        if d: var.set(d)

    def log(self, msg, tag=None):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"{msg}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def run_simple_command(self, cmd, success_msg="Done"):
        def _target():
            self.log(f"CMD: {' '.join(cmd)}", "cmd")
            try:
                # Use bufsize=1 and text=True for real-time output reading
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in p.stdout:
                    # Use self.after to update GUI from the thread
                    self.after(0, lambda l=line.strip(): self.log(l))
                p.wait()
                if p.returncode == 0:
                    self.after(0, lambda: self.log(f"✓ {success_msg}", "ok"))
                    self.after(0, lambda: messagebox.showinfo("Success", success_msg))
                else:
                    self.after(0, lambda: self.log(f"✗ Failed: Code {p.returncode}", "err"))
                    self.after(0, lambda: messagebox.showerror("Error", f"Process failed with code {p.returncode}"))
            except Exception as e:
                self.after(0, lambda: self.log(f"Exception: {e}", "err"))
        
        threading.Thread(target=_target, daemon=True).start()


# ==========================================
# 1. TAB: TOKENIZER EXTENSION (BPE) - SMART ANALYSIS
# ==========================================
class TabExtendBPE(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "1. Extend Tokenizer (BPE) ✨")
        
        self.add_file("Original / base / pretrained BPE:", "base_model", dir=".", types=[("SentencePiece", "*.model")])
        self.add_file("JSONL Manifest:", "manifests", dir="1- datasets", types=[("JSONL", "*.jsonl")])
        self.add_file("Output Model:", "output_model", dir="1- datasets", is_save=True, types=[("SentencePiece", "*.model")])
        
        # Button for smart analysis
        btn_calc = ttk.Button(self, text="🔬 Analyze & Get EXACT Token Count", command=self.analyze_exact_needs, style="Accent.TButton")
        btn_calc.grid(row=self.current_row, column=1, sticky="ew", padx=5, pady=5)
        self.current_row += 1

        self.add_separator()
        
        self.add_entry("Target Size:", "target_size", "", width=10)
        self.add_entry("Candidate Size:", "candidate_size", "", width=10)
        self.add_entry("Char Coverage:", "char_cov", "1.0", width=10)
        
        self.widgets["model_type"] = tk.StringVar(value="bpe") 
        self.add_check("Byte Fallback (Experimental)", "byte_fallback", False) 
        
        self.add_separator()
        ttk.Button(self, text="▶ Run Extension", command=self.run, style="Accent.TButton").grid(row=self.current_row, column=0, columnspan=3, pady=10)
        self.current_row += 1
        self.add_log_box()

    def analyze_exact_needs(self):
        """
        Analyzes the dataset to find EXACTLY how many unique new tokens exist.
        Avoids using Dummy Tokens by calculating the real intersection.
        """
        # --- LOCAL IMPORTS TO AVOID ERRORS IF MISSING ABOVE ---
        import tempfile
        import json
        import re
        # ------------------------------------------------------------

        manifest_path = self.widgets["manifests"].get()
        base_model_path = self.widgets["base_model"].get()
        
        if not manifest_path or not os.path.exists(manifest_path):
            self.log("Error: Manifest file not found.", "err")
            return
        if not base_model_path or not os.path.exists(base_model_path):
            self.log("Error: Base model file not found.", "err")
            return

        # Check if sentencepiece is available
        try:
            import sentencepiece as spm
        except ImportError:
            self.log("CRITICAL: 'sentencepiece' library not found in GUI environment.", "err")
            self.log("Run: pip install sentencepiece", "cmd")
            return
        
        self.log("----------------------------------------", "cmd")
        self.log("🔍 Starting Exact Analysis...", "cmd")

        try:
            # 1. Load Base Model
            sp_base = spm.SentencePieceProcessor(model_file=base_model_path)
            base_size = sp_base.get_piece_size()
            self.log(f"✓ Base Vocab Size: {base_size}", "ok")
            
            # Get base tokens set for comparison
            base_tokens = set()
            for i in range(base_size):
                base_tokens.add(sp_base.id_to_piece(i))

            # 2. Prepare temporary corpus
            # Using delete=False so Windows doesn't lock the file when reading it later
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as tmp_file:
                temp_corpus_path = tmp_file.name
                line_count = 0
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "text" in data:
                                        tmp_file.write(data["text"] + "\n")
                                        line_count += 1
                                except: pass
                except Exception as e:
                    self.log(f"Error reading manifest: {e}", "err")
                    return
            
            self.log(f"✓ Temporary corpus created: {line_count} lines", "cmd")

            # 3. Train "Draft" Model
            # Attempt to train a model with high vocabulary.
            vocab_attempt = 20000 
            
            with tempfile.TemporaryDirectory() as tmpdir:
                prefix = os.path.join(tmpdir, "draft")
                
                try:
                    # Initial attempt
                    spm.SentencePieceTrainer.Train(
                        input=temp_corpus_path,
                        model_prefix=prefix,
                        vocab_size=vocab_attempt,
                        model_type="bpe",
                        character_coverage=1.0,
                        train_extremely_large_corpus=False
                    )
                except RuntimeError as e:
                    # Catch the "Please set it to a value <= X" message if the dataset is small
                    err_str = str(e)
                    match = re.search(r"Please set it to a value <= (\d+)", err_str)
                    if match:
                        max_real_vocab = int(match.group(1))
                        self.log(f"⚠ Dataset too small for {vocab_attempt}. Max possible tokens: {max_real_vocab}", "warn")
                        self.log("  Re-training with max possible size...", "cmd")
                        
                        # Re-train with the exact real limit
                        spm.SentencePieceTrainer.Train(
                            input=temp_corpus_path,
                            model_prefix=prefix,
                            vocab_size=max_real_vocab,
                            model_type="bpe",
                            character_coverage=1.0
                        )
                    else:
                        raise e

                # 4. Compare Tokens (Draft vs Base)
                sp_draft = spm.SentencePieceProcessor(model_file=prefix + ".model")
                draft_size = sp_draft.get_piece_size()
                
                new_unique_tokens = 0
                for i in range(draft_size):
                    piece = sp_draft.id_to_piece(i)
                    # If the token is NOT in the base model, it is new and useful
                    if piece not in base_tokens:
                        new_unique_tokens += 1

            # Cleanup temporary file
            if os.path.exists(temp_corpus_path):
                try:
                    os.remove(temp_corpus_path)
                except: pass

            # 5. RESULTS
            final_target = base_size + new_unique_tokens
            
            # Safety margin for candidate (necessary for the mining process)
            # Candidate must be larger than target for the script to have something to choose from
            final_candidate = final_target + max(500, new_unique_tokens * 2)

            self.log(f"✓ Analysis Complete:", "ok")
            self.log(f"  - Tokens found in dataset: {draft_size}")
            self.log(f"  - Tokens ALREADY in base: {draft_size - new_unique_tokens}")
            self.log(f"  - ACTUALLY NEW tokens: {new_unique_tokens}", "ok")

            if new_unique_tokens == 0:
                messagebox.showwarning("Analysis", "No new tokens found! Your base model already covers this dataset completely.")
                self.widgets["target_size"].set(str(base_size))
                self.widgets["candidate_size"].set(str(base_size + 100)) # Dummy safe value
            else:
                self.widgets["target_size"].set(str(final_target))
                self.widgets["candidate_size"].set(str(final_candidate))
                self.log(f"-> Recommended Target Size: {final_target}", "cmd")

        except Exception as e:
            self.log(f"Analysis Failed: {e}", "err")
            # Try to cleanup file if failed
            if 'temp_corpus_path' in locals() and os.path.exists(temp_corpus_path):
                 try: os.remove(temp_corpus_path)
                 except: pass

    def run(self):
        # Standard execution logic
        if not self.widgets["target_size"].get():
             self.log("Error: Calculate target size first!", "err")
             return

        cmd = [
            "uv", "run", "python", "tools/tokenizer/extend_bpe.py",
            "--base-model", self.widgets["base_model"].get(),
            "--manifests", self.widgets["manifests"].get(),
            "--output-model", self.widgets["output_model"].get(),
            "--target-size", self.widgets["target_size"].get(),
            "--candidate-size", self.widgets["candidate_size"].get(),
            "--character-coverage", self.widgets["char_cov"].get(),
            "--model-type", self.widgets["model_type"].get()
        ]
        if self.widgets["byte_fallback"].get():
            cmd.append("--byte-fallback")
            
        self.run_simple_command(cmd, "Tokenizer Extended Successfully")

# ==========================================
# 2. TAB: DATASET PREPROCESSING
# ==========================================
class TabPreprocess(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "2. Preprocess Dataset 📊")
        
        self.add_file("JSONL Manifest:", "manifest", dir="1- datasets", types=[("JSONL", "*.jsonl")])
        self.add_dir("Output Directory:", "output_dir", dir="1- datasets")
        self.add_file("BPE Extended Tokenizer:", "tokenizer", dir="1- datasets", types=[("Model", "*.model")])
        self.add_file("Config YAML:", "config", dir="0- checkpoints", types=[("YAML", "*.yaml")])
        self.add_file("GPT Checkpoint / base model / pretrained  model:", "ckpt", dir=".", types=[("PTH", "*.pth")])
        
        self.add_separator()
        
        # --- Vocab Detection Section ---
        vocab_frame = ttk.LabelFrame(self, text="Vocab Calculator (Avoids mismatch errors) 🔢")
        vocab_frame.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        self.current_row += 1
        
        ttk.Label(vocab_frame, text="GPT Vocab Size:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.widgets["vocab_size"] = tk.StringVar(value="N/A") 
        ttk.Label(vocab_frame, textvariable=self.widgets["vocab_size"], font=("Arial", 10, "bold")).grid(row=0, column=1, sticky="w")
        
        btn_detect = ttk.Button(vocab_frame, text="🔮 Check & Set in Config", command=self.detect_and_set_vocab, style="Accent.TButton")
        btn_detect.grid(row=0, column=2, padx=10, pady=5)
        
        self._detected_vocab_size = None
        
        self.add_separator()
        
        # --- Basic Settings (Visible) ---
        settings_frame = ttk.Frame(self)
        settings_frame.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10)
        self.current_row += 1

        # Language
        ttk.Label(settings_frame, text="Language:").grid(row=0, column=0, sticky="e", padx=5)
        self.widgets["lang"] = tk.StringVar(value="en")
        ttk.Entry(settings_frame, textvariable=self.widgets["lang"], width=10).grid(row=0, column=1, sticky="w", padx=5)

        # Validation Split (%) 
        ttk.Label(settings_frame, text="Validation Split (%):").grid(row=0, column=2, sticky="e", padx=5)
        self.widgets["val_ratio"] = tk.StringVar(value="10") 
        ttk.Entry(settings_frame, textvariable=self.widgets["val_ratio"], width=10).grid(row=0, column=3, sticky="w", padx=5)
        
        self.add_separator()

        # --- Advanced Arguments ---
        lf = ttk.LabelFrame(self, text="Advanced Arguments ⚙️")
        lf.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10)
        self.current_row += 1
        
        def _add_lf(lbl, key, def_val, row, col):
            ttk.Label(lf, text=lbl).grid(row=row, column=col*2, sticky="e", padx=5)
            v = tk.StringVar(value=def_val)
            self.widgets[key] = v
            ttk.Entry(lf, textvariable=v, width=8).grid(row=row, column=col*2+1, sticky="w", padx=5)

        _add_lf("Workers:", "workers", "4", 0, 0)
        _add_lf("Num Processes:", "num_proc", "1", 0, 1)
        _add_lf("Batch Size:", "batch", "4", 0, 2)
        
        self.widgets["skip"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Skip Existing", variable=self.widgets["skip"]).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        self.widgets["force"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Force Overwrite", variable=self.widgets["force"]).grid(row=1, column=2, columnspan=2, sticky="w", padx=5, pady=5)


        self.add_separator()
        ttk.Button(self, text="▶ Preprocess Dataset", command=self.run, style="Accent.TButton").grid(row=self.current_row, column=0, columnspan=3, pady=10)
        self.current_row += 1
        self.add_log_box()

    def detect_and_set_vocab(self):
        ckpt_path = self.widgets["ckpt"].get()
        cfg_path = self.widgets["config"].get()

        if not os.path.exists(ckpt_path):
            self.log("Error: Select a GPT Checkpoint first (.pth)", "err")
            return
        if not os.path.exists(cfg_path):
            self.log("Error: Select the Config YAML file first.", "err")
            return
        
        if 'torch' not in sys.modules:
            self.log("Error: 'torch' module not loaded. Cannot read checkpoint.", "err")
            return
            
        try:
            self.log(f"Analyzing checkpoint: {ckpt_path}...", "cmd")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt.get('state_dict', ckpt)
            
            if 'text_embedding.weight' in state_dict:
                vocab_size = state_dict['text_embedding.weight'].shape[0]
                compensated_size = vocab_size - 1
                
                self._detected_vocab_size = compensated_size
                self.log(f"✓ Checkpoint size (text_embedding.weight): {vocab_size} tokens", "ok")
                self.log(f"-> Calculated YAML value (V_ckpt - 1): {compensated_size} tokens", "ok")
                self.widgets["vocab_size"].set(str(compensated_size))
                
                if self.update_yaml_file(cfg_path, compensated_size):
                    self.log("✓ Config YAML updated successfully.", "ok")
                else:
                    self.log("Error: Failed to update config.yaml.", "err")
                return compensated_size
            else:
                self.log("Warning: 'text_embedding.weight' not found. Assuming fallback 12000.", "warn")
                self._detected_vocab_size = 12000
                self.widgets["vocab_size"].set("12000")
                return 12000
        except Exception as e:
            self.log(f"Error reading checkpoint: {e}", "err")
            self._detected_vocab_size = None
            self.widgets["vocab_size"].set("ERROR")
            return None
            
    def update_yaml_file(self, cfg_path, new_val):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            updated = False
            for line in lines:
                if "number_text_tokens:" in line:
                    new_line = re.sub(r'(number_text_tokens:\s*)(\d+)', fr'\g<1>{new_val}', line)
                    new_lines.append(new_line)
                    updated = True
                else:
                    new_lines.append(line)
            
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return updated
        except Exception as e:
            self.log(f"Error updating YAML: {e}", "err")
            return False

    def run(self):
        vocab_val = self.widgets["vocab_size"].get()
        if vocab_val == "N/A" or vocab_val == "ERROR":
            if not messagebox.askyesno("Warning", "Vocab size not calculated or failed. Continue with default?"):
                return
        
        # --- MATHEMATICAL CORRECTION ---
        val_ratio_arg = "0.1" # Safe default
        try:
            val_percent = float(self.widgets["val_ratio"].get())
            if val_percent < 0 or val_percent > 100:
                self.log("Error: Validation Split must be between 0 and 100", "err")
                return
            
            # Convert 10 -> 0.1, 5 -> 0.05, etc.
            val_ratio_float = val_percent / 100.0
            val_ratio_arg = str(val_ratio_float)
            self.log(f"ℹ Converting {val_percent}% to ratio {val_ratio_arg} for the script.", "cmd")
            
        except ValueError:
            self.log("Error: Validation Split must be a number", "err")
            return
        # -----------------------------

        cmd = [
            "uv", "run", "tools/preprocess_multiproc.py",
            "--manifest", self.widgets["manifest"].get(),
            "--output-dir", self.widgets["output_dir"].get(),
            "--tokenizer", self.widgets["tokenizer"].get(),
            "--config", self.widgets["config"].get(),
            "--gpt-checkpoint", self.widgets["ckpt"].get(),
            "--language", self.widgets["lang"].get(),
            "--val-ratio", val_ratio_arg,  # Using converted value (0.1) instead of original (10)
            "--workers", self.widgets["workers"].get(),
            "--num-processes", self.widgets["num_proc"].get(),
            "--batch-size", self.widgets["batch"].get(),
        ]
        if self.widgets["skip"].get(): cmd.append("--skip-existing")
        if self.widgets["force"].get(): cmd.append("--force")
        
        self.run_simple_command(cmd, "Preprocessing Complete")

# ==========================================
# 3. TAB: GENERATE PAIRS (DATA PREPARATION)
# ==========================================
class TabPairs(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "3. Generate Pairs 🧬")
        
        self.add_dir("Preprocessed Dataset Directory:", "dataset", dir="1- datasets")
        self.add_entry("Pairs Per Target:", "ppt", "1")
        self.add_entry("Seed:", "seed", "1234")
        self.add_check("Force Overwrite", "force", True)
        
        self.add_separator()
        ttk.Button(self, text="▶ Generate Pairs", command=self.run, style="Accent.TButton").grid(row=self.current_row, column=0, columnspan=3, pady=10)
        self.current_row += 1
        self.add_log_box()

    def run(self):
        cmd = [
            "uv", "run", "tools/generate_gpt_pairs.py",
            "--dataset", self.widgets["dataset"].get(),
            "--pairs-per-target", self.widgets["ppt"].get(),
            "--seed", self.widgets["seed"].get()
        ]
        if self.widgets["force"].get(): cmd.append("--force")
        self.run_simple_command(cmd, "Pairs Generated")

# ==========================================
# 4. TAB: MODEL TRAINING
# ==========================================
class TabTrain(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "4. Train Model 🚀")
        
        # File/Directory Inputs
        self.add_file("GPT Pair Train Manifest:", "train_man", dir="1- datasets", types=[("JSONL", "*.jsonl")])
        self.add_file("GPT Pair Validation Manifest:", "val_man", dir="1- datasets", types=[("JSONL", "*.jsonl")])
        self.add_file(" BPE Expended Tokenizer:", "tok", dir="1- datasets", types=[("Model", "*.model")])
        self.add_file("Config:", "cfg", dir="0- checkpoints", types=[("YAML", "*.yaml")])
        self.add_file("GPT Checkpoint / base model / pretrained  model:", "base", dir="4- pretrained", types=[("PTH", "*.pth")])
        self.add_dir("Output Directory:", "out", dir="2- trained_models")
        
        self.add_separator()

        # Training Parameters Frame
        pf = ttk.LabelFrame(self, text="Training Parameters (Arguments exposed) ⚙️")
        pf.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        self.current_row += 1
        
        def _p(lbl, key, val, r, c, w=10):
            ttk.Label(pf, text=lbl).grid(row=r, column=c*3, sticky="e", padx=5, pady=2)
            v = tk.StringVar(value=str(val))
            self.widgets[key] = v
            ttk.Entry(pf, textvariable=v, width=w).grid(row=r, column=c*3+1, sticky="w", padx=5)


        _p("Batch Size:", "batch", "8", 0, 0)
        _p("Grad Accumulation:", "grad_acc", "1", 0, 1)
        _p("Epochs:", "epochs", "10", 0, 2)
        
        _p("Learning Rate:", "lr", "1e-4", 1, 0)
        _p("Warmup Steps:", "warmup", "100", 1, 1)
        _p("Weight Decay:", "wd", "0.01", 1, 2)
        
        _p("Log Interval:", "log_int", "10", 2, 0)
        _p("Validation Interval:", "val_int", "100", 2, 1)
        _p("Save Interval:", "save_int", "500", 2, 2)
        
        _p("Text Loss Weight:", "text_w", "0.2", 3, 0)
        _p("Mel Loss Weight:", "mel_w", "0.8", 3, 1)
        _p("Gradient Clip:", "clip", "1.0", 3, 2)
        
        _p("Max Steps (0 for no limit):", "max_steps", "0", 4, 0)
        _p("Seed:", "seed", "1234", 4, 1)
        _p("Resume (auto or file):", "resume", "auto", 4, 2)
        
        # Toggles and Duration Control
        tf = ttk.Frame(pf)
        tf.grid(row=5, column=0, columnspan=9, sticky="ew", pady=5)
        self.widgets["amp"] = tk.BooleanVar(value=True)
        self.widgets["dur"] = tk.BooleanVar(value=True)
        self.widgets["dur_drop"] = tk.StringVar(value="0.3")
        
        ttk.Checkbutton(tf, text="Enable AMP", variable=self.widgets["amp"]).pack(side="left", padx=10)
        ttk.Checkbutton(tf, text="Use Duration Control", variable=self.widgets["dur"]).pack(side="left", padx=10)
        ttk.Label(tf, text="Duration Dropout:").pack(side="left", padx=5)
        ttk.Entry(tf, textvariable=self.widgets["dur_drop"], width=5).pack(side="left")


        # Calculator
        cf = ttk.LabelFrame(self, text="Optimal Parameter Calculator 🧮")
        cf.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        self.current_row += 1
        
        # Target Steps Input Frame
        target_frame = ttk.Frame(cf)
        target_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(target_frame, text="Target Steps (Goal):").pack(side="left", padx=5)
        self.widgets["target_steps"] = tk.StringVar(value="3000")
        ttk.Entry(target_frame, textvariable=self.widgets["target_steps"], width=10).pack(side="left", padx=5)
        
        ttk.Button(cf, text="📊 Calculate Optimal Parameters", command=self.calculate_optimal, style="Accent.TButton").pack(fill="x", padx=10, pady=(0, 5))
        self.lbl_calc_info = ttk.Label(cf, text="Info: Waiting for calculation...", foreground="gray")
        self.lbl_calc_info.pack(pady=5)

        # Controls & Progress
        self.add_separator()
        
        # --- Control Buttons Frame ---
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=self.current_row, column=0, columnspan=3, pady=10)
        
        self.btn_run = ttk.Button(btn_frame, text="▶ START TRAINING", command=self.run_training, style="Accent.TButton")
        self.btn_run.pack(side="left", padx=10)
        
        self.btn_stop = ttk.Button(btn_frame, text="⏹ STOP TRAINING", command=self.stop_training, state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        
        self.current_row += 1
        
        # Progress UI
        pf_ui = ttk.Frame(self)
        pf_ui.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10)
        self.current_row += 1
        
        self.progress = ttk.Progressbar(pf_ui, mode='determinate')
        self.progress.pack(fill='x', pady=5)
        
        self.lbl_step = ttk.Label(pf_ui, text="Step: 0 / 0", font=("Arial", 10, "bold"))
        self.lbl_step.pack(side="left")
        
        self.lbl_eta = ttk.Label(pf_ui, text="ETA: --:--:--", font=("Consolas", 10))
        self.lbl_eta.pack(side="right")

        self.add_log_box(height=12)
        
        self.process = None
        self.is_running = False

    def count_lines(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f if _.strip())
        except: return 0

    def calculate_optimal(self):
        train_path = self.widgets["train_man"].get()
        if not os.path.exists(train_path):
            messagebox.showerror("Error", "Train manifest not found.")
            return

        n_samples = self.count_lines(train_path)
        if n_samples == 0:
            messagebox.showerror("Error", "Train manifest is empty.")
            return

        # 1. GET TARGET STEPS FROM USER
        try:
            target_steps_val = int(self.widgets["target_steps"].get())
            if target_steps_val < 100:
                target_steps_val = 100 
        except ValueError:
            target_steps_val = 3000 
            self.widgets["target_steps"].set("3000")

        MIN_LR = 1e-5
        MAX_LR = 1e-4

        # 2. Determine recommended Batch Size based on dataset size
        if n_samples < 1000:
            recommended_bs = 2
            target_eff_bs = 8
        elif n_samples < 5000:
            recommended_bs = 4
            target_eff_bs = 16
        else:
            recommended_bs = 8
            target_eff_bs = 32
        
        # 3. Calculate how many physical batches are in one epoch
        batches_per_epoch = math.ceil(n_samples / recommended_bs)

        # 4. Adjust Grad Accumulation
        ideal_ga = max(1, math.ceil(target_eff_bs / recommended_bs))
        
        if batches_per_epoch < ideal_ga:
            recommended_ga = max(1, batches_per_epoch)
        else:
            recommended_ga = ideal_ga
        
        # 5. CALCULATE ACTUAL STEPS
        real_steps_per_epoch = batches_per_epoch // recommended_ga
        
        if real_steps_per_epoch == 0:
            real_steps_per_epoch = 1
            recommended_ga = max(1, batches_per_epoch) 
        
        # 6. Calculate necessary Epochs
        needed_epochs = math.ceil(target_steps_val / real_steps_per_epoch)
        final_total_steps = real_steps_per_epoch * needed_epochs
        
        # 7. LR and Warmup Recommendations
        recommended_lr = MAX_LR if n_samples < 2000 else MIN_LR
        recommended_warmup = max(50, int(final_total_steps * 0.05)) # 5% of warmup

        # 8. Intervals
        log_int = max(1, min(50, real_steps_per_epoch // 2))
        val_int = max(real_steps_per_epoch, 500)
        save_int = min(max(real_steps_per_epoch * 2, 1000), final_total_steps)
        
        # --- APPLY TO GUI ---
        self.widgets["batch"].set(str(recommended_bs))
        self.widgets["grad_acc"].set(str(recommended_ga))
        self.widgets["epochs"].set(str(needed_epochs))
        self.widgets["lr"].set(f"{recommended_lr:.1e}")
        self.widgets["warmup"].set(str(recommended_warmup))
        
        self.widgets["log_int"].set(str(log_int))
        self.widgets["val_int"].set(str(val_int))
        self.widgets["save_int"].set(str(save_int))
        
        info = (f"Samples: {n_samples} | Batch: {recommended_bs} | GA: {recommended_ga}\n"
                f"Real Steps/Epoch: {real_steps_per_epoch} (Discarded batches: {batches_per_epoch % recommended_ga})\n"
                f"Target: {target_steps_val} -> Set Epochs: {needed_epochs} -> Final Steps: {final_total_steps}")
        
        self.lbl_calc_info.config(text=info, foreground="blue")
        self.log(info, "cmd")

    def stop_training(self):
        """Stops the training process if active."""
        if self.process and self.process.poll() is None:
            if messagebox.askyesno("Stop Training", "Are you sure you want to stop the training?"):
                self.log("🛑 STOPPING TRAINING PROCESS...", "warn")
                try:
                    self.process.terminate()  # Sends SIGTERM
                except Exception as e:
                    self.log(f"Error terminating process: {e}", "err")
        else:
            self.log("No active training process found.", "warn")

    def run_training(self):
        if self.is_running: return
        
        w = self.widgets
        
        try:
            ns = self.count_lines(w["train_man"].get())
            bs = int(w["batch"].get())
            ga = int(w["grad_acc"].get())
            eps = int(w["epochs"].get())
            if bs == 0 or ga == 0 or eps == 0:
                 raise ValueError("Batch Size, Grad Acc, and Epochs must be greater than zero.")
            
            batches_total = math.ceil(ns / bs)
            steps_per_epoch = batches_total // ga 
            if steps_per_epoch == 0: steps_per_epoch = 1 
            
            total_steps = steps_per_epoch * eps
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric parameter: {e}")
            return
        except Exception as e:
            messagebox.showerror("Setup Error", f"Check manifest paths: {e}")
            return
            
        cmd = [
            "uv", "run", "python", "trainers/train_gpt_v2.py", 
            "--train-manifest", w["train_man"].get(),
            "--val-manifest", w["val_man"].get(),
            "--tokenizer", w["tok"].get(),
            "--config", w["cfg"].get(),
            "--base-checkpoint", w["base"].get(),
            "--output-dir", w["out"].get(),
            "--batch-size", w["batch"].get(),
            "--grad-accumulation", w["grad_acc"].get(),
            "--epochs", w["epochs"].get(),
            "--learning-rate", w["lr"].get(),
            "--warmup-steps", w["warmup"].get(),
            "--weight-decay", w["wd"].get(),
            "--log-interval", w["log_int"].get(),
            "--val-interval", w["val_int"].get(),
            "--save-interval", w["save_int"].get(),
            "--text-loss-weight", w["text_w"].get(),
            "--mel-loss-weight", w["mel_w"].get(),
            "--grad-clip", w["clip"].get(),
            "--max-steps", w["max_steps"].get(),
            "--seed", w["seed"].get()
        ]
        
        if w["amp"].get(): cmd.append("--amp")
        if w["dur"].get(): 
            cmd.append("--use-duration-control")
            cmd.extend(["--duration-dropout", w["dur_drop"].get()])
            
        if w["resume"].get(): cmd.extend(["--resume", w["resume"].get()])

        self.progress["maximum"] = total_steps
        self.progress["value"] = 0
        self.is_running = True
        
        # Update button states
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        
        def _train_thread():
            self.log(f"CMD: {' '.join(cmd)}", "cmd")
            start_time = time.time()
            step_pattern = re.compile(r"step=(\d+)")
            
            try:
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                
                si = None
                if os.name == 'nt':
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    
                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                                text=True, bufsize=1, env=env, startupinfo=si)

                for line in self.process.stdout:
                    clean = line.strip()
                    self.after(0, lambda l=clean: self.log(l))
                    
                    match = step_pattern.search(clean)
                    if match:
                        current_step = int(match.group(1))
                        elapsed = time.time() - start_time
                        
                        if current_step > 0 and current_step <= total_steps:
                            sps = elapsed / current_step 
                            remaining = total_steps - current_step
                            eta_sec = remaining * sps
                            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                        else:
                            eta_str = "--:--:--"

                        self.after(0, lambda s=current_step, e=eta_str: self._update_ui(s, total_steps, e))

                self.process.wait()
                ret = self.process.returncode
                
                # Exit code handling
                if ret == 0:
                    self.after(0, lambda: self.log("TRAINING SUCCESSFUL", "ok"))
                    self.after(0, lambda: messagebox.showinfo("Done", "Training Finished!"))
                elif ret == 1 or ret == -15: # -15 is usually SIGTERM
                     self.after(0, lambda: self.log("Training Stopped or Failed.", "warn"))
                else:
                    self.after(0, lambda: messagebox.showerror("Error", f"Training failed code {ret}"))
                    
            except Exception as e:
                self.after(0, lambda: self.log(f"CRITICAL: {e}", "err"))
            finally:
                self.is_running = False
                # Restore button states
                self.after(0, lambda: self.btn_run.config(state="normal"))
                self.after(0, lambda: self.btn_stop.config(state="disabled"))
                self.process = None

        threading.Thread(target=_train_thread, daemon=True).start()

    def _update_ui(self, step, total, eta):
        display_step = min(step, total)
        self.progress["value"] = display_step
        self.lbl_step.config(text=f"Step: {step} / {total}")
        self.lbl_eta.config(text=f"ETA: {eta}")

# ==========================================
# 5. TAB: PRUNE & EXPORT
# ==========================================
class TabPrune(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "5. Prune & Export ✂️")
        self.add_file("Input new trained model (.pth):", "in", dir="2- trained_models", types=[("PTH", "*.pth")])
        self.add_file("Output new trained pruned Model (.pth):", "out", dir="3- pruned_models", is_save=True, types=[("PTH", "*.pth")])
        
        self.add_separator()
        ttk.Button(self, text="✂️ Prune Checkpoint", command=self.run, style="Accent.TButton").grid(row=self.current_row, column=0, columnspan=3, pady=10)
        self.current_row += 1
        self.add_log_box()

    def run(self):
        cmd = [
            "uv", "run", "python", "tools/prune_gpt_checkpoint.py",
            "--input", self.widgets["in"].get(),
            "--output", self.widgets["out"].get(),
        ]
        self.run_simple_command(cmd, "Model Pruned")

# ==========================================
# 6. TAB: FILE MANAGEMENT
# ==========================================
class TabFileManagement(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "6. File Management 📂")
        
        # --- SECTION: COPY FILES ---
        copy_frame = ttk.LabelFrame(self, text="Copy New trained model to 0- checkpoints for inference")
        copy_frame.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        self.current_row += 1
        
        # BPE Source
        ttk.Label(copy_frame, text="Extended BPE").grid(row=0, column=0, sticky="w", padx=5)
        self.widgets["util_bpe"] = tk.StringVar()
        ttk.Entry(copy_frame, textvariable=self.widgets["util_bpe"]).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(copy_frame, text="Browse", command=lambda: self._browse_file(self.widgets["util_bpe"], "1- datasets", [("Model", "*.model")], False)).grid(row=0, column=2, padx=5)
        
        # Model Source
        ttk.Label(copy_frame, text="New trained pruned model:").grid(row=1, column=0, sticky="w", padx=5)
        self.widgets["util_model"] = tk.StringVar()
        ttk.Entry(copy_frame, textvariable=self.widgets["util_model"]).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(copy_frame, text="Browse", command=lambda: self._browse_file(self.widgets["util_model"], "3- pruned_models", [("PTH", "*.pth")], False)).grid(row=1, column=2, padx=5)
        
        ttk.Button(copy_frame, text="📂 Copy Files to '0- checkpoints'", command=self.run_copy, style="Accent.TButton").grid(row=2, column=0, columnspan=3, pady=10)
        
        self.add_log_box()
    
    def run_copy(self):
        bpe_path = self.widgets["util_bpe"].get()
        model_path = self.widgets["util_model"].get()
        dest_dir = "0- checkpoints"

        files_to_copy = []
        if bpe_path and os.path.exists(bpe_path): files_to_copy.append(bpe_path)
        if model_path and os.path.exists(model_path): files_to_copy.append(model_path)

        if not files_to_copy:
            self.log("Alert: No valid files have been selected or found.", "err")
            return
            
        def copy_thread():
            self.after(0, lambda: self.log(f"Starting copy process to '{dest_dir}'...", "cmd"))
            try:
                os.makedirs(dest_dir, exist_ok=True)
                for src in files_to_copy:
                    shutil.copy(src, dest_dir)
                    self.after(0, lambda n=os.path.basename(src): self.log(f"Copied: {n}", "ok"))
                self.after(0, lambda: messagebox.showinfo("Success", "Files copied successfully."))
            except Exception as e:
                self.after(0, lambda e=e: self.log(f"Copy error: {e}", "err"))

        threading.Thread(target=copy_thread, daemon=True).start()

# ==========================================
# 7. TAB: TENSORBOARD
# ==========================================
class TabTensorBoard(BaseTab):
    def __init__(self, parent):
        super().__init__(parent, "TensorBoard 📈")
        
        # --- SECTION: TENSORBOARD ---
        lf = ttk.LabelFrame(self, text="TensorBoard (Run in background)")
        lf.grid(row=self.current_row, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        self.current_row += 1
        
        ttk.Label(lf, text="Log Directory:").grid(row=0, column=0, padx=5, sticky="e")
        
        self.widgets["tb_dir"] = tk.StringVar(value="") 
        
        # Entry + Browse Button (opens 2- trained_models by default)
        ttk.Entry(lf, textvariable=self.widgets["tb_dir"]).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(lf, text="Browse", command=lambda: self._browse_dir(self.widgets["tb_dir"], "2- trained_models")).grid(row=0, column=2, padx=5)
        
        btn_frm = ttk.Frame(lf)
        btn_frm.grid(row=1, column=0, columnspan=3, pady=10) 
        ttk.Button(btn_frm, text="🚀 Start TensorBoard", command=self.start_tb, style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frm, text="🌐 Open Browser (port 6006)", command=lambda: webbrowser.open("http://localhost:6006")).pack(side="left", padx=5)
        
        self.add_log_box()

    def start_tb(self):
        cmd = ["uv", "run", "tensorboard", "--logdir", self.widgets["tb_dir"].get(), "--port", "6006"]
        
        si = None
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        try:
            # Start TensorBoard process without waiting (daemonized)
            subprocess.Popen(cmd, startupinfo=si)
            self.log(f"TensorBoard started in background on http://localhost:6006", "ok")
        except FileNotFoundError:
            self.log("Error: 'uv' or 'tensorboard' not found. Check dependencies.", "err")


# ==========================================
#  MAIN APPLICATION CLASS
# ==========================================
class App(tk.Tk):
    def __init__(self):
        # 0. Initial Setup
        check_dependencies()
        ensure_directories()
        
        super().__init__()
        
        self.title("IndexTTS Trainer GUI")
        self.geometry("900x950")
        
        # Styling for a modern look
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="black")
        
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab Registration
        nb.add(TabExtendBPE(nb), text="1. Tokenizer")
        nb.add(TabPreprocess(nb), text="2. Preprocess")
        nb.add(TabPairs(nb), text="3. Pairs")
        nb.add(TabTrain(nb), text="4. Train")
        nb.add(TabPrune(nb), text="5. Prune")
        nb.add(TabFileManagement(nb), text="6. File Management")
        
        # Utility Tab at the end
        nb.add(TabTensorBoard(nb), text="TensorBoard")

if __name__ == "__main__":
    App().mainloop()