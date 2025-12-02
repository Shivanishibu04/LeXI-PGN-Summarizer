# Step-by-Step Guide: Running Summarization

## 🚀 Quick Run (5 Steps)

### Step 1: Open Terminal
Open PowerShell (Windows) or Terminal (Linux/Mac) and navigate to the project:

```bash
cd C:\Users\sivap\OneDrive\Desktop\S7_Project_try\dev2\LexiDesk
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

✅ You should see `(venv)` in your prompt.

### Step 3: Install Dependencies (if not done)

```bash
pip install -r requirements.txt
```

### Step 4: Check Models Exist

```bash
# Windows
dir saved_models

# Linux/Mac
ls saved_models
```

✅ Should see: `cnn_model.pth`, `crf_baseline_model.joblib`, `crf_hybrid_model.joblib`

❌ If missing, run: `python -m src.crf_model` (takes 10-30 minutes)

### Step 5: Run Summarization

```bash
python predict.py --summarize
```

🎉 Done! You should see the summary output.

---

## 📝 Detailed Examples

### Example 1: Basic Summarization
```bash
python predict.py --summarize
```
Uses the default sample legal text.

### Example 2: From File
```bash
# Create a text file first
echo "Your legal text here..." > my_text.txt

# Then run
python predict.py --summarize --text-file my_text.txt
```

### Example 3: Top 20% of Sentences
```bash
python predict.py --summarize --text-file my_text.txt --compression 0.2
```

### Example 4: Top 3 Sentences
```bash
python predict.py --summarize --text-file my_text.txt --top-k 3
```

### Example 5: From stdin
```bash
# Windows PowerShell
Get-Content my_text.txt | python predict.py --summarize --stdin

# Linux/Mac
cat my_text.txt | python predict.py --summarize --stdin
```

### Example 6: With Enhanced Embeddings
```bash
# First install: pip install sentence-transformers
python predict.py --summarize --text-file my_text.txt --use-embeddings
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model file not found" | Run: `python -m src.crf_model` |
| "ModuleNotFoundError: networkx" | Run: `pip install networkx` |
| "NLTK data not found" | Run: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"` |
| Virtual env not activating | Make sure you're in the project directory |

---

## 📊 Expected Output

```
==================================================
--- Summarization Mode ---

Original Text:
---
[Your text]
---

Segmenting input...

Segmented Sentences:
[1] First sentence...
[2] Second sentence...
[3] Third sentence...

Computing sentence weights...

Weights (normalized):
[1] 0.364 (36.4%)
[2] 0.210 (21.0%)
[3] 0.280 (28.0%)

Summary (top 2 sentences):
1) [1] First sentence... — 36.4%
2) [3] Third sentence... — 28.0%

==================================================
```

---

## 🎯 All Available Options

```bash
python predict.py --summarize [OPTIONS]

Options:
  --summarize          Enable summarization mode
  --text-file PATH     Path to input text file
  --stdin              Read input from stdin
  --compression RATIO  Compression ratio (0-1), e.g., 0.2 = 20%
  --top-k N            Number of top sentences
  --preserve-order     Keep original sentence order (default: True)
  --use-embeddings     Use sentence-transformers (better quality)
```

---

## 📚 More Information

- **Detailed Guide**: See [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Design Docs**: See [docs/SUMMARIZER_DESIGN.md](docs/SUMMARIZER_DESIGN.md)
- **Examples**: See [examples/summary_example.txt](examples/summary_example.txt)


