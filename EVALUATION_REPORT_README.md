# LeXiDesk Phase-1 Evaluation Report

## Overview

This directory contains the comprehensive evaluation report for Phase-1 modules:
1. **Sentence Boundary Detection (SBD)**: Hybrid CNN + CRF model
2. **Summarization**: Weighted Extractive Summarizer

## Files Created

- `LeXiDesk_SBD_Summarizer_Report.ipynb` - Main evaluation notebook
- `results/` - Directory containing all evaluation outputs
- `results/plots/` - Directory containing all visualization plots

## Running the Evaluation

### Option 1: Using Jupyter Notebook (Recommended)

1. **Install Jupyter** (if not already installed):
   ```bash
   pip install jupyter notebook
   ```

2. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Open the notebook**:
   - Navigate to `LeXiDesk_SBD_Summarizer_Report.ipynb`
   - Click to open it

4. **Run all cells**:
   - Go to `Cell > Run All`
   - Or use `Kernel > Restart & Run All`

5. **Export to PDF**:
   - After all cells execute successfully, go to `File > Download as > PDF via LaTeX (.pdf)`
   - OR: `File > Download as > HTML`, then open in browser and Print to PDF

### Option 2: Using nbconvert (Command Line)

1. **Install nbconvert**:
   ```bash
   pip install nbconvert
   ```

2. **Execute and export to PDF**:
   ```bash
   jupyter nbconvert --to pdf --execute LeXiDesk_SBD_Summarizer_Report.ipynb
   ```

   Note: PDF export requires LaTeX. If LaTeX is not installed:
   ```bash
   # Export to HTML instead
   jupyter nbconvert --to html --execute LeXiDesk_SBD_Summarizer_Report.ipynb
   ```

### Option 3: Manual Execution

If you encounter issues with Jupyter/nbconvert:

1. The notebook will automatically generate synthetic demo data if input files are missing
2. All code cells are self-contained and can be run independently
3. Results will be saved to the `results/` directory automatically

## Input Data Format

The notebook expects the following input files (optional - will generate synthetic data if missing):

### Sentence Boundary Detection:
- `data/sbd_gold.csv`: Columns: `doc_id`, `token_index`, `gold_label`
- `outputs/sbd_pred.csv`: Columns: `doc_id`, `token_index`, `pred_label`

### Summarization:
- `data/summ_refs.jsonl`: JSON lines with `{"doc_id": "...", "summary": "..."}`
- `outputs/summ_preds.jsonl`: JSON lines with `{"doc_id": "...", "summary": "..."}`

## Output Files

After running the notebook, you will find:

### CSV Tables:
- `results/sbd_metrics.csv` - Overall SBD metrics
- `results/sbd_per_document_metrics.csv` - Per-document SBD metrics
- `results/summarizer_metrics.csv` - Overall summarization metrics
- `results/summarizer_rouge_per_document.csv` - Per-document ROUGE scores
- `results/summarizer_bertscore_per_document.csv` - Per-document BERTScore
- `results/summarizer_length_stats.csv` - Summary length statistics
- `results/summarizer_qualitative_comparison.csv` - Sample qualitative comparisons
- `results/final_summary.csv` - Key metrics summary

### JSON Files:
- `results/bootstrap_confidence_intervals.json` - Bootstrap CI results
- `results/ablation_framework.json` - Ablation study framework

### Plots (PNG + PDF):
- `results/plots/sbd_metrics_bar.png/pdf` - SBD core metrics bar chart
- `results/plots/sbd_confusion_matrix.png/pdf` - SBD confusion matrix
- `results/plots/sbd_per_document_f1.png/pdf` - Per-document F1 scores
- `results/plots/rouge_metrics.png/pdf` - ROUGE metrics visualization
- `results/plots/summary_length_comparison.png/pdf` - Length comparison
- `results/plots/rougeL_vs_length.png/pdf` - ROUGE-L vs length scatter
- `results/plots/bootstrap_distributions.png/pdf` - Bootstrap distributions

## Dependencies

The notebook will automatically install required packages, but you can pre-install them:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn rouge-score bert-score tqdm
```

For PDF export:
```bash
pip install nbconvert
# Also need LaTeX for PDF export (or use HTML export instead)
```

## Troubleshooting

### Issue: nbconvert not found
- Solution: Install with `pip install nbconvert`
- Alternative: Use Jupyter Notebook GUI instead

### Issue: PDF export fails
- Solution: Export to HTML first, then use browser Print to PDF
- Alternative: Install LaTeX distribution (MiKTeX on Windows, TeX Live on Linux/Mac)

### Issue: Missing input files
- Solution: The notebook will automatically generate synthetic demo data
- To use real data: Place your evaluation files in the expected locations (see Input Data Format above)

### Issue: BERTScore computation is slow
- This is normal - BERTScore uses deep learning models
- The notebook uses CPU by default to avoid GPU memory issues
- For faster computation, install CUDA and modify the device parameter

## Contact

For issues or questions about the evaluation report, please refer to the project documentation or create an issue in the repository.

