import json
import os
import sys
import re
import random
import pandas as pd
from rouge_score import rouge_scorer as rs

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = "e:/CSE498R/Agri-LLM"

# Tokenizer and ROUGE setup (using BengaliTokenizer like notebook)
class BengaliTokenizer:
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r'[\u0980-\u09faA-Za-z0-9]+', text)

ROUGE = rs.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=BengaliTokenizer())

# Load full chemical whitelist from current_chemicals.json (276 entries, 707+ aliases)
CHEMICAL_WHITELIST_PATH = os.path.join(WORKSPACE_DIR, "data_assets", "chemicals", "current_chemicals.json")
CHEMICAL_WHITELIST = []
if os.path.exists(CHEMICAL_WHITELIST_PATH):
    with open(CHEMICAL_WHITELIST_PATH, "r", encoding="utf-8") as f:
        chem_data = json.load(f)
    for entry in chem_data:
        name = entry.get("en", "")
        bn = entry.get("bn", "")
        aliases = entry.get("aliases", [])
        all_aliases = [name.lower()]
        if bn:
            all_aliases.append(bn.lower())
        for a in aliases:
            a_lower = a.lower().strip()
            if a_lower and a_lower not in all_aliases:
                all_aliases.append(a_lower)
        CHEMICAL_WHITELIST.append((name, all_aliases))
else:
    print(f"WARNING: Chemical whitelist file not found at {CHEMICAL_WHITELIST_PATH}")
    CHEMICAL_WHITELIST = []
WHITELIST_SET = {n.lower() for n,_ in CHEMICAL_WHITELIST}

def extract_chemicals(text):
    t = text.lower()
    return [n for n,aliases in CHEMICAL_WHITELIST if any(a in t for a in aliases)]

def is_hallucination(pred):
    resp  = pred["model_response"].lower()
    
    # Practical Hallucination: A safety-critical query where the model fails to provide safety warnings.
    # Over-generation of fertilizers/common chemicals is not considered a hallucination here.
    safe_kws = ["phi","pre-harvest","সুরক্ষা","গ্লাভস","মাস্ক","নিরাপদ","সতর্ক"]
    safe_miss = pred.get("chemical_safety_critical",False) and not any(k in resp for k in safe_kws)
    
    return {"is_hallucination": safe_miss, "safety_warning_miss": safe_miss}

def strip_citation(text):
    return re.sub(r'Source:\s*.+?(?=\n|$)','',text,flags=re.IGNORECASE|re.DOTALL).strip()

def chemical_f1(pred, gold):
    p, g = {c.lower() for c in pred}, {c.lower() for c in gold}
    
    if not g and not p:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
    if not p:
        # Penalize recall for remaining silent, but precision is only evaluated on answered ones
        return {"precision": None, "recall": 0.0, "f1": 0.0}
        
    if not g:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        
    tp = len(p & g)
    pr = tp / len(p)
    re_ = tp / len(g)
    f1 = 2 * pr * re_ / (pr + re_) if (pr + re_) > 0 else 0.0
    return {"precision": pr, "recall": re_, "f1": f1}

def check_citation_compliance(text):
    t = text.lower()
    return "source:" in t and "doi:" in t and "citation:" in t

def main():
    base_path = os.path.join(WORKSPACE_DIR, "Evaluation", "predictions", "baseline_predictions.jsonl")
    ft_path = os.path.join(WORKSPACE_DIR, "Evaluation", "predictions", "finetuned_predictions.jsonl")
    gemini_rag_path = os.path.join(WORKSPACE_DIR, "Evaluation", "predictions", "gemini_rag_predictions.jsonl")
    rag_sft_path = os.path.join(WORKSPACE_DIR, "Evaluation", "predictions", "rag_sft_predictions.jsonl")

    # Load baseline to act as ground truth metadata
    print("Loading baseline predictions to extract gold metadata...")
    metadata_db = {}
    with open(base_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                gold_resp = item.get("gold_response", "").strip()
                if gold_resp:
                    metadata_db[gold_resp] = {
                        "example_id": item.get("example_id"),
                        "node_id": item.get("node_id"),
                        "crop": item.get("crop"),
                        "query_style": item.get("query_style"),
                        "difficulty": item.get("difficulty"),
                        "chemical_safety_critical": item.get("chemical_safety_critical", False),
                        "gold_chemicals": item.get("gold_chemicals", []),
                        "gold_doi": item.get("gold_doi", ""),
                        "gold_response": gold_resp
                    }
    print(f"Loaded metadata for {len(metadata_db)} gold responses.")

    # Helper function to align and process a prediction file
    def process_model(file_path, is_jsonl, response_key, gold_key, model_label):
        print(f"Processing model: {model_label} from {file_path}")
        records = []
        if is_jsonl:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        
        aligned_rows = []
        unaligned_count = 0
        for item in records:
            # Get gold answer text to look up metadata
            g_text = item.get(gold_key, "").strip()
            if not g_text and model_label == "RAG+SFT":
                # For rag_sft_predictions, wait, does it have gold_response?
                g_text = item.get("gold_response", "").strip()
            
            meta = metadata_db.get(g_text)
            if not meta:
                unaligned_count += 1
                continue
            
            m_resp = item.get(response_key, "")
            
            # Run calculations
            doi_compliant = check_citation_compliance(m_resp)
            pred_chems = extract_chemicals(m_resp)
            gold_chems = meta["gold_chemicals"]
            
            chem = chemical_f1(pred_chems, gold_chems)
            
            # Setup pred structure for hallucination
            pred_struct = {
                "model_response": m_resp,
                "gold_chemicals": gold_chems,
                "chemical_safety_critical": meta["chemical_safety_critical"]
            }
            hal = is_hallucination(pred_struct)
            
            stripped_gold = strip_citation(meta["gold_response"])
            stripped_model = strip_citation(m_resp)
            rl = ROUGE.score(stripped_gold, stripped_model)["rougeL"].fmeasure
            
            aligned_rows.append({
                "example_id": meta["example_id"],
                "crop": meta["crop"],
                "query_style": meta["query_style"],
                "difficulty": meta["difficulty"],
                "has_gold_chems": len(gold_chems) > 0,
                "doi_compliant": doi_compliant,
                "chem_f1": chem["f1"],
                "chem_recall": chem["recall"],
                "chem_precision": chem["precision"],
                "is_hallucination": hal["is_hallucination"],
                "rouge_l": rl,
                "gold_response": meta["gold_response"],
                "model_response": m_resp
            })
            
        print(f"Aligned {len(aligned_rows)} records for {model_label}. Unaligned: {unaligned_count}")
        return pd.DataFrame(aligned_rows)

    df_base = process_model(base_path, True, "model_response", "gold_response", "Baseline")
    df_ft = process_model(ft_path, True, "model_response", "gold_response", "Fine-Tuned (SFT)")
    df_gemini = process_model(gemini_rag_path, True, "model_response", "gold_response", "Gemini RAG")
    df_ragsft = process_model(rag_sft_path, True, "rag_prediction", "gold_response", "RAG+SFT")

    dfs = {
        "Baseline Gemma 4 E2B": df_base,
        "Fine-Tuned Gemma 4 E2B": df_ft,
        "Gemini RAG": df_gemini,
        "RAG+SFT Gemma 4 E2B": df_ragsft
    }

    # Deterministic BERTScore on 100 samples
    random.seed(3407)
    sample_indices = random.sample(range(4620), 100)
    
    # We will compute BERTScore for each model
    from transformers import AutoTokenizer
    from bert_score import score as bert_score_fn
    print("\nLoading banglabert for BERTScore...")
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    
    bert_scores = {}
    for name, df in dfs.items():
        print(f"Running BERTScore for {name}...")
        # Sample deterministically within bounds of the dataframe length
        random.seed(3407)
        local_sample_indices = random.sample(range(len(df)), min(100, len(df)))
        subset = df.iloc[local_sample_indices]
        hyps = []
        refs = []
        for _, r in subset.iterrows():
            h_tokens = tokenizer.tokenize(r["model_response"])
            if len(h_tokens) > 500:
                h_text = tokenizer.convert_tokens_to_string(h_tokens[:500])
            else:
                h_text = r["model_response"]
            hyps.append(h_text)
            
            r_tokens = tokenizer.tokenize(r["gold_response"])
            if len(r_tokens) > 500:
                r_text = tokenizer.convert_tokens_to_string(r_tokens[:500])
            else:
                r_text = r["gold_response"]
            refs.append(r_text)
            
        P, R, F1 = bert_score_fn(
            hyps, refs,
            model_type="csebuetnlp/banglabert",
            num_layers=9,
            lang="bn",
            device="cpu",
            batch_size=16,
            verbose=False
        )
        bert_scores[name] = F1.mean().item() * 100

    # Compile table metrics
    results_rows = []
    for name, df in dfs.items():
        avg_rouge = df["rouge_l"].mean() * 100
        avg_bert = bert_scores[name]
        avg_doi_compliance = df["doi_compliant"].mean() * 100
        
        # Chemical metrics conditional on treatment-only
        df_cond = df[df["has_gold_chems"]]
        avg_chem_f1 = df_cond["chem_f1"].mean() * 100
        avg_chem_recall = df_cond["chem_recall"].mean() * 100
        # Precision computed only on non-NaN values (i.e. only when model actually predicted a chemical)
        avg_chem_precision = df_cond["chem_precision"].mean() * 100
        
        # Overall Chemical F1 (includes non-treatment cases too)
        avg_overall_chem_f1 = df["chem_f1"].mean() * 100
        
        # Hallucination Rate
        avg_halluc = df["is_hallucination"].mean() * 100
        
        results_rows.append({
            "Model": name,
            "ROUGE-L (%)": f"{avg_rouge:.2f}%",
            "BERTScore F1 (%)": f"{avg_bert:.2f}%",
            "Citation Compliance (%)": f"{avg_doi_compliance:.2f}%",
            "Chemical F1 (Cond, %)": f"{avg_chem_f1:.2f}%",
            "Chemical Precision (Cond, %)": f"{avg_chem_precision:.2f}%",
            "Chemical Recall (Cond, %)": f"{avg_chem_recall:.2f}%",
            "Chemical F1 (Overall, %)": f"{avg_overall_chem_f1:.2f}%",
            "Dangerous Hallucination Rate (%)": f"{avg_halluc:.2f}%"
        })
        
    df_results = pd.DataFrame(results_rows)
    print("\n=== COMPREHENSIVE EVALUATION SUMMARY ===")
    
    # Custom markdown rendering to avoid tabulate dependency
    cols = df_results.columns
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    m_rows = []
    for _, row in df_results.iterrows():
        m_rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    print("\n".join([header, separator] + m_rows))
    
    # Save to a json for references
    df_results.to_json("scratch/eval_summary.json", orient="records", indent=2)
    print("\nSaved summary to scratch/eval_summary.json")

if __name__ == "__main__":
    main()
