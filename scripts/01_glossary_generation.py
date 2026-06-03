import os
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from threading import Lock

# Constants
RPM_LIMIT = 5
COOLDOWN = 60 / RPM_LIMIT  # seconds per request per key
MAX_WORKERS = 3

# Paths
ENV_FILE = ".env"
INPUT_FILES = [
    "evaluation/predictions/finetuned_predictions.jsonl",
    "evaluation/predictions/rag_sft_predictions.jsonl",
    "evaluation/predictions/baseline_predictions.jsonl",
    "docs/agri_expert_corpus_bengali.json"
]
OUTPUT_GLOSSARY = "data_assets/glossary/current_glossary.json"
OUTPUT_CHEMICALS = "data_assets/chemicals/current_chemicals.json"

# Load API Keys
def load_api_keys():
    keys = []
    if not os.path.exists(ENV_FILE):
        return keys
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY_"):
                parts = line.strip().split("=")
                if len(parts) >= 2:
                    val = parts[1].split()[0] # remove comments
                    keys.append(val)
    return keys

API_KEYS = load_api_keys()
print(f"Loaded {len(API_KEYS)} API keys.")

if not API_KEYS:
    print("No API keys found. Exiting.")
    exit(1)

# Rate limiting state
disabled_keys = set()
exhausted_until = {}  # key -> timestamp when it becomes usable again after 429
key_last_used = {key: 0.0 for key in API_KEYS}
key_lock = Lock()

def get_next_available_key():
    while True:
        with key_lock:
            now = time.time()
            # Filter active keys
            active_keys = [k for k in API_KEYS if k not in disabled_keys]
            if not active_keys:
                raise Exception("All API keys have been disabled or are invalid!")
                
            # Filter keys that are NOT currently in 429 cooldown
            usable_keys = []
            for k in active_keys:
                resume_time = exhausted_until.get(k, 0.0)
                if now >= resume_time:
                    usable_keys.append(k)
            
            if not usable_keys:
                # If all keys are in 429 cooldown, wait for the one that resumes earliest
                earliest_resume = min(exhausted_until.get(k, 0.0) for k in active_keys)
                wait_time = earliest_resume - now
                if wait_time > 0:
                    print(f"All keys exhausted. Waiting {wait_time:.1f}s for earliest key recovery...")
                    time.sleep(wait_time)
                continue
                
            # Find key with the longest time since last use among usable ones
            best_key = None
            best_time = float("inf")
            for key in usable_keys:
                last_used = key_last_used.get(key, 0.0)
                if last_used < best_time:
                    best_time = last_used
                    best_key = key
            
            elapsed = now - best_time
            if elapsed < COOLDOWN:
                time.sleep(COOLDOWN - elapsed)
                
            key_last_used[best_key] = time.time()
            return best_key

def setup_model(api_key):
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
        "response_mime_type": "application/json",
    }
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    return genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

def extract_terms(text_batch):
    prompt = f"""You are an expert agricultural botanist and linguist.
Extract any technical agricultural terminology, crop diseases, pests, and agrochemicals from the following text batch.

Output JSON format EXACTLY as:
{{
  "glossary": [
    {{"bn": "Bengali Term", "en": "English Term", "category": "disease/pest/general"}}
  ],
  "chemicals": [
    {{"bn": "Chemical Name (BN)", "en": "Chemical Name (EN)", "mode_of_action": "Brief mode of action", "type": "fungicide/pesticide/fertilizer/herbicide"}}
  ]
}}

Text to process:
{text_batch}
"""
    retries = 8  # Allow multiple retries to find a good working key
    for attempt in range(retries):
        try:
            key = get_next_available_key()
        except Exception as e:
            print(f"Error getting key: {e}")
            break
            
        model = setup_model(key)
        try:
            response = model.generate_content(prompt)
            # Safeguard JSON parse
            try:
                data = json.loads(response.text)
                return data
            except json.JSONDecodeError as je:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                data = json.loads(cleaned_text.strip())
                return data
        except Exception as e:
            err_msg = str(e)
            print(f"Error on attempt {attempt+1} with key ...{key[-6:] if len(key) > 6 else 'key'}: {err_msg}")
            
            # Handle key issues
            if "API Key not found" in err_msg or "API_KEY_INVALID" in err_msg or "not found" in err_msg.lower():
                with key_lock:
                    disabled_keys.add(key)
                print(f"--> Disabled invalid API key: ...{key[-6:] if len(key) > 6 else 'key'}")
            elif "denied access" in err_msg or "403" in err_msg:
                with key_lock:
                    disabled_keys.add(key)
                print(f"--> Disabled forbidden/blocked API key: ...{key[-6:] if len(key) > 6 else 'key'}")
            elif "429" in err_msg or "exhausted" in err_msg.lower() or "quota" in err_msg.lower():
                with key_lock:
                    exhausted_until[key] = time.time() + 300
                print(f"--> Quota exhausted (429) for key: ...{key[-6:] if len(key) > 6 else 'key'}. Cooling down for 300s.")
            
            time.sleep(2)
    return {"glossary": [], "chemicals": []}

GLOBAL_START_TIME = time.time()
BATCHES_COMPLETED_GLOBAL = 0
TOTAL_BATCHES_ESTIMATED = 0

def save_realtime_results(new_g, new_c, current_file, current_batch_num, total_batches_file):
    global BATCHES_COMPLETED_GLOBAL
    BATCHES_COMPLETED_GLOBAL += 1
    
    # Load existing glossary
    existing_g = []
    if os.path.exists(OUTPUT_GLOSSARY):
        try:
            with open(OUTPUT_GLOSSARY, "r", encoding="utf-8") as f:
                existing_g = json.load(f)
        except Exception:
            existing_g = []
            
    # Load existing chemicals
    existing_c = []
    if os.path.exists(OUTPUT_CHEMICALS):
        try:
            with open(OUTPUT_CHEMICALS, "r", encoding="utf-8") as f:
                existing_c = json.load(f)
        except Exception:
            existing_c = []

    # Merge
    existing_g.extend(new_g)
    existing_c.extend(new_c)
    
    # Deduplicate based on unique fields
    def dedup(lst):
        seen = set()
        unique = []
        for x in lst:
            rep = json.dumps(x, sort_keys=True)
            if rep not in seen:
                seen.add(rep)
                unique.append(x)
        return unique

    unique_g = dedup(existing_g)
    unique_c = dedup(existing_c)
    
    # Write back to files in real-time
    with open(OUTPUT_GLOSSARY, "w", encoding="utf-8") as f:
        json.dump(unique_g, f, ensure_ascii=False, indent=2)
        
    with open(OUTPUT_CHEMICALS, "w", encoding="utf-8") as f:
        json.dump(unique_c, f, ensure_ascii=False, indent=2)
        
    # Calculate stats
    now = time.time()
    elapsed = now - GLOBAL_START_TIME
    
    # Estimate times
    avg_time_per_batch = elapsed / BATCHES_COMPLETED_GLOBAL if BATCHES_COMPLETED_GLOBAL > 0 else 0
    remaining_batches = max(0, TOTAL_BATCHES_ESTIMATED - BATCHES_COMPLETED_GLOBAL)
    est_remaining_sec = avg_time_per_batch * remaining_batches
    
    with key_lock:
        active_cnt = len(API_KEYS) - len(disabled_keys)
        disabled_cnt = len(disabled_keys)
        cooldown_cnt = sum(1 for k in API_KEYS if k not in disabled_keys and exhausted_until.get(k, 0.0) > now)

    stats = {
        "current_file": current_file,
        "current_file_progress": f"{current_batch_num}/{total_batches_file}",
        "global_batches_progress": f"{BATCHES_COMPLETED_GLOBAL}/{TOTAL_BATCHES_ESTIMATED}",
        "elapsed_time_seconds": round(elapsed, 1),
        "estimated_remaining_time_minutes": round(est_remaining_sec / 60, 1),
        "estimated_remaining_time_seconds": round(est_remaining_sec, 1),
        "total_extracted_unique_glossary": len(unique_g),
        "total_extracted_unique_chemicals": len(unique_c),
        "api_keys_status": {
            "total_keys": len(API_KEYS),
            "active_working_keys": active_cnt,
            "disabled_invalid_keys": disabled_cnt,
            "keys_in_429_cooldown": cooldown_cnt
        }
    }
    
    # Write progress file
    PROGRESS_FILE = "data_assets/extraction_progress.json"
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print(f"[{BATCHES_COMPLETED_GLOBAL}/{TOTAL_BATCHES_ESTIMATED}] Real-time saved. Glossary size: {len(unique_g)}, Chemicals size: {len(unique_c)}. Est remaining: {round(est_remaining_sec / 60, 1)}m", flush=True)

def process_file(file_path):
    print(f"Reading {file_path}", flush=True)
    text_batches = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        current_batch = ""
        for i, line in enumerate(f):
            current_batch += line
            if len(current_batch) > 50000:
                text_batches.append(current_batch)
                current_batch = ""
        if current_batch:
            text_batches.append(current_batch)
            
    text_batches = text_batches[:300] 
    total_batches_file = len(text_batches)
    print(f"Created {total_batches_file} batches from {file_path}", flush=True)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_terms, batch): i+1 for i, batch in enumerate(text_batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"Future error for batch {batch_idx}: {e}", flush=True)
                res = {"glossary": [], "chemicals": []}
                
            new_g = res.get("glossary", []) if res else []
            new_c = res.get("chemicals", []) if res else []
            
            # Print immediate count
            print(f"Batch {batch_idx}/{total_batches_file} finished extraction: {len(new_g)} glossary, {len(new_c)} chemicals.", flush=True)
            
            # Realtime merge & save progress
            save_realtime_results(new_g, new_c, file_path, batch_idx, total_batches_file)

def main():
    global TOTAL_BATCHES_ESTIMATED, GLOBAL_START_TIME
    GLOBAL_START_TIME = time.time()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(OUTPUT_GLOSSARY), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CHEMICALS), exist_ok=True)
    
    # Pre-calculate total batches across all files
    total_batches = 0
    valid_files = []
    
    for file_path in INPUT_FILES:
        if os.path.exists(file_path):
            valid_files.append(file_path)
            # Estimate batches for this file
            with open(file_path, "r", encoding="utf-8") as f:
                char_count = 0
                for line in f:
                    char_count += len(line)
            batches = (char_count // 50000) + 1
            if batches > 300:
                batches = 300
            total_batches += batches
        else:
            print(f"File not found: {file_path}", flush=True)
            
    TOTAL_BATCHES_ESTIMATED = total_batches
    print(f"Estimated total batches across all files: {TOTAL_BATCHES_ESTIMATED}", flush=True)
    
    for file_path in valid_files:
        process_file(file_path)
        
    print("All files processed completely!", flush=True)

if __name__ == "__main__":
    import sys
    if "--dry-run" in sys.argv:
        print("Dry run complete. API keys loaded successfully.", flush=True)
    else:
        main()
