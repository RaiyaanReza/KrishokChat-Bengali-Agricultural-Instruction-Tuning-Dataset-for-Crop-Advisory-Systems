import os
import sys
import json
import re
import time
import multiprocessing
import urllib.request
import google.generativeai as genai

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
NODES_PATH = os.path.join(WORKSPACE_DIR, "Agri-LLM", "docs", "agri_expert_knowledge_nodes.json")
OUTPUT_SFT_PATH = os.path.join(WORKSPACE_DIR, "Agri-LLM", "dataset", "agri_expert_sft_dataset_bengali_20k.jsonl")
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "Agri-LLM", "docs", "qa_generation_checkpoint.json")

# 28 highly distinct thematic seeds to ensure absolute uniqueness across calls
FOCAL_SEEDS = [
    "Early vegetative stage leaf symptoms (spots, color changes, early signs).",
    "Late reproductive stage symptoms (grain discoloration, sheath rot, neck blast, whiteheads).",
    "Nursery bed/seedling stage symptoms, early rotting, or damping off scenarios.",
    "Chemical spraying, specific active ingredients, correct fungicide dilutions, and spraying equipment.",
    "Southern salt-prone zones (Barisal, Khulna) & seasonal Aman crop context.",
    "Prevention through land preparation, soil treatment, balanced potash application, and weed control.",
    "Eastern high-rainfall zones (Sylhet, Chittagong dialects) and waterlogging scenarios.",
    "Post-harvest storage issues, grain moisture, and pest infestations in warehouses.",
    "Non-chemical cultural control, hand weeding, crop rotation, and Boro winter season context.",
    "Northern drought-prone zones (Rajshahi, Rangpur dialects) and dual-pest attacks.",
    "Severe leaf drying, yellowing, and late-stage crop collapses.",
    "Early morning bacterial ooze observations, flag leaf infections, and plant breeding resistance.",
    "Fungal spore transmission, wind/rain spread, and field sanitation practices.",
    "Fertilizer dose adjustment, avoiding excessive nitrogen, and crop rotation schedules.",
    "Comparative diagnostic confusion (mistaking leaf spots for insect tunnels or nutrient deficiency).",
    "Flooding, standing water drainage, and Aus crop season context.",
    "Organic farming methods, bio-pesticides, and safe harvesting practices.",
    "Economic impacts, crop loss estimation, and market price reductions due to disease.",
    "Weather resilience, preemptive actions before cyclones or heavy storms.",
    "Nutritional impacts of diseases on grain quality and safety for consumption.",
    "Pesticide safety, protective gear, and safe chemical handling for farmers.",
    "Deep-water rice varieties and haor basin (wetland) specific challenges.",
    "Hill tract farming (Jhum cultivation) and terrace-specific agricultural issues.",
    "Seed treatment before sowing and sourcing certified disease-free seeds.",
    "Irrigation management, alternate wetting and drying (AWD), and moisture stress.",
    "Beneficial insects, natural predators, and avoiding harm to pollinators.",
    "Mixed cropping, intercropping, and border planting for disease barrier.",
    "Soil testing, pH imbalance symptoms, and micro-nutrient (Zinc, Boron) deficiencies.",
    "Interdisciplinary agro-forestry practices, shade crop integration, and organic fungal management in mixed-orchard settings.",
    "Traditional heritage farming wisdom (locally-adapted practices) contrasted with modern scientific advisory and chemical-dosage guidelines.",
    "Post-cyclone recovery, silt deposition, flash flood crop drainage, and rapid field disease diagnostic protocols under waterlogging.",
    "Foliar fertilizer spraying schedules, micro-nutrient absorption rates, and root-dip treatment formulations for high-yield transplantation."
]

PROMPT_TEMPLATE = """
You are a senior agricultural extension specialist in Bangladesh's Department of Agricultural Extension.
Your task is to generate exactly 15 highly unique, diverse, and practical Question-Answer pairs in Bengali based on the provided Knowledge Node.

RULES FOR THE QUESTIONS:
1. Generate queries in the following specific farmer styles:
   - "formal_bangla": standard formal Bengali query.
   - "colloquial": spoken, informal Bengali.
   - "typo_noisy": query with typos, bad spelling, or double characters.
   - "dialect_hint": queries with regional dialects (Sylheti, Chatgaya, Rangpuri, or Barisal style).
   - "vague_description": vague query where the farmer describes their leaves/plant looking "bad" or "unhealthy".
   - "wrong_assumption": farmer wrongly assumes a virus is a pest, or a disease is caused by fertilizer, etc.
   - "chemical_specific": farmer directly asks about a fungicide, pesticide, or active ingredient.
   - "seasonal_context": relating to a specific rice/crop season (Aman, Boro, Aus).
   - "location_context": relating to a specific district in Bangladesh (e.g., Barisal, Rangpur, Jessore).
   - "multi_symptom": farmer describes multiple overlapping issues.

2. Do NOT make the questions repetitive. Each question must describe a unique, distinct farmer situation, crop stage, or problem aspect.
3. STRICTLY FOCUS on this focal theme for this run to ensure absolute uniqueness:
   Focal Seed & Context: {focal_seed}

RULES FOR THE ANSWERS:
1. Every answer must follow a DUAL-STRUCTURE:
   - Part 1: Empathetic, highly understandable, and formal agricultural Bengali guidance. Talk humbly and respectfully to the farmer, addressing them with dynamic, varied neutral greetings to keep it natural (e.g., vary between "শুভ দিন, আশা করি ভালো আছেন", "কেমন আছেন কৃষক ভাই?", or sometimes directly jumping into the advice without a greeting to avoid monotony). STRICTLY AVOID any religious greetings (such as "আসসালামু আলাইকুম", "আসসালামু আলাইকুম রহমতুল্লাহ", or "নমস্কার") anywhere in the text. Explain the diagnosis, chemical name, exact dosage, and preventive cultural actions in simple, clear, and action-oriented terms. Crucially, when recommending any chemical, pesticide, fungicide, or fertilizer, do NOT just state the name abruptly. Start by providing a brief, farmer-friendly explanation of what the chemical is or how it works (e.g., its mode of action, such as "এটি একটি পদ্ধতিগত ছত্রাকনাশক যা ছত্রাকের বৃদ্ধি বন্ধ করে" or "এটি একটি স্পর্শক বালাইনাশক যা পোকার স্নায়ুতন্ত্র নিষ্ক্রিয় করে") to maintain semantic clarity and educational value for the farmer.
   - Part 2: Rigorous academic citations in English, formatted exactly as:
     "Source: {source} | DOI: {doi} | Citation: {academic_citation}"

2. Output ONLY a valid JSON list of exactly 15 objects. Each object must have keys: "style" (one of the 10 query styles), "question_bn", and "answer_bn". Do not include markdown fences outside the JSON.

KNOWLEDGE NODE DATA:
{node_json}
"""

def load_keys_from_env():
    env_path = os.path.join(WORKSPACE_DIR, "Agri-LLM", ".env")
    keys = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r"GEMINI_API_KEY_(\d+)\s*=\s*([A-Za-z0-9_-]+)", line)
                if match:
                    key_num = int(match.group(1))
                    keys.append((key_num, match.group(2)))
    keys.sort(key=lambda x: x[0])
    return [k[1] for k in keys]

def test_single_key(key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": "Hello"}]}],
        "generationConfig": {"maxOutputTokens": 5}
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15.0) as response:
            return True
    except Exception:
        return False

def generate_qa_batch(key, node, seed):
    prompt = PROMPT_TEMPLATE.format(
        focal_seed=seed,
        source=node.get("source", "IRRI/BRRI Manual"),
        doi=node.get("doi", "N/A"),
        academic_citation=node.get("academic_citation", "N/A"),
        node_json=json.dumps(node, ensure_ascii=False)
    )
    
    genai.configure(api_key=key)
    # Strictly enforce gemini-3.1-flash-lite model as requested
    model_name = 'models/gemini-3.1-flash-lite'
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "response_mime_type": "application/json"
                }
            )
            result = json.loads(response.text.strip())
            if isinstance(result, list) and len(result) > 0:
                return result
        except Exception as e:
            err_str = str(e)
            print(f"  [Attempt {attempt+1}/{max_retries}] API call failed: {err_str}")
            if attempt < max_retries - 1:
                sleep_time = 15.0 if "429" in err_str else 5.0
                print(f"  Sleeping for {sleep_time}s before retrying...")
                time.sleep(sleep_time)
            else:
                break
    return None

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return set(tuple(x) for x in json.load(f))
        except Exception:
            pass
    return set()

def save_checkpoint(filepath, completed_runs):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(list(completed_runs), f, indent=2, ensure_ascii=False)
    except Exception:
        pass

class RotatingKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.last_used = {k: 0.0 for k in keys}
        self.use_count = {k: 0 for k in keys}

    def get_key(self):
        if not self.keys:
            return None
        
        # Find a key that hasn't exceeded 400 uses
        start_index = self.index
        while True:
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            if self.use_count[key] < 400:
                break
            if self.index == start_index:
                # All keys in this worker have reached the 400 RPD limit!
                # Reuse the key but warn
                print("WARNING: All keys in this worker have reached the 400 RPD limit!")
                break
        
        # Enforce RPM 8 (minimum 7.5 seconds between uses of the same key)
        now = time.time()
        elapsed = now - self.last_used[key]
        if elapsed < 7.5:
            sleep_needed = 7.5 - elapsed
            time.sleep(sleep_needed)
            
        self.last_used[key] = time.time()
        self.use_count[key] += 1
        return key

def worker_process(worker_id, worker_keys, assigned_tasks, file_lock):
    print(f"[Worker {worker_id}] Started with {len(worker_keys)} keys and {len(assigned_tasks)} tasks.")
    
    # Load nodes locally to prevent pickling overhead across multiprocessing boundaries on Windows
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        nodes = json.load(f)
        
    key_manager = RotatingKeyManager(worker_keys)
    total_calls = len(assigned_tasks)
    
    for run_idx, (node_idx, seed_idx) in enumerate(assigned_tasks):
        node = nodes[node_idx]
        seed = FOCAL_SEEDS[seed_idx]
        run_key = (node["node_id"], seed_idx)
        
        # Pull global checkpoint cleanly via lock to ensure we don't repeat
        with file_lock:
            global_completed = load_checkpoint(CHECKPOINT_PATH)
        if run_key in global_completed:
            continue
            
        print(f"[Worker {worker_id} - {run_idx+1}/{total_calls}] Processing '{node['node_id']}' | Seed {seed_idx+1}/{len(FOCAL_SEEDS)}")
        
        # Rotate key locally within this worker's key group
        active_key = key_manager.get_key()
        if not active_key:
            print(f"  [Worker {worker_id}] Error: No active keys assigned!")
            time.sleep(5.0)
            continue
            
        qa_batch = generate_qa_batch(active_key, node, seed)
        if qa_batch:
            print(f"  [Worker {worker_id}] Success! Generated {len(qa_batch)} QA pairs.")
            
            # --- CRITICAL: Use the multiprocessing lock to write safely to the UNIFIED file ---
            with file_lock:
                with open(OUTPUT_SFT_PATH, "a", encoding="utf-8") as f_out:
                    for qa in qa_batch:
                        q_bn = qa.get("question_bn", "") or qa.get("question", "")
                        a_bn = qa.get("answer_bn", "") or qa.get("answer", "")
                        
                        # Strict policy compliance check and replace
                        for bad_greeting in ["আসসালামু আলাইকুম রহমতুল্লাহ", "আসসালামু আলাইকুম", "নমস্কার"]:
                            q_bn = q_bn.replace(bad_greeting, "")
                            a_bn = a_bn.replace(bad_greeting, "")
                        # Clean up any leading punctuation or whitespace left after removing greetings
                        for _ in range(5):
                            q_bn = q_bn.strip().lstrip(",").lstrip("।").lstrip(".").lstrip("?").lstrip("!")
                            a_bn = a_bn.strip().lstrip(",").lstrip("।").lstrip(".").lstrip("?").lstrip("!")
                        q_bn = q_bn.strip()
                        a_bn = a_bn.strip()
                        
                        sft_object = {
                            "node_id": node["node_id"],
                            "crop": node.get("crop", "Unknown"),
                            "style": qa.get("style", "unknown"),
                            "question": q_bn,
                            "answer": a_bn
                        }
                        f_out.write(json.dumps(sft_object, ensure_ascii=False) + "\n")
                
                # Update checkpoint
                chk_data = load_checkpoint(CHECKPOINT_PATH)
                chk_data.add(run_key)
                save_checkpoint(CHECKPOINT_PATH, chk_data)
        else:
            print(f"  [Worker {worker_id}] Failed to generate QA batch for run {run_key}.")
            
        # Sleep 1.0s to allow quick task processing while the KeyManager handles rates per-key
        time.sleep(1.0)
        
    print(f"[Worker {worker_id}] Finished processing.")

def main():
    api_keys = load_keys_from_env()
    # Deduplicate key values while preserving order
    seen = set()
    unique_api_keys = []
    for k in api_keys:
        if k not in seen:
            seen.add(k)
            unique_api_keys.append(k)
    api_keys = unique_api_keys

    if not api_keys:
        print("ERROR: No keys found in .env")
        sys.exit(1)
        
    print(f"Loaded {len(api_keys)} unique API keys from .env. testing active status specifically with gemini-3.1-flash-lite...")
    working_keys = []
    for idx, key in enumerate(api_keys, 1):
        if test_single_key(key):
            working_keys.append(key)
            print(f"  - Key {idx}: ACTIVE")
        else:
            print(f"  - Key {idx}: INACTIVE / BLOCKED")
            
    print(f"Total working keys: {len(working_keys)} / {len(api_keys)}")
    if len(working_keys) < 12:
        print(f"WARNING: User requested 2 workers with 6 keys each (12 keys total), but only {len(working_keys)} keys are active.")
        print("Proceeding by grouping the active keys into 2 parallel workers...")
    
    if not working_keys:
        print("ERROR: No working keys available.")
        sys.exit(1)
        
    if not os.path.exists(NODES_PATH):
        print(f"Error: Knowledge Nodes file not found at {NODES_PATH}")
        sys.exit(1)
        
    with open(NODES_PATH, "r", encoding="utf-8") as f:
        nodes = json.load(f)
        
    global_completed_runs = load_checkpoint(CHECKPOINT_PATH)
    global_completed_set = set(tuple(x) for x in global_completed_runs)
    
    all_tasks = [(node_idx, seed_idx) 
                 # Pass indices only to avoid large multiprocessing pickling overhead on Windows
                 for node_idx, node in enumerate(nodes) 
                 for seed_idx in range(len(FOCAL_SEEDS))]
                 
    pending_tasks = [task for task in all_tasks if (nodes[task[0]]["node_id"], task[1]) not in global_completed_set]
    
    if not pending_tasks:
        print("All runs are already completed!")
        sys.exit(0)
        
    # User Request: Four parallel workers
    num_workers = 4
    print(f"Configuring exactly {num_workers} parallel workers...")
    
    # Split working keys into exactly 4 groups
    chunk_keys = len(working_keys) // num_workers
    keys_w0 = working_keys[:chunk_keys]
    keys_w1 = working_keys[chunk_keys:2*chunk_keys]
    keys_w2 = working_keys[2*chunk_keys:3*chunk_keys]
    keys_w3 = working_keys[3*chunk_keys:]
    
    print(f"  - Worker 0: Assigned keys = {[api_keys.index(k)+1 for k in keys_w0]}")
    print(f"  - Worker 1: Assigned keys = {[api_keys.index(k)+1 for k in keys_w1]}")
    print(f"  - Worker 2: Assigned keys = {[api_keys.index(k)+1 for k in keys_w2]}")
    print(f"  - Worker 3: Assigned keys = {[api_keys.index(k)+1 for k in keys_w3]}")
    
    chunk_size = len(pending_tasks) // num_workers
    tasks_w0 = pending_tasks[:chunk_size]
    tasks_w1 = pending_tasks[chunk_size:2*chunk_size]
    tasks_w2 = pending_tasks[2*chunk_size:3*chunk_size]
    tasks_w3 = pending_tasks[3*chunk_size:]
    
    file_lock = multiprocessing.Lock()
    
    p0 = multiprocessing.Process(
        target=worker_process, 
        args=(0, keys_w0, tasks_w0, file_lock)
    )
    p1 = multiprocessing.Process(
        target=worker_process, 
        args=(1, keys_w1, tasks_w1, file_lock)
    )
    p2 = multiprocessing.Process(
        target=worker_process, 
        args=(2, keys_w2, tasks_w2, file_lock)
    )
    p3 = multiprocessing.Process(
        target=worker_process, 
        args=(3, keys_w3, tasks_w3, file_lock)
    )
    
    print("Launching parallel worker processes (Exactly 4 workers)...")
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    
    print("Unified Parallel Run Complete.")

if __name__ == "__main__":
    main()
