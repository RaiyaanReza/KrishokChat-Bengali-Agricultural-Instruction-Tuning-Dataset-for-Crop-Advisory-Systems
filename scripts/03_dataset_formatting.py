"""
AgriBot-BD: ChatML Conversion & Traceback Ledger Setup Pipeline
==============================================================
Author: Antigravity AI Pair Programmer
Purpose:
  1. Converts train.json and test.json datasets into Gemma-compatible ChatML format.
  2. Embeds tracing parameters (node_id, crop, style) directly in the test.jsonl lines.
  3. Generates a trace traceback ledger (test_trace.json) for high-precision dosage and chemical error audits.
  4. Places outputs in the dedicated `AgriML_52K_ChatML` directory.
"""

import os
import json

# Setup absolute file paths
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "AgriML_52K_ChatML")

TRAIN_INPUT_PATH = os.path.join(DATASET_DIR, "train.json")
TEST_INPUT_PATH = os.path.join(DATASET_DIR, "test.json")

TRAIN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")
TEST_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test.jsonl")
TRACE_LEDGER_PATH = os.path.join(OUTPUT_DIR, "test_trace.json")

SYSTEM_PROMPT = (
    "আপনি একজন অভিজ্ঞ বাঙালি কৃষি সম্প্রসারণ কর্মকর্তা এবং কৃষি এআই উপদেষ্টা। "
    "ব্যবহারকারীর প্রশ্নের জবাবে অত্যন্ত বিনম্র ভাষায় সঠিক সমাধান (রাসায়নিক বালাইনাশক ও সুনির্দিষ্ট ডোজ অথবা "
    "জৈব প্রতিরোধ ব্যবস্থা) প্রদান করুন। উত্তর প্রদানের ক্ষেত্রে অবশ্যই ইংরেজি ফরম্যাটে "
    "বৈজ্ঞানিক মূল উৎস ও সাইটেশন যুক্ত করা বাধ্যতামূলক, যেমন:\n"
    "Source: <source_name> | DOI: <doi_link> | Citation: <citation_details>"
)

def convert_to_chatml():
    print("--------------------------------------------------")
    print("Starting ChatML Conversion & Trace Pipeline...")
    print("--------------------------------------------------")

    # 1. Ensure target output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target directory verified: {OUTPUT_DIR}")

    # 2. Convert Training Split
    print(f"Reading training split from: {TRAIN_INPUT_PATH}...")
    with open(TRAIN_INPUT_PATH, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} training instances. Writing ChatML to {TRAIN_OUTPUT_PATH}...")
    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(train_data, 1):
            chatml_item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ]
            }
            f_out.write(json.dumps(chatml_item, ensure_ascii=False) + "\n")
            
    print(f"Successfully wrote {len(train_data)} ChatML training lines.")

    # 3. Convert Testing Split with Traceback Metadata
    print(f"Reading testing split from: {TEST_INPUT_PATH}...")
    with open(TEST_INPUT_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} testing instances. Writing ChatML with traces to {TEST_OUTPUT_PATH}...")
    
    trace_ledger = []
    
    with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(test_data, 1):
            # Format standard ChatML structure
            chatml_item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ],
                "node_id": item["node_id"],
                "crop": item["crop"],
                "style": item["style"]
            }
            f_out.write(json.dumps(chatml_item, ensure_ascii=False) + "\n")
            
            # Setup trace ledger item
            trace_item = {
                "trace_id": idx,
                "node_id": item["node_id"],
                "crop": item["crop"],
                "style": item["style"],
                "question": item["question"],
                "reference_answer": item["answer"]
            }
            trace_ledger.append(trace_item)

    print(f"Successfully wrote {len(test_data)} ChatML testing lines.")

    # 4. Save Tracing Ledger File
    print(f"Writing trace evaluation ledger to: {TRACE_LEDGER_PATH}...")
    with open(TRACE_LEDGER_PATH, "w", encoding="utf-8") as f_trace:
        json.dump(trace_ledger, f_trace, indent=2, ensure_ascii=False)
        
    print(f"Successfully compiled traceback ledger with {len(trace_ledger)} tracked instances.")
    print("--------------------------------------------------")
    print("SUCCESS: ChatML splits and trace ledger successfully generated!")
    print("--------------------------------------------------")

if __name__ == "__main__":
    convert_to_chatml()
