#!/usr/bin/env python3
"""Submit vace_prop_addition.json to ComfyUI API and poll until done."""

import json
import time
import sys
import urllib.request
import urllib.error
import uuid

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "/home/laks/projects/video_swap/workflows/vace_prop_addition.json"
MAX_WAIT = 600  # 10 minutes
POLL_INTERVAL = 15


def check_server():
    try:
        with urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def queue_prompt(workflow: dict) -> str:
    payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["prompt_id"]


def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10) as r:
        return json.loads(r.read())


def main():
    with open(WORKFLOW_PATH) as f:
        workflow = json.load(f)

    # Wait for server
    print("Waiting for ComfyUI server...")
    for i in range(30):
        if check_server():
            print(f"Server ready after {i+1} checks.")
            break
        time.sleep(2)
    else:
        print("ERROR: ComfyUI server did not respond within 60 seconds.")
        sys.exit(1)

    # Submit
    print("Submitting workflow...")
    prompt_id = queue_prompt(workflow)
    print(f"Queued with prompt_id: {prompt_id}")

    # Poll
    start = time.time()
    while time.time() - start < MAX_WAIT:
        history = get_history(prompt_id)
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                print("DONE: Workflow completed successfully.")
                # Print output filenames
                outputs = entry.get("outputs", {})
                for node_id, out in outputs.items():
                    if "gifs" in out:
                        for f in out["gifs"]:
                            print(f"Output file: {f}")
                return 0
            if status.get("status_str") in ("error", "failed"):
                print(f"ERROR: Workflow failed. Status: {status}")
                msgs = entry.get("status", {}).get("messages", [])
                for m in msgs:
                    print(f"  {m}")
                return 1
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] Still running...")
        time.sleep(POLL_INTERVAL)

    print(f"TIMEOUT: Workflow did not complete within {MAX_WAIT} seconds.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
