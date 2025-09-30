#!/usr/bin/env python3
"""
Simple FastAPI server for JSON consensus alignment visualization.
Run with: python3 api_server.py
Then visit: http://localhost:8000
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="JSON Consensus Alignment API")

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

def run_alignment_script(files: List[str]) -> Dict[str, Any]:
    """Run the alignment script and parse the output."""
    try:
        # Build command
        files_str = " ".join(files)
        cmd = f"python3 {SCRIPT_DIR}/analyze_consensus_json.py {files_str}"
        
        # Run the script
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=SCRIPT_DIR,
            capture_output=True, 
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"Script failed: {result.stderr}")
        
        # Parse the output
        lines = result.stdout.split('\n')
        current_section = ''
        current_json = ''
        json_count = 0
        mappings_json = ''
        
        data = {
            'original_jsons': [],
            'aligned_jsons': [],
            'key_mappings': {}
        }
        
        for line in lines:
            if line.startswith('ORIGINAL_JSONS:'):
                current_section = 'original'
                json_count = 0
                continue
            elif line.startswith('ALIGNED_JSONS:'):
                # Parse the last JSON from original section before switching
                if current_json.strip() and current_section == 'original':
                    try:
                        parsed = json.loads(current_json)
                        data['original_jsons'].append(parsed)
                        print(f"Parsed final original JSON {len(data['original_jsons'])}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse final original JSON: {e}")
                current_section = 'aligned'
                json_count = 0
                current_json = ''
                continue
            elif line.startswith('KEY_MAPPINGS:'):
                # Parse the last JSON from aligned section before switching
                if current_json.strip() and current_section == 'aligned':
                    try:
                        parsed = json.loads(current_json)
                        data['aligned_jsons'].append(parsed)
                        print(f"Parsed final aligned JSON {len(data['aligned_jsons'])}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse final aligned JSON: {e}")
                current_section = 'mappings'
                mappings_json = ''
                current_json = ''
                continue
            elif line.startswith('# JSON '):
                if current_json.strip():
                    try:
                        parsed = json.loads(current_json)
                        if current_section == 'original':
                            data['original_jsons'].append(parsed)
                        elif current_section == 'aligned':
                            data['aligned_jsons'].append(parsed)
                        print(f"Parsed {current_section} JSON {json_count + 1}")
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse {current_section} JSON {json_count + 1}: {e}")
                        print(f"JSON content: {current_json[:200]}...")
                current_json = ''
                json_count += 1
                continue
            
            if current_section in ['original', 'aligned']:
                current_json += line + '\n'
            elif current_section == 'mappings':
                mappings_json += line + '\n'
        
        # Parse the last JSON if exists
        if current_json.strip():
            try:
                parsed = json.loads(current_json)
                if current_section == 'original':
                    data['original_jsons'].append(parsed)
                elif current_section == 'aligned':
                    data['aligned_jsons'].append(parsed)
                print(f"Parsed final {current_section} JSON {len(data.get(f'{current_section}_jsons', []))}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse final {current_section} JSON: {e}")
                print(f"JSON content: {current_json[:200]}...")
        
        # Parse key mappings
        if mappings_json.strip():
            try:
                data['key_mappings'] = json.loads(mappings_json)
                print(f"Parsed key mappings with {len(data['key_mappings'])} entries")
            except json.JSONDecodeError as e:
                print(f"Failed to parse key mappings: {e}")
                print(f"Mappings content: {mappings_json[:200]}...")
        
        print(f"Final counts - Original: {len(data['original_jsons'])}, Aligned: {len(data['aligned_jsons'])}, Mappings: {len(data['key_mappings'])}")
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JSON Consensus Alignment</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            button { padding: 10px 20px; font-size: 16px; margin: 10px; }
            .tabs { margin-top: 20px; }
            .tab { display: none; }
            .tab.active { display: block; }
            .tab-buttons { margin-bottom: 20px; }
            .tab-button { padding: 8px 16px; margin-right: 5px; cursor: pointer; border: 1px solid #ccc; background: #f0f0f0; }
            .tab-button.active { background: #007bff; color: white; }
            pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow: auto; max-height: 400px; }
            .json-container { margin-bottom: 20px; }
            .loading { color: #666; font-style: italic; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>JSON Consensus Alignment</h1>
            <button onclick="runAlignment()">Run Alignment on json1-4</button>
            <div id="status"></div>
            
            <div id="results" style="display: none;">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('original')">Original JSONs</button>
                    <button class="tab-button" onclick="showTab('aligned')">Aligned JSONs</button>
                    <button class="tab-button" onclick="showTab('mappings')">Key Mappings</button>
                    <button class="tab-button" onclick="showTab('comparison')">Comparison</button>
                </div>
                
                <div id="original" class="tab active">
                    <h2>Original JSONs</h2>
                    <div id="original-content"></div>
                </div>
                
                <div id="aligned" class="tab">
                    <h2>Aligned JSONs</h2>
                    <div id="aligned-content"></div>
                </div>
                
                <div id="mappings" class="tab">
                    <h2>Key Mappings</h2>
                    <div id="mappings-content" style="max-height: 600px; overflow-y: auto;"></div>
                </div>
                
                <div id="comparison" class="tab">
                    <h2>Before vs After Comparison</h2>
                    <div id="comparison-content"></div>
                </div>
            </div>
        </div>

        <script>
            let currentData = null;
            
            async function runAlignment() {
                const status = document.getElementById('status');
                const results = document.getElementById('results');
                
                status.innerHTML = '<div class="loading">Running alignment...</div>';
                results.style.display = 'none';
                
                try {
                    const response = await fetch('/api/align', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ files: ['json1.json', 'json2.json', 'json3.json', 'json4.json'] })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Alignment failed');
                    }
                    
                    currentData = await response.json();
                    displayResults(currentData);
                    status.innerHTML = '<div style="color: green;">Alignment completed successfully!</div>';
                    results.style.display = 'block';
                    
                } catch (error) {
                    status.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
            
            function displayResults(data) {
                // Original JSONs
                const originalContent = document.getElementById('original-content');
                originalContent.innerHTML = data.original_jsons.map((json, i) => 
                    `<div class="json-container">
                        <h3>JSON ${i + 1}</h3>
                        <pre>${JSON.stringify(json, null, 2)}</pre>
                    </div>`
                ).join('');
                
                // Aligned JSONs
                const alignedContent = document.getElementById('aligned-content');
                alignedContent.innerHTML = data.aligned_jsons.map((json, i) => 
                    `<div class="json-container">
                        <h3>JSON ${i + 1}</h3>
                        <pre>${JSON.stringify(json, null, 2)}</pre>
                    </div>`
                ).join('');
                
                // Key Mappings
                const mappingsContent = document.getElementById('mappings-content');
                const mappingsHtml = Object.entries(data.key_mappings)
                    .slice(0, 50) // Show first 50 mappings
                    .map(([key, value]) => 
                        `<div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-left: 3px solid #007bff;">
                            <strong>${key}:</strong><br>
                            <code>${JSON.stringify(value, null, 2)}</code>
                        </div>`
                    ).join('');
                
                mappingsContent.innerHTML = mappingsHtml + 
                    (Object.keys(data.key_mappings).length > 50 ? 
                        `<div style="color: #666; font-style: italic; margin-top: 10px;">
                            ... and ${Object.keys(data.key_mappings).length - 50} more mappings
                        </div>` : '');
                
                // Comparison
                const comparisonContent = document.getElementById('comparison-content');
                comparisonContent.innerHTML = data.original_jsons.map((original, i) => 
                    `<div class="json-container">
                        <h3>JSON ${i + 1}</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <h4>Original</h4>
                                <pre style="background: #ffe6e6;">${JSON.stringify(original, null, 2)}</pre>
                            </div>
                            <div>
                                <h4>Aligned</h4>
                                <pre style="background: #e6ffe6;">${JSON.stringify(data.aligned_jsons[i], null, 2)}</pre>
                            </div>
                        </div>
                    </div>`
                ).join('');
            }
            
            function showTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                
                // Show selected tab
                document.getElementById(tabName).classList.add('active');
                event.target.classList.add('active');
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/align")
async def align_jsons(request: Dict[str, Any]):
    """API endpoint to run alignment."""
    files = request.get('files', ['json1.json', 'json2.json', 'json3.json', 'json4.json'])
    return run_alignment_script(files)

if __name__ == "__main__":
    print("Starting JSON Consensus Alignment API...")
    print("Visit: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
