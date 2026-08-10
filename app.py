"""
Medical AI Assistant - Web Application Server
============================================
Provides a modern, glassmorphic Web UI for the MedQuAD Medical RAG System.
Uses Python's built-in http.server to run out-of-the-box on http://localhost:5000
"""

import json
import os
import sys
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from medical_rag_pipeline import MedicalAIAssistantRAG

# Global RAG Instance
RAG_SYSTEM = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical AI Assistant | Clinical Decision Support</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --bg-card: rgba(18, 26, 43, 0.7);
            --bg-card-hover: rgba(26, 38, 63, 0.8);
            --border-glass: rgba(255, 255, 255, 0.08);
            --border-accent: rgba(45, 212, 191, 0.3);
            --accent-teal: #2dd4bf;
            --accent-cyan: #38bdf8;
            --accent-blue: #6366f1;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --shadow-glow: 0 0 25px rgba(45, 212, 191, 0.15);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 15% 20%, rgba(45, 212, 191, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 75%, rgba(99, 102, 241, 0.08) 0%, transparent 40%);
        }

        /* Header / Navbar */
        header {
            padding: 1.25rem 2rem;
            background: rgba(9, 13, 22, 0.8);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-glass);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-teal), var(--accent-cyan));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(45, 212, 191, 0.3);
        }

        .brand-icon svg {
            width: 22px;
            height: 22px;
            fill: #090d16;
        }

        .brand-text h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .brand-text p {
            font-size: 0.75rem;
            color: var(--accent-teal);
            font-weight: 500;
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(45, 212, 191, 0.1);
            border: 1px solid rgba(45, 212, 191, 0.2);
            padding: 0.4rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            color: var(--accent-teal);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-teal);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--accent-teal);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }

        /* Main App Layout */
        main {
            flex: 1;
            max-width: 1300px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 2rem;
        }

        @media (max-width: 992px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        /* Chat & Query Panel */
        .query-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-glass);
            border-radius: 20px;
            padding: 1.75rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .card-header {
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.2rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-subtitle {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        /* Quick Prompts */
        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 1rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
            padding: 0.5rem 0.9rem;
            border-radius: 12px;
            font-size: 0.825rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .chip:hover {
            background: rgba(45, 212, 191, 0.12);
            border-color: var(--border-accent);
            color: var(--text-primary);
            transform: translateY(-1px);
        }

        /* Input Form */
        .input-group {
            position: relative;
            display: flex;
            gap: 0.75rem;
        }

        textarea {
            width: 100%;
            min-height: 100px;
            background: rgba(9, 13, 22, 0.6);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s ease;
        }

        textarea:focus {
            border-color: var(--accent-teal);
            box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.15);
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--accent-teal), var(--accent-cyan));
            color: #090d16;
            font-weight: 600;
            font-size: 0.95rem;
            border: none;
            border-radius: 14px;
            padding: 0 1.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            align-self: flex-end;
            height: 48px;
        }

        .btn-submit:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(45, 212, 191, 0.35);
        }

        .btn-submit:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        /* Answer Output Section */
        .response-box {
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .answer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-glass);
        }

        .latency-badge {
            font-size: 0.75rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            padding: 0.25rem 0.6rem;
            border-radius: 8px;
            font-weight: 500;
        }

        .answer-content {
            font-size: 1rem;
            line-height: 1.7;
            color: #e2e8f0;
            white-space: pre-line;
        }

        /* Pipeline Visualizer (Sidebar) */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .stat-card {
            background: rgba(9, 13, 22, 0.5);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
        }

        .stat-val {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent-teal);
        }

        .stat-lbl {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.2rem;
        }

        /* Context Cards */
        .context-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-height: 450px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }

        .context-list::-webkit-scrollbar {
            width: 6px;
        }

        .context-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }

        .context-item {
            background: rgba(9, 13, 22, 0.5);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.85rem;
            transition: border-color 0.2s ease;
        }

        .context-item:hover {
            border-color: var(--border-accent);
        }

        .context-tag {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }

        .context-text {
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* Loader */
        .loader {
            display: none;
            text-align: center;
            padding: 2rem 0;
        }

        .spinner {
            width: 36px;
            height: 36px;
            border: 3px solid rgba(45, 212, 191, 0.15);
            border-top-color: var(--accent-teal);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-icon">
                <svg viewBox="0 0 24 24"><path d="M19 10.5h-5.5V5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v5.5H5c-.83 0-1.5.67-1.5 1.5s.67 1.5 1.5 1.5h5.5V19c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-5.5H19c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5z"/></svg>
            </div>
            <div class="brand-text">
                <h1>Medical AI Assistant</h1>
                <p>Clinical Decision Support System (MedQuAD RAG)</p>
            </div>
        </div>
        <div class="nav-status">
            <div class="status-dot"></div>
            <span>PubMedBERT + FAISS + GPT-2</span>
        </div>
    </header>

    <main>
        <section class="query-section">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Ask a Medical Question</h2>
                    <p class="card-subtitle">Answers are retrieved from verified NIH medical databases and synthesized using PubMedBERT + GPT-2.</p>
                </div>

                <div class="chip-container">
                    <button class="chip" onclick="setQuery('What are the symptoms of Glaucoma?')">👁️ Glaucoma Symptoms</button>
                    <button class="chip" onclick="setQuery('What causes Diabetes?')">🩸 Causes of Diabetes</button>
                    <button class="chip" onclick="setQuery('What are the treatments for High Blood Pressure?')">🫀 Hypertension Treatment</button>
                    <button class="chip" onclick="setQuery('What is Parkinson\'s Disease?')">🧠 Parkinson's Info</button>
                </div>

                <form id="query-form" onsubmit="handleQuery(event)">
                    <div class="input-group">
                        <textarea id="user-query" placeholder="e.g. What are the early signs and treatment options for Glaucoma?" required></textarea>
                    </div>
                    <button type="submit" id="btn-submit" class="btn-submit" style="margin-top: 1rem;">
                        <span>Generate Answer</span>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>
            </div>

            <div id="loader" class="loader card">
                <div class="spinner"></div>
                <p style="font-weight: 500;">Searching PubMedBERT Vector Database...</p>
                <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem;">Retrieving & Re-Ranking Top NIH Medical Contexts</p>
            </div>

            <div id="response-box" class="card response-box">
                <div class="answer-header">
                    <h3 class="card-title" style="color: var(--accent-teal);">Grounded Answer</h3>
                    <span id="latency-badge" class="latency-badge">0.0s</span>
                </div>
                <div id="answer-content" class="answer-content"></div>
            </div>
        </section>

        <aside class="sidebar">
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Dataset Metrics</h3>
                    <p class="card-subtitle">MedQuAD National Institutes of Health</p>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-val">16,412</div>
                        <div class="stat-lbl">QA Pairs</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">768d</div>
                        <div class="stat-lbl">PubMedBERT Dim</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">5,126</div>
                        <div class="stat-lbl">Focus Areas</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">FAISS</div>
                        <div class="stat-lbl">Vector Search</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Retrieved NIH Contexts</h3>
                    <p class="card-subtitle">Top Passages Grounding the Generated Answer</p>
                </div>
                <div id="context-list" class="context-list">
                    <p style="font-size: 0.85rem; color: var(--text-muted); text-align: center; padding: 1.5rem 0;">
                        Submit a question to view retrieved PubMedBERT contexts.
                    </p>
                </div>
            </div>
        </aside>
    </main>

    <script>
        function setQuery(text) {
            document.getElementById('user-query').value = text;
        }

        async function handleQuery(e) {
            e.preventDefault();
            const queryInput = document.getElementById('user-query');
            const query = queryInput.value.trim();
            if (!query) return;

            const btn = document.getElementById('btn-submit');
            const loader = document.getElementById('loader');
            const responseBox = document.getElementById('response-box');
            const answerContent = document.getElementById('answer-content');
            const contextList = document.getElementById('context-list');
            const latencyBadge = document.getElementById('latency-badge');

            btn.disabled = true;
            loader.style.display = 'block';
            responseBox.style.display = 'none';

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

                const data = await res.json();
                
                // Set Answer & Latency
                answerContent.textContent = data.generated_answer;
                latencyBadge.textContent = `${data.elapsed_seconds}s`;

                // Set Contexts
                contextList.innerHTML = '';
                data.retrieved_contexts.forEach((ctx, idx) => {
                    const item = document.createElement('div');
                    item.className = 'context-item';
                    item.innerHTML = `
                        <div class="context-tag">Passage #${idx + 1}</div>
                        <div class="context-text">${ctx.replace(/\\n/g, '<br>')}</div>
                    `;
                    contextList.appendChild(item);
                });

                responseBox.style.display = 'block';
            } catch (err) {
                alert("Error querying Medical RAG system: " + err.message);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

class MedicalRequestHandler(BaseHTTPRequestHandler):
    def _send_headers(self, content_type="text/html", code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_headers("text/html")
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self._send_headers("text/plain", 404)
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                query = payload.get("query", "").strip()

                if not query:
                    self._send_headers("application/json", 400)
                    self.wfile.write(json.dumps({"error": "Empty query"}).encode("utf-8"))
                    return

                print(f"[*] Web Query received: '{query}'")
                result = RAG_SYSTEM.answer_question(query)

                self._send_headers("application/json", 200)
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                print(f"[!] Server Error: {e}")
                self._send_headers("application/json", 500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._send_headers("text/plain", 404)
            self.wfile.write(b"404 Not Found")


def run_server(port=5000):
    global RAG_SYSTEM
    print("=" * 65)
    print("      Medical AI Assistant - Web Application Server")
    print("=" * 65)

    csv_path = "medquad.csv"
    if not os.path.exists(csv_path):
        print(f"[!] Please ensure '{csv_path}' is available in the current directory.")
        return

    print("[*] Initializing Medical RAG Engine...")
    RAG_SYSTEM = MedicalAIAssistantRAG(csv_path=csv_path, use_reranker=True)

    server_address = ("", port)
    httpd = HTTPServer(server_address, MedicalRequestHandler)
    print(f"\n[+] Web Application Server is LIVE at: http://localhost:{port}")
    print("[*] Open your browser and navigate to: http://localhost:5000")
    print("[*] Press Ctrl+C to stop the server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Web Server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server(port=5000)
