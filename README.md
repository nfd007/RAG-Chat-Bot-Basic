# RAG Chatbot Training - fully offline

A Retrieval-Augmented Generation chatbot backend you can read line by line. Everything
runs on your laptop: **Ollama** does embeddings and generation, **ChromaDB** stores the
vectors on disk. No API keys, no LangChain/LlamaIndex, nothing leaves the machine.

```
 data/*.md ──load──> Documents ──chunk──> Chunks ──embed──> vectors ──store──> ChromaDB
                                                                                  │
 question ──embed──> query vector ──────────── nearest chunks (top-k, threshold) <┘
                                                     │
                     numbered context + question ──> llama3.2:3b ──> answer with [1][2] citations
```

## Prerequisites

| What | Version | Check |
|---|---|---|
| Python | 3.10 or newer | `python --version` |
| Ollama | any recent | `ollama --version` - install from <https://ollama.com/download> |
| Disk | ~3 GB | for the two models |
| RAM | 8 GB minimum | llama3.2:3b needs ~3 GB while answering |
| VS Code (optional) | | Python + Python Debugger extensions (recommended on open) |

## Setup

Run from the project root (`rag-chatbot-training/`).

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
ollama pull nomic-embed-text
ollama pull llama3.2:3b
python steps/step_00_check_setup.py
```
If `Activate.ps1` is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
ollama pull nomic-embed-text
ollama pull llama3.2:3b
python steps/step_00_check_setup.py
```

> Do the `pip install` and `ollama pull` steps **before** the session - together they
> download about 2.5 GB. After that, no network is needed.

In VS Code, open the `rag-chatbot-training` folder itself (not its parent) so
`.vscode/launch.json` is picked up. Every step is then one entry in Run and Debug (F5).
On macOS/Linux, change `python.defaultInterpreterPath` in `.vscode/settings.json` to
`${workspaceFolder}/.venv/bin/python`.

## The steps

Each script runs standalone from the project root and prints the command for the next step.

| Step | Command | Needs Ollama | What you learn |
|---|---|---|---|
| 00 | `python steps/step_00_check_setup.py` | checks it | Fail fast: packages, Ollama, both models, with exact fixes |
| 01 | `python steps/step_01_load.py` | no | Look at extracted text before trusting it |
| 02 | `python steps/step_02_chunk.py 800 120` | no | Chunk size and overlap trade-offs |
| 03 | `python steps/step_03_embed.py` | yes | A vector is a list of floats; meaning = direction |
| 04 | `python steps/step_04_index.py` | yes | load -> chunk -> embed -> store, done ahead of time |
| 05 | `python steps/step_05_search.py "your question"` | yes | Debug retrieval before prompting; scores and threshold |
| 06 | `python steps/step_06_answer.py "your question" --show-prompt` | yes | The prompt is the whole trick; citations; refusal |
| 07 | `python steps/step_07_chat.py` | yes | Follow-up rewriting for multi-turn chat |
| eval | `python evaluation/evaluate.py` | yes | Measure retrieval, answers and refusals separately |

Library code lives in `rag/`, one module per concept, in pipeline order:
`config -> loader -> chunker -> embeddings -> store -> retriever -> generator`.

## Running the UI

A Streamlit chat front end (`app.py`) sits on top of the same `rag/` package. It adds no
RAG logic of its own, so it behaves exactly like steps 05-06, but you can watch it live.

```powershell
python steps/step_04_index.py     # once, if you haven't indexed yet
streamlit run app.py
```

Then open **http://localhost:8501**. The browser does not open by itself, because
`.streamlit/config.toml` runs headless and turns off Streamlit's usage statistics so
nothing leaves the machine. In VS Code you can use the **Streamlit UI** launch
configuration (F5) or the **Run: Streamlit UI** task instead. Stop the server with Ctrl+C.

| Sidebar control | What it does |
|---|---|
| Status | Ollama and both models checked on every interaction; shows the exact `ollama pull` fix if one is missing |
| Chunks indexed | Size of the vector store. At 0 the chat box is disabled |
| Top-k / Similarity threshold | Override `TOP_K` / `MIN_SCORE` for the **next** question, no restart needed |
| Show retrieved passages | Expander above each answer with passage number, source, score and text |
| Chat model | Any chat model Ollama has pulled. `ollama pull llama3.1:8b` and it appears here |
| Clear conversation | Empties the chat history |
| Rebuild index from data/ | Re-runs load -> chunk -> embed -> store. Drop a new file into `data/`, click, ask about it |

Live demos for the UI:
- Set the threshold to 0.60 and ask "How long is paternity leave?". It refuses without
  calling the LLM. Lower it step by step until it answers, and compare with the scores shown.
- Set top-k to 1, then 10, for "Who approves a refund of USD 500?" and compare the passages.
- Write a new `.md` file into `data/`, click **Rebuild index**, and ask about it straight away.

The UI is single-turn: each question is searched on its own. For follow-up rewriting
("and for P2?"), use `steps/step_07_chat.py`.

## Live exercises

1. **Chunk size, three ways (step 02).** Run `300 0`, `800 120`, `2000 200`.
   With 300/0, find the chunk that is just `## Context` - an orphaned heading, separated
   from the text it introduces. With 2000/200, three of the four documents become a
   single chunk. Which would you rather retrieve for "How long is paternity leave?"
2. **No shared keywords (step 03).** Add your own sentence pair to `SENTENCES` that means
   the same thing in different words. Does it score higher than an unrelated pair?
3. **Retrieval before generation (step 05).** Run the demo questions and note the top score
   of the Wi-Fi question versus the others. Is `MIN_SCORE=0.25` doing anything for these
   documents? Pick a better value and justify it from the numbers.
4. **Read the prompt (step 06).** `--show-prompt` on "Who approves a USD 500 refund?".
   Which passage number holds the answer? Does the citation match?
5. **Make it refuse (step 06).** Ask "Who won the 2022 World Cup?". The model knows the
   answer - does it still refuse? Now delete rule 3 from `SYSTEM_PROMPT` in
   `rag/generator.py` and ask again.
6. **Follow-ups (step 07).** Ask about P1 first response time, then "and for P2?". Watch the
   `(searching for: ...)` line. Then comment out the `rewrite_question` call and retry.
7. **Tune and measure (eval).** Change one knob in `.env`, re-run `step_04_index.py`, then
   `evaluate.py`. Record the three numbers before and after. One knob at a time.

## Tuning knobs

All in `.env`. Re-run `step_04_index.py` after changing anything marked *re-index*.

| Knob | Default | Raise it when... | Lower it when... | Re-index? |
|---|---|---|---|---|
| `CHUNK_SIZE` | 800 | answers need more surrounding context | chunks mix several topics; retrieval is imprecise | yes |
| `CHUNK_OVERLAP` | 120 | facts get split across chunk boundaries | the index has too much duplicated text | yes |
| `TOP_K` | 4 | the right chunk is ranked 5th-8th (see step 05) | irrelevant chunks distract the small model | no |
| `MIN_SCORE` | 0.25 | off-topic questions still get answered | real questions get refused (check step 05 scores) | no |
| `TEMPERATURE` | 0.1 | (rarely, for RAG) | answers paraphrase numbers or invent details | no |
| `EMBEDDING_MODEL` | nomic-embed-text | - | - | yes, and the vector dimension changes |
| `CHAT_MODEL` | llama3.2:3b | a bigger model follows instructions better (e.g. `llama3.1:8b`) | answers are too slow | no |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot reach Ollama at http://localhost:11434` / `ConnectionError` | Ollama server not running | Start the Ollama app, or run `ollama serve` in another terminal. Check `OLLAMA_HOST` in `.env`. |
| `model 'nomic-embed-text' is NOT installed` / `model "llama3.2:3b" not found` | Model not pulled, or name in `.env` has a typo | `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b`; `ollama list` shows exact names |
| `The vector store is empty. Run: python steps/step_04_index.py` | Steps 05-07 ran before indexing, or `CHROMA_DIR`/`COLLECTION_NAME` changed | `python steps/step_04_index.py` |
| `ModuleNotFoundError: No module named 'rag'` | Script run from inside `steps/`, or a copied script lost its `sys.path` lines | Run from the project root: `python steps/step_05_search.py`. In VS Code, use the provided launch configs. |
| `ModuleNotFoundError: No module named 'chromadb'` (or `ollama`, `dotenv`) | venv not activated / wrong interpreter | Activate `.venv`, or in VS Code: *Python: Select Interpreter* -> `.venv` |
| `Collection expecting embedding with dimension of 768, got 1024` or `was built with embedding model ...` | `EMBEDDING_MODEL` changed but old vectors are still stored | `python steps/step_04_index.py` (it resets the collection). If in doubt, delete the `chroma_db/` folder. |
| Every answer is "I don't have that in the knowledge base." | `MIN_SCORE` too high, or the index is stale | Check scores with step 05; re-index with step 04 |
| Answers without `[n]` citations | Small models sometimes ignore format rules | Lower `TEMPERATURE`, reduce `TOP_K`, or try a larger `CHAT_MODEL` |
| First answer takes 20+ seconds | Ollama is loading the model into memory | Normal on the first call; later calls are fast |
| `streamlit : The term 'streamlit' is not recognized` | venv not activated | Activate `.venv`, or run `python -m streamlit run app.py` |
| `Port 8501 is already in use` | An earlier `streamlit run` is still running | Stop it with Ctrl+C in its terminal, or `streamlit run app.py --server.port 8502` |

## Project layout

```
rag-chatbot-training/
├── .vscode/       launch configs (one per step), tasks, settings, recommended extensions
├── data/          4 sample documents: Odoo inventory guide, support playbook, HR leave policy, ADR
├── rag/           the library: config, loader, chunker, embeddings, store, retriever, generator
├── steps/         step_00 ... step_07 walkthrough scripts
├── evaluation/    questions.json (8 answerable + 2 unanswerable) and evaluate.py
├── app.py         Streamlit chat UI (imports rag/, adds no RAG logic)
├── .streamlit/    config.toml: headless, usage statistics off
├── .env.example   copy to .env
└── requirements.txt
```
