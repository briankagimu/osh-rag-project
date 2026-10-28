
# OSH-RAG: Uganda OSH Compliance Assistant

OSH-RAG is an AI-assisted decision support tool designed for Uganda occupational safety and health (OSH) inspectors. It retrieves relevant sections of Ugandan OSH legal texts and generates answers with exact, verifiable legal citations.

> **Disclaimer:** This is not legal advice. Always verify the cited provision in the Act. The system is designed to refuse to answer when retrieval evidence is weak to prevent harmful hallucinations.

## 📖 Project Context

Uganda has roughly 230 labour inspectors for about 10.2 million employed persons. Inspectors must manually cross-reference unstructured PDFs, which is slow, inconsistent, and hard to audit. OSH-RAG addresses this by using a safety-first Retrieval-Augmented Generation (RAG) pipeline.

Built as a Final Project for the **AI and Advanced Large Language Models** course (Group 17, Beihang University / RCSSTEAP).

* **Group Leader:** Kagimu Brian Vicent
* **Member:** Alberto Elias Araújo Tomo

## ⚙️ Architecture

The pipeline prioritizes traceability and safety over fluent but ungrounded generation:

1. **Ingestion & Chunking:** `PyMuPDF` extracts text. The OSH Act is chunked hierarchically (Part, Section, Clause), while subsidiary rules are chunked at the page level.
2. **Hybrid Retrieval:** Combines Dense retrieval (`BAAI/bge-small-en-v1.5` via ChromaDB) and Sparse retrieval (BM25).
3. **Fusion & Re-ranking:** Reciprocal Rank Fusion (RRF, k=60) merges the lists. A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks the top 20 down to the top 5.
4. **External Abstention Gate:** If the top re-ranker score is **< -2.0**, the system returns a fixed refusal *without* calling the LLM (saving cost and preventing hallucinations).
5. **Generation:** `GPT-4o-mini` (temperature 0) generates an answer strictly from the top 5 passages, appending `[n]` citations.
6. **Interface:** A Streamlit app displaying the answer, expandable source texts, latency, and API cost.

## 📊 Corpus

The current index contains **191 passages** from three instruments:

* **OSH Act 2006** (129 passages)
* **Factories Act SI 220-8** (30 passages)
* **Employment (Migrant Workers) Regulations 2021** (32 passages)

*Note: The Workplace Registration SI 87 of 2014 was excluded because it is an image-only PDF scan with no text layer (OCR is planned for future work).*

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/briankagimu/osh-rag-project.git
   cd osh-rag-project
   ```
