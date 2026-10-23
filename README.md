# OSH-RAG: Uganda OSH Compliance Assistant

## What is this?
OSH-RAG is an AI-assisted decision support tool for Uganda occupational safety and health inspectors. It retrieves relevant sections of the OSH Act 2006 and generates answers with exact legal citations.

**Disclaimer:** This is not legal advice. Always verify the cited provision in the Act.

## How to run
1. Activate environment: `conda activate osh_rag`
2. Start the app: `streamlit run src/app.py`

## How to use
1. Type a question or choose from the sample questions.
2. Read the answer. Citations appear as `[1]`, `[2]`.
3. Click the "Sources retrieved" panels to read the exact legal text.
4. If the answer says "I cannot answer this with sufficient confidence", it means the Act does not contain the answer (or the search score was too low).

## Known Limitations
- The corpus only contains the OSH Act 2006 (no subsidiary regulations yet).
- English only.
- Requires internet connection for the OpenAI API.