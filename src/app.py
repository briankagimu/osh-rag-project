import streamlit as st
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from pipeline import answer, get_retriever

st.set_page_config(page_title='OSH-RAG', layout='wide')
st.title('OSH-RAG: Uganda OSH compliance assistant')
st.caption('AI-assisted decision support. Not legal advice. Always verify the cited provision in the Act.')

@st.cache_resource
def load():
    return get_retriever()

R = load()

SAMPLES = [
    'what are the employer duties regarding protective equipment?',
    'who appoints safety representatives?',
    'what are the rules for fencing machinery?',
    'how should accidents be reported?',
    'what is the penalty for obstructing an inspector?'
]

pick = st.sidebar.selectbox('Sample questions', [''] + SAMPLES)
q = st.text_input('Ask an OSH compliance question', value=pick)

if st.button('Ask') and q.strip():
    try:
        with st.spinner('Searching the Act...'):
            out = answer(q, retriever=R)
    except Exception as e:
        st.error(f'The answer service could not be reached: {e}')
        st.stop()
        
    if out['abstained']:
        st.warning(out['answer'])
    else:
        st.success(out['answer'])
        st.caption(f"Latency: {out['latency_s']}s | Cost: ${out['cost_usd']:.5f}")
        
    st.subheader('Sources retrieved')
    for n, cid in enumerate(out['ids'], 1):
        c = R.by_id[cid]
        with st.expander(f"[{n}] {c['citation']} | {c['section_title']}"):
            st.write(c['text'])