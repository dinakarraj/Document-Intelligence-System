import streamlit as st
FILE_PATH= r"D:\RAG\Attention_Research_paper.pdf"

st.title("Document Intelligence System")


from rag import setup, ingest, answer_query


@st.cache_resource(show_spinner = "Loading models and databases")

def load_pipeline():
    collection, re_ranker = setup()
    ingest(FILE_PATH, collection)

    return collection, re_ranker 

collection, re_ranker = load_pipeline()

# st.success("Pipeline done")
if "load_toast" not in st.session_state:
    st.toast("Pipeline ready", icon="✅", duration=3)
    
    st.session_state.load_toast=True

if "history" not in st.session_state:
    st.session_state.history = []

if "current" not in st.session_state:
    st.session_state.current = None

with st.form("ask_form", clear_on_submit=True, border=False):

    question = st.text_input("Ask your question: ", key="question_container")
    submitted = st.form_submit_button("submit")


    if submitted:
        if not question.strip():
            print("Please try a question first")
        else:
            with st.spinner("\nSearching the Paper.....\n"):
                out  = answer_query(question, collection, re_ranker)

        item ={"question":question, "out":out}

        st.session_state.current = item 

        # if out["status"] == "ok":
        st.session_state.history.append(item)

def new_chat():
    st.session_state.current = None
    st.session_state.question_container = ""


with st.sidebar:
    st.button("➕ New Query", use_container_width=True, on_click=new_chat)
       
    st.header("Chat History")

    if not st.session_state.history:
        st.caption("No questions yet")

    
    
    for i, past in enumerate(reversed(st.session_state.history)):
            icon = "✅" if past["out"]["status"] == "ok" else "⚠️"
            if st.button(f"{icon} {past['question'][:30]}", key=f"hist_{i}", use_container_width=True, help=past["question"]):

                st.session_state.current=past
                
             
current = st.session_state.current

if current:
    st.subheader(current["question"])
    out = current["out"]
    if out["status"] == "ok":
        st.write(out["answer"])
        st.caption(f"Sources: pages {out['pages']}")
    elif out["status"] == "not_in_document":
        st.warning("This question can't be answered from the document.")
    else:
        st.error("The model is busy right now, please try again in a minute.")


