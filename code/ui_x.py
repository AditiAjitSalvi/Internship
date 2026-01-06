import streamlit as st
import v4
import pandas as pd
import os

st.set_page_config(page_title="Wagon Sequence Generator", layout="wide")

st.title("🚂 Wagon Sequence Generator (AI Model)")
st.caption("Transformer-based Auto Sequencing with Safety Awareness")

# ================= SIDEBAR =================
st.sidebar.header("Configuration")

model_path = st.sidebar.text_input(
    "Model Path",
    r"e:\Internship\code\model_v4.pth"
)

project_id = st.sidebar.number_input(
    "Project Seq ID",
    min_value=1,
    value=1,
    step=1
)

start_station = st.sidebar.number_input(
    "Start Station ID",
    min_value=1,
    value=1
)

max_steps = st.sidebar.slider(
    "Max Sequence Steps",
    10, 100, 50
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_the_model(path):
    return v4.load_trained_model(path)

if st.sidebar.button("🔄 Reload Model"):
    st.cache_resource.clear()

agent = None
try:
    if os.path.exists(model_path):
        agent, vocab_cmd, adj_list = load_the_model(model_path)
        st.sidebar.success("✅ Model Loaded")
    else:
        st.sidebar.error("❌ Model path not found")
except Exception as e:
    st.sidebar.error(str(e))

# ================= MAIN UI =================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Generate New Sequence")

    if st.button("🚀 Generate Sequence", type="primary"):
        if agent:
            with st.spinner("AI is generating sequence..."):
                steps = v4.generate_sequence_inference(
                    agent=agent,
                    project_id=project_id,
                    start_station=start_station,
                    max_steps=max_steps
                )

            st.session_state["last_seq"] = steps
            st.success(f"Generated {len(steps)} steps")
        else:
            st.error("Load model first")

with col2:
    st.subheader("Generated Output")

    if "last_seq" in st.session_state:
        steps = st.session_state["last_seq"]

        df = pd.DataFrame(
            steps,
            columns=[
                "command",
                "station_no",
                "wait_sec",
                "critical_status"
            ]
        )

        df.insert(0, "step_no", range(1, len(df) + 1))
        df.insert(1, "project_id", project_id)

        st.dataframe(df, use_container_width=True)

        st.caption("📈 Station Flow")
        st.line_chart(df["station_no"])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            csv,
            "generated_sequence.csv",
            "text/csv"
        )
    else:
        st.info("Click **Generate Sequence** to see output")

# ================= SAFETY =================
st.divider()
st.subheader("🔒 Safety Constraints (GNN Knowledge)")

if agent:
    st.caption("Safety graph & conflict detection integrated during inference")
