import streamlit as st
import v4
import pandas as pd
import os

st.set_page_config(page_title="Wagon Sequence Generator", layout="wide")

st.title("🚂 Wagon Sequence Generator (AI Model)")
st.caption("Using Transformer (Seq), PPO (Opt), & GNN (Safety)")

# Sidebar for Config
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", r"e:\Internship\code\model_v4.pth")
project_id = st.sidebar.text_input("Project ID", "Prj001")
start_station = st.sidebar.number_input("Start Station ID", min_value=1, value=1)
max_steps = st.sidebar.slider("Max Sequence Steps", 10, 100, 50)

# Load Model
@st.cache_resource
def load_the_model(path):
    return v4.load_trained_model(path)

if st.sidebar.button("Reload Model"):
    st.cache_resource.clear()

try:
    if os.path.exists(model_path):
        agent, vocab_cmd, adj_list = load_the_model(model_path)
        st.sidebar.success("Model Loaded Successfully!")
    else:
        st.sidebar.error(f"Model not found at {model_path}")
        agent = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    agent = None

# Main Interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Generate New Sequence")
    if st.button("🚀 Generate Sequence", type="primary"):
        if agent:
            with st.spinner("AI is thinking..."):
                steps = v4.generate_sequence_inference(agent, start_station, max_steps)
                
            st.session_state['last_seq'] = steps
            st.success(f"Generated {len(steps)} steps!")
        else:
            st.error("Please load a valid model first.")

with col2:
    st.subheader("Generated Output")
    
    if 'last_seq' in st.session_state:
        steps = st.session_state['last_seq']
        
        # Display as Table
        data = [{"Step": i+1, "Command": cmd, "Station": stn} for i, (cmd, stn) in enumerate(steps)]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Simple Viz
        st.caption("Station Sequence Flow")
        st.line_chart(df["Station"])
        
        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "generated_sequence.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.info("Click 'Generate Sequence' to see results.")

# Safety View
st.divider()
st.subheader("Safety Constraints (GNN Knowledge)")
if agent:
    # Visualize top conflicts?
    pass
