import streamlit as st
import pandas as pd
import os
import io
import v4  # Ensure this module is available in the same directory

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Wagon Sequence System",
    layout="wide",
    page_icon="🚂"
)

# ------------------ STYLE ------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #111827, #000000);
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #1f2933;
    border-right: 1px solid #2c2f36;
}
[data-testid="stSidebar"] * {
    color: white;
}
.stButton > button {
    background-color: #ff4b4b;
    border-radius: 10px;
    font-weight: bold;
    color: white;
}
/* Adjust tab styling if needed */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #1f2933;
    border-radius: 5px;
    color: white;
    padding: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ------------------ NAVIGATION ------------------
st.sidebar.title("⚙️ Control Panel")
app_mode = st.sidebar.radio("Select Mode", ["Station Editor & Rule-Based Gen", "AI Model Inference"], index=0)

# ------------------ MODE 1: STATION EDITOR (from UI_nupur.py) ------------------
if app_mode == "Station Editor & Rule-Based Gen":
    st.markdown("## 📝 Station Editor & Sequence Generator")
    st.caption("Manually edit tank data and generate sequences using standard rules.")

    # --- Data Loading ---
    CSV_PATH = r"e:\Internship\code\tanks_csv.csv"

    def load_data():
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH)
        return pd.DataFrame({
            "project_id": [1],
            "station_no": [1],
            "process_name": ["Start Station"],
            "critical_status": ["High"],
            "distance_mm": [0],
            "dip_time_sec": [0]
        })

    if "df_tanks" not in st.session_state:
        st.session_state.df_tanks = load_data()

    if "generated_rule_df" not in st.session_state:
        st.session_state.generated_rule_df = None

    # --- Tabs ---
    tab_edit, tab_output = st.tabs(["📝 Station Editor", "📤 Generated Sequence"])

    with tab_edit:
        st.subheader("Step 1: Edit Station / Tank Data")
        
        edited_df = st.data_editor(
            st.session_state.df_tanks,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_station"
        )
        st.session_state.df_tanks = edited_df
        
        st.markdown("###")
        
        if st.button("🚀 Generate Sequence (Rule-Based)", type="primary"):
            with st.spinner("Generating sequence..."):
                try:
                    records = edited_df.astype(str).to_dict(orient="records")
                    seq_data = v4.generate_sequence_data(records)

                    if seq_data and hasattr(seq_data, '__iter__') and len(seq_data) > 0 and seq_data[0][0] != "Error":
                        headers = seq_data[0]
                        rows = seq_data[1:]
                        st.session_state.generated_rule_df = pd.DataFrame(rows, columns=headers)
                        st.success("Sequence generated successfully!")
                    else:
                        error_msg = seq_data[0][1] if seq_data and len(seq_data) > 0 else "Unknown error returned."
                        st.error(f"Error: {error_msg}")
                except Exception as e:
                    st.error(f"An exception occurred: {e}")

    with tab_output:
        if st.session_state.generated_rule_df is None:
            st.info("Generate a sequence from the Station Editor tab.")
        else:
            st.success("Generated Sequence")
            st.dataframe(
                st.session_state.generated_rule_df,
                use_container_width=True,
                height=420
            )

            csv_buffer = io.StringIO()
            st.session_state.generated_rule_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download Sequence CSV",
                data=csv_buffer.getvalue(),
                file_name="generated_sequence_rule.csv",
                mime="text/csv"
            )

    # --- Sidebar Stats ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Stats")
    st.sidebar.markdown(f"**Total Stations:** {len(st.session_state.df_tanks)}")
    st.sidebar.markdown(f"**Critical Stations:** {(st.session_state.df_tanks.get('critical_status') == 'High').sum()}")
    st.sidebar.markdown(f"**Total Dip Time:** {int(pd.to_numeric(st.session_state.df_tanks.get('dip_time_sec', 0), errors='coerce').sum())} sec")


# ------------------ MODE 2: AI INFERENCE (from ui.py) ------------------
elif app_mode == "AI Model Inference":
    st.markdown("## 🤖 AI Model Inference")
    st.caption("Generate sequences using Transformer (Seq), PPO (Opt), & GNN (Safety) models.")
    
    # --- Sidebar Configuration ---
    st.sidebar.markdown("---")
    st.sidebar.header("Model Config")
    model_path = st.sidebar.text_input("Model Path", r"e:\Internship\code\model_v4.pth")
    project_id = st.sidebar.text_input("Project ID", "Prj001")
    start_station = st.sidebar.number_input("Start Station ID", min_value=1, value=1)
    max_steps = st.sidebar.slider("Max Sequence Steps", 10, 100, 50)
    
    # --- Load Model ---
    @st.cache_resource
    def load_the_model(path):
        return v4.load_trained_model(path)

    if st.sidebar.button("Reload Model"):
        st.cache_resource.clear()

    # Initialize model
    agent = None
    try:
        if os.path.exists(model_path):
            agent, vocab_cmd, adj_list = load_the_model(model_path)
            st.sidebar.success("Model Loaded Successfully!")
        else:
            st.sidebar.warning(f"Model not found at {model_path}")
    except Exception as e:
        # Don't show error immediately on load unless explicit
        if st.button("Check Model Path"):
             st.sidebar.error(f"Error loading model: {e}")

    # --- Main Interface ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Control")
        if st.button("🚀 Generate Sequence (AI)", type="primary"):
            if agent:
                with st.spinner("AI is thinking..."):
                    try:
                        steps = v4.generate_sequence_inference(agent, start_station, max_steps)
                        st.session_state['last_ai_seq'] = steps
                        st.success(f"Generated {len(steps)} steps!")
                    except Exception as e:
                        st.error(f"Inference Error: {e}")
            else:
                st.error("Please ensure the model is loaded first.")

    with col2:
        st.subheader("Generated Output")
        
        if 'last_ai_seq' in st.session_state:
            steps = st.session_state['last_ai_seq']
            
            # Display as Table
            try:
                data = [{"Step": i+1, "Command": cmd, "Station": stn} for i, (cmd, stn) in enumerate(steps)]
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Simple Viz
                st.caption("Station Sequence Flow (AI)")
                if "Station" in df.columns:
                     st.line_chart(df["Station"])
                
                # Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    "generated_ai_sequence.csv",
                    "text/csv",
                    key='download-ai-csv'
                )
            except Exception as e:
                st.error(f"Error formatting output: {e}")
                st.write("Raw Output:", steps)
        else:
            st.info("Click 'Generate Sequence (AI)' to see results.")

        # --- Rule-Based Visualization (Cross-Reference) ---
        if st.session_state.get('generated_rule_df') is not None:
            st.divider()
            st.subheader("Comparison: Rule-Based Sequence")
            st.caption("Station Sequence Flow (From Station Editor)")
            
            try:
                graph_data = st.session_state.generated_rule_df.copy()
                plot_vals = []
                last_vis_station = 0
                
                for _, row in graph_data.iterrows():
                    cmd = str(row.get('Command', ''))
                    val = row.get('Value', 0)
                    
                    if "Wait" in cmd:
                        plot_vals.append(last_vis_station)
                    else:
                        try:
                            curr_stn = float(val)
                            last_vis_station = curr_stn
                            plot_vals.append(curr_stn)
                        except:
                            plot_vals.append(last_vis_station)
                            
                st.line_chart(plot_vals)
            except Exception as e:
                st.warning(f"Could not render rule-based graph: {e}")

    # Safety View
    st.divider()

    st.subheader("Safety Constraints (GNN Knowledge)")
    st.info("Safety constraints visualization will appear here if implemented.")
