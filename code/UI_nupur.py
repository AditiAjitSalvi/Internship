import streamlit as st
import pandas as pd
import os
import v4
import io

st.set_page_config(
    page_title="Wagon Sequence Editor",
    layout="wide",
    page_icon="🛠️"
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
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("## 🛠️ Wagon Sequence Editor")
st.caption("Industrial Station Data Editor & Sequence Generator")

# ------------------ DATA ------------------
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

if "generated_df" not in st.session_state:
    st.session_state.generated_df = None

# ------------------ TABS ------------------
tab_edit, tab_output = st.tabs(["📝 Station Editor", "📤 Generated Sequence"])

# ------------------ TAB 1 ------------------
with tab_edit:
    st.subheader("Step 1 : Edit Station / Tank Data")

    edited_df = st.data_editor(
        st.session_state.df_tanks,
        num_rows="dynamic",
        use_container_width=True
    )
    st.session_state.df_tanks = edited_df

    st.markdown("###")

    generate_btn = st.button(
        "🚀 Generate Sequence",
        type="primary",
        use_container_width=False
    )

    if generate_btn:
        with st.spinner("Generating sequence..."):
            records = edited_df.astype(str).to_dict(orient="records")
            seq_data = v4.generate_sequence_data(records)

        if seq_data and seq_data[0][0] != "Error":
            headers = seq_data[0]
            rows = seq_data[1:]
            st.session_state.generated_df = pd.DataFrame(rows, columns=headers)
            st.success("Sequence generated. Open **Generated Sequence** tab.")
        else:
            st.error(seq_data[0][1])

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Live station analysis")

    st.markdown("### Total Stations")
    st.markdown(f"## {len(edited_df)}")

    st.markdown("### Critical Stations")
    st.markdown(f"## {(edited_df['critical_status'] == 'High').sum()}")

    st.markdown("### Total Dip Time (sec)")
    st.markdown(f"## {int(pd.to_numeric(edited_df['dip_time_sec'], errors='coerce').sum())}")

# ------------------ TAB 2 ------------------
with tab_output:
    if st.session_state.generated_df is None:
        st.info("Generate a sequence from the Station Editor tab.")
    else:
        st.success("Generated Sequence")
        st.dataframe(
            st.session_state.generated_df,
            use_container_width=True,
            height=420
        )

        csv_buffer = io.StringIO()
        st.session_state.generated_df.to_csv(csv_buffer, index=False)

        st.download_button(
            "⬇️ Download Sequence CSV",
            data=csv_buffer.getvalue(),
            file_name="generated_sequence.csv",
            mime="text/csv"
        )