import streamlit as st
import pandas as pd
import os
import v4
import io

st.set_page_config(page_title="Wagon Sequence Editor", layout="wide")

st.title("🛠️ Wagon Sequence Editor")
st.caption("Manually edit station data and generate operational sequences.")

# File Path for Persistence
CSV_PATH = r"e:\Internship\code\tanks_csv.csv"

# 1. Load Data
def load_data():
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame(columns=["project_id", "station_no", "process_name", "critical_status", "distance_mm", "dip_time_sec"])
    else:
        # Default Template
        return pd.DataFrame({
            "project_id": [1],
            "station_no": [1],
            "process_name": ["Start Station"],
            "critical_status": ["High"],
            "distance_mm": [0],
            "dip_time_sec": [0]
        })

# Initialize Session State
if "df_tanks" not in st.session_state:
    st.session_state["df_tanks"] = load_data()

# 2. Data Editor
st.subheader("1. Edit Station Data")
edited_df = st.data_editor(
    st.session_state["df_tanks"],
    num_rows="dynamic",
    use_container_width=True,
    key="tank_editor"
)

# 3. Save Button (Optional explicit save, but we use the dataframe for generation)
# st.button("Save Changes to Disk") ... implementing auto-use for generation

# 4. Generate Button
st.divider()
st.subheader("2. Generate Sequence")

col1, col2 = st.columns([1, 4])

with col1:
    generate_btn = st.button("🚀 Generate Sequence", type="primary")

if generate_btn:
    # Convert DataFrame to List of Dicts (mimicking csv.DictReader format)
    # Ensure types are string/compatible if logic expects them, or handle types in v4
    # v4 expects dicts with keys matching CSV headers.
    
    # Pre-processing: ensure 'station_no' is present
    if "station_no" not in edited_df.columns:
        st.error("Error: 'station_no' column is missing from the data.")
    else:
        # Convert to records
        records = edited_df.astype(str).to_dict(orient="records")
        
        # Call v4 Logic
        # Note: v4.generate_sequence_data returns a list of lists [["Command", "Value"], ...]
        with st.spinner("Generating..."):
            seq_data = v4.generate_sequence_data(records)
            
        if seq_data and len(seq_data) > 1 and seq_data[0][0] == "Error":
             st.error(seq_data[0][1])
        else:
            # Convert to DataFrame for display
            # seq_data[0] is Header ["Command", "Value"]
            headers = seq_data[0]
            rows = seq_data[1:]
            
            df_result = pd.DataFrame(rows, columns=headers)
            
            st.success("Sequence Generated Successfully!")
            st.dataframe(df_result, use_container_width=True)
            
            # CSV Download
            csv_buffer = io.StringIO()
            # Write just the rows as per requirement? Or with header?
            # v4 outputted header "Command,Value" in print.
            # Let's save standard CSV.
            df_result.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="⬇️ Download Sequence CSV",
                data=csv_buffer.getvalue(),
                file_name="generated_sequence.csv",
                mime="text/csv"
            )
#python -m streamlit run ui_editor.py
