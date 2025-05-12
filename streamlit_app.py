import streamlit as st
from attendance_system import mark_attendance_from_images
from train_model import train_model
import os
import pandas as pd
from fpdf import FPDF
from PIL import Image
from io import BytesIO

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.title("📸 Face Recognition Attendance System")

# Ensure Attendance folder exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

# ---------------------- Excel to PDF Function ----------------------
def convert_excel_to_pdf(excel_path):
    df = pd.read_excel(excel_path)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Attendance Sheet", ln=True, align="C")
    pdf.ln(10)

    for column in df.columns:
        pdf.cell(48, 10, txt=str(column), border=1)
    pdf.ln()

    for _, row in df.iterrows():
        for item in row:
            pdf.cell(48, 10, txt=str(item), border=1)
        pdf.ln()

    # Write PDF to memory buffer
    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO()
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)
    return pdf_buffer

# ---------------------- Sidebar Menu ----------------------
st.sidebar.title("Menu")
menu_option = st.sidebar.radio("Select an action:", ["Train Model", "Mark Attendance", "Download Attendance PDF"])

# ---------------------- Train Model ----------------------
if menu_option == "Train Model":
    st.subheader("🧠 Train Face Recognition Model")
    if st.button("Train Now"):
        with st.spinner("Training model..."):
            train_model()
        st.success("✅ Training completed successfully!")

# ---------------------- Mark Attendance ----------------------
elif menu_option == "Mark Attendance":
    st.subheader("🖼️ Upload Images for Attendance")
    uploaded_files = st.file_uploader("Choose image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        image_paths = []
        os.makedirs("temp_images", exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp_images", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(file_path)

            img = Image.open(file_path)
            st.image(img, caption=uploaded_file.name, width=150)

        if st.button("Mark Attendance"):
            with st.spinner("Processing images..."):
                result = mark_attendance_from_images(image_paths)
            st.success("✅ Attendance marked!")
            st.info(result)

        if st.button("Clear Uploaded Images"):
            for file_path in image_paths:
                os.remove(file_path)
            os.rmdir("temp_images")
            st.success("🗑️ Temporary images cleared.")

# ---------------------- Download PDF ----------------------
elif menu_option == "Download Attendance PDF":
    st.subheader("📁 Download Attendance Sheet as PDF")

    attendance_files = [f for f in os.listdir("Attendance") if f.endswith(".xlsx")]

    if attendance_files:
        selected_file = st.selectbox("Select an attendance Excel file", attendance_files)

        if selected_file:
            selected_path = os.path.join("Attendance", selected_file)
            pdf_buf = convert_excel_to_pdf(selected_path)

            st.download_button(
                label="📄 Download Selected Attendance PDF",
                data=pdf_buf,
                file_name=selected_file.replace(".xlsx", ".pdf"),
                mime="application/pdf"
            )
    else:
        st.warning("⚠️ No attendance Excel files found.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("© 2025 Face Recognition Attendance System")
