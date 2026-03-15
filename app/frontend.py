# # import streamlit as st
# # import requests
# # from PIL import Image
# # import io
# # import base64

# # # 1. Page Configuration
# # st.set_page_config(page_title="RetinaGuard AI", page_icon="👁️", layout="centered")

# # st.title("👁️ RetinaGuard: Diabetic Retinopathy AI")
# # st.write("Upload a high-resolution retinal scan to receive an instant AI-powered screening.")

# # # 2. Sidebar Information
# # with st.sidebar:
# #     st.header("System Architecture")
# #     st.write("This application uses a deep learning pipeline (EfficientNet) to classify Diabetic Retinopathy into 5 distinct stages.")
# #     st.divider()
# #     st.warning("Note: This is a portfolio demonstration tool, not a substitute for a professional medical diagnosis.")

# # # 3. File Uploader UI
# # uploaded_file = st.file_uploader("Choose a retinal image (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     # Read image into memory
# #     image = Image.open(uploaded_file)
    
# #     # 4. The Prediction Trigger
# #     if st.button("Run AI Diagnosis", type="primary"):
# #         with st.spinner("Connecting to FastAPI backend and analyzing image..."):
# #             try:
# #                 # Package the image to send over HTTP
# #                 files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
# #                 # Send the POST request to our running FastAPI server
# #                 response = requests.post("http://127.0.0.1:8000/predict", files=files)
                
# #                 # 5. Handle the Response
# #                 if response.status_code == 200:
# #                     result = response.json()
# #                     diagnosis = result["diagnosis"]
# #                     confidence = result["confidence"]
# #                     heatmap_base64 = result.get("heatmap")
                    
# #                     st.success("Analysis Complete!")
                    
# #                     # Create clean visual columns for the metrics
# #                     col1, col2 = st.columns(2)
# #                     with col1:
# #                         st.metric(label="Predicted Stage", value=diagnosis)
# #                     with col2:
# #                         st.metric(label="AI Confidence", value=confidence)
                        
# #                     st.divider()
# #                     st.subheader("Explainable AI Analysis")
# #                     st.write("The heatmap below highlights the specific regions of the retina that drove the AI's diagnosis.")
                    
# #                     # Display original and heatmap side-by-side
# #                     img_col1, img_col2 = st.columns(2)
# #                     with img_col1:
# #                         st.image(image, caption="Original Retinal Scan", use_container_width=True)
# #                     with img_col2:
# #                         if heatmap_base64:
# #                             heatmap_bytes = base64.b64decode(heatmap_base64)
# #                             heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
# #                             st.image(heatmap_img, caption="Grad-CAM Focus Heatmap", use_container_width=True)
# #                         else:
# #                             st.warning("⚠️ The backend did not return a heatmap. Please verify your FastAPI server was restarted.")
                            
# #                 else:
# #                     st.error(f"Backend API Error: {response.text}")
            
# #             except requests.exceptions.ConnectionError:
# #                 st.error("🚨 Could not connect to the AI backend. Make sure your FastAPI server is running on port 8000!")


# import streamlit as st
# import requests
# from PIL import Image
# import io
# import base64
# from fpdf import FPDF


# # 1. Page Configuration
# st.set_page_config(page_title="RetinaGuard Clinical AI", page_icon="👁️", layout="wide")

# # Title and styling
# st.title("👁️ RetinaGuard: Clinical Triage & AI Screening")
# st.markdown("An enterprise-grade Diabetic Retinopathy detection system combining deep learning with patient metadata.")

# # 2. Sidebar: Patient Intake Form
# with st.sidebar:
#     st.header("📋 Patient Intake Form")
#     st.write("Enter clinical metadata to generate a comprehensive triage report.")
    
#     # Input fields for the backend
#     patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=45)
#     years_diabetic = st.number_input("Years Diagnosed with Diabetes", min_value=0, max_value=80, value=5)
#     hba1c = st.number_input("Latest HbA1c Level (%)", min_value=3.0, max_value=20.0, value=6.5, step=0.1)
    
#     st.divider()
#       # --- PDF GENERATION LOGIC ---
#                         st.divider()
                        
#                         def create_pdf():
#                             pdf = FPDF()
#                             pdf.add_page()
#                             pdf.set_font("Arial", 'B', 16)
#                             pdf.cell(200, 10, txt="RetinaGuard Clinical AI - Medical Report", ln=True, align='C')
                            
#                             pdf.set_font("Arial", '', 12)
#                             pdf.ln(10)
#                             pdf.cell(200, 10, txt=f"Patient Age: {patient_age}", ln=True)
#                             pdf.cell(200, 10, txt=f"Years Diabetic: {years_diabetic}", ln=True)
#                             pdf.cell(200, 10, txt=f"Latest HbA1c: {hba1c}%", ln=True)
                            
#                             pdf.ln(10)
#                             pdf.set_font("Arial", 'B', 12)
#                             pdf.cell(200, 10, txt=f"AI Diagnosis: {diagnosis} (Confidence: {confidence})", ln=True)
                            
#                             pdf.ln(5)
#                             pdf.set_font("Arial", '', 11)
#                             pdf.multi_cell(0, 10, txt=f"Clinical Triage & Next Steps:\n{triage}")
                            
#                             pdf.ln(10)
#                             pdf.set_font("Arial", 'I', 10)
#                             pdf.cell(200, 10, txt="Disclaimer: This is an AI-generated report and not a certified medical diagnosis.", ln=True)
                            
#                             return pdf.output(dest='S').encode('latin-1')

#                         # Create the download button
#                         pdf_bytes = create_pdf()
#                         st.download_button(
#                             label="📄 Download Clinical PDF Report",
#                             data=pdf_bytes,
#                             file_name="RetinaGuard_Clinical_Report.pdf",
#                             mime="application/pdf",
#                             type="primary"
#                         )  
#     st.warning("⚠️ Disclaimer: This is a portfolio demonstration system, not a certified medical device.")

# # 3. Main Dashboard Tabs
# tab1, tab2, tab3 = st.tabs(["🩺 Clinical Screening", "📖 Patient Education", "⚙️ System Architecture"])

# # --- TAB 1: The Core Application ---
# with tab1:
#     st.subheader("Upload Retinal Scan")
#     uploaded_file = st.file_uploader("Select a high-resolution fundus image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
        
#         if st.button("Generate Clinical Report", type="primary"):
#             with st.spinner("Processing image and analyzing patient data..."):
#                 try:
#                     # Package the multipart form data (Image + Patient Data)
#                     files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
#                     data = {
#                         "age": patient_age,
#                         "years_diabetic": years_diabetic,
#                         "hba1c": hba1c
#                     }
                    
#                     # Send to FastAPI
#                     response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)
                    
#                     if response.status_code == 200:
#                         result = response.json()
#                         diagnosis = result["diagnosis"]
#                         confidence = result["confidence"]
#                         triage = result.get("triage_recommendation", "No recommendation provided.")
#                         heatmap_base64 = result.get("heatmap")
                        
#                         st.success("Analysis Complete!")
                        
#                         # Top metrics row
#                         m1, m2, m3 = st.columns(3)
#                         m1.metric("Predicted Stage", diagnosis)
#                         m2.metric("AI Confidence", confidence)
#                         m3.metric("Patient HbA1c", f"{hba1c}%")
                        
#                         # Highlighted Triage Recommendation
#                         st.info(f"**Clinical Triage & Next Steps:**\n\n{triage}")
                        
#                         st.divider()
#                         st.subheader("Visual Analysis (Explainable AI)")
                        
#                         # Image displays
#                         img_col1, img_col2 = st.columns(2)
#                         with img_col1:
#                             st.image(image, caption="Original Scan", use_container_width=True)
#                         with img_col2:
#                             if heatmap_base64:
#                                 heatmap_bytes = base64.b64decode(heatmap_base64)
#                                 heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
#                                 st.image(heatmap_img, caption="Grad-CAM Focus Heatmap", use_container_width=True)
#                             else:
#                                 st.warning("No heatmap returned from server.")
#                     else:
#                         st.error(f"Backend Error: {response.text}")
#                 except requests.exceptions.ConnectionError:
#                     st.error("🚨 Could not connect to the backend. Ensure FastAPI is running on port 8000.")

# # --- TAB 2: Patient Education ---
# with tab2:
#     st.header("Understanding Diabetic Retinopathy")
#     st.write("Diabetic Retinopathy (DR) is a complication of diabetes that damages the blood vessels in the retina.")
#     st.markdown('''
#     * **Healthy:** Normal blood vessels.
#     * **Mild/Moderate:** Microaneurysms and minor blockages.
#     * **Severe:** Extensive blocked vessels, depriving the retina of blood supply.
#     * **Proliferative:** New, fragile blood vessels grow (neovascularization) and can bleed easily, leading to severe vision loss.
#     ''')
#     st.info("Maintaining healthy blood sugar levels, blood pressure, and cholesterol are the best ways to prevent DR.")

# # --- TAB 3: Architecture for Interviewers ---
# with tab3:
#     st.header("Technical Architecture")
#     st.write("This system was engineered as a robust, decoupled microservice application.")
#     st.markdown('''
#     * **Frontend:** Streamlit providing a dynamic, multi-tab SaaS interface.
#     * **Backend:** FastAPI for high-performance, asynchronous REST API endpoints.
#     * **Machine Learning:** EfficientNetB3 trained via Transfer Learning on a heavily imbalanced medical dataset.
#     * **Explainable AI:** Grad-CAM (Gradient-weighted Class Activation Mapping) implemented with OpenCV to ensure transparency in AI decision-making.
#     ''')


import streamlit as st
import requests
from PIL import Image
import io
import base64
from fpdf import FPDF

# -------------------------------
# PDF GENERATION FUNCTION
# -------------------------------
def create_pdf(patient_age, years_diabetic, hba1c, diagnosis, confidence, triage):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="RetinaGuard Clinical AI - Medical Report", ln=True, align='C')

    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Age: {patient_age}", ln=True)
    pdf.cell(200, 10, txt=f"Years Diabetic: {years_diabetic}", ln=True)
    pdf.cell(200, 10, txt=f"Latest HbA1c: {hba1c}%", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"AI Diagnosis: {diagnosis} (Confidence: {confidence})", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, txt=f"Clinical Triage & Next Steps:\n{triage}")

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Disclaimer: This is an AI-generated report and not a certified medical diagnosis.", ln=True)

    return pdf.output(dest='S').encode('latin-1')


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="RetinaGuard Clinical AI",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ RetinaGuard: Clinical Triage & AI Screening")
st.markdown(
    "An enterprise-grade **Diabetic Retinopathy detection system** combining deep learning with patient metadata."
)

# -------------------------------
# SIDEBAR (PATIENT FORM)
# -------------------------------
with st.sidebar:

    st.header("📋 Patient Intake Form")
    st.write("Enter clinical metadata to generate a comprehensive triage report.")

    patient_age = st.number_input(
        "Patient Age",
        min_value=1,
        max_value=120,
        value=45
    )

    years_diabetic = st.number_input(
        "Years Diagnosed with Diabetes",
        min_value=0,
        max_value=80,
        value=5
    )

    hba1c = st.number_input(
        "Latest HbA1c Level (%)",
        min_value=3.0,
        max_value=20.0,
        value=6.5,
        step=0.1
    )

    st.divider()

    st.warning(
        "⚠️ Disclaimer: This is a portfolio demonstration system, not a certified medical device."
    )

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🩺 Clinical Screening", "📖 Patient Education", "⚙️ System Architecture"]
)

# =====================================================
# TAB 1 : CORE APPLICATION
# =====================================================
with tab1:

    st.subheader("Upload Retinal Scan")

    uploaded_file = st.file_uploader(
        "Select a high-resolution fundus image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Retinal Scan", use_container_width=True)

        if st.button("Generate Clinical Report", type="primary"):

            with st.spinner("Processing image and analyzing patient data..."):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    data = {
                        "age": patient_age,
                        "years_diabetic": years_diabetic,
                        "hba1c": hba1c
                    }

                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        files=files,
                        data=data
                    )

                    if response.status_code == 200:

                        result = response.json()

                        diagnosis = result["diagnosis"]
                        confidence = result["confidence"]
                        triage = result.get(
                            "triage_recommendation",
                            "No recommendation provided."
                        )

                        heatmap_base64 = result.get("heatmap")

                        st.success("Analysis Complete!")

                        # Metrics
                        m1, m2, m3 = st.columns(3)

                        m1.metric("Predicted Stage", diagnosis)
                        m2.metric("AI Confidence", confidence)
                        m3.metric("Patient HbA1c", f"{hba1c}%")

                        st.info(f"**Clinical Triage & Next Steps:**\n\n{triage}")

                        st.divider()

                        st.subheader("Visual Analysis (Explainable AI)")

                        img_col1, img_col2 = st.columns(2)

                        with img_col1:
                            st.image(
                                image,
                                caption="Original Scan",
                                use_container_width=True
                            )

                        with img_col2:

                            if heatmap_base64:

                                heatmap_bytes = base64.b64decode(heatmap_base64)
                                heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

                                st.image(
                                    heatmap_img,
                                    caption="Grad-CAM Focus Heatmap",
                                    use_container_width=True
                                )

                            else:
                                st.warning("No heatmap returned from server.")

                        # ---------------------------------
                        # PDF DOWNLOAD
                        # ---------------------------------
                        pdf_bytes = create_pdf(
                            patient_age,
                            years_diabetic,
                            hba1c,
                            diagnosis,
                            confidence,
                            triage
                        )

                        st.download_button(
                            label="📄 Download Clinical PDF Report",
                            data=pdf_bytes,
                            file_name="RetinaGuard_Clinical_Report.pdf",
                            mime="application/pdf"
                        )

                    else:
                        st.error(f"Backend Error: {response.text}")

                except requests.exceptions.ConnectionError:

                    st.error(
                        "🚨 Could not connect to the backend. Ensure FastAPI is running on port 8000."
                    )

# =====================================================
# TAB 2 : PATIENT EDUCATION
# =====================================================
with tab2:

    st.header("Understanding Diabetic Retinopathy")

    st.write(
        "Diabetic Retinopathy (DR) is a complication of diabetes that damages the blood vessels in the retina."
    )

    st.markdown(
        """
* **Healthy:** Normal blood vessels.

* **Mild/Moderate:** Microaneurysms and minor blockages.

* **Severe:** Extensive blocked vessels depriving the retina of blood supply.

* **Proliferative:** New fragile blood vessels grow and may bleed, causing severe vision loss.
"""
    )

    st.info(
        "Maintaining healthy blood sugar levels, blood pressure, and cholesterol are the best ways to prevent DR."
    )

# =====================================================
# TAB 3 : ARCHITECTURE
# =====================================================
with tab3:

    st.header("Technical Architecture")

    st.write(
        "This system was engineered as a robust, decoupled microservice application."
    )

    st.markdown(
        """
* **Frontend:** Streamlit providing a dynamic multi-tab SaaS interface.

* **Backend:** FastAPI for high-performance asynchronous REST API.

* **Machine Learning:** EfficientNetB3 trained using Transfer Learning on medical retinal datasets.

* **Explainable AI:** Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize AI decision focus.
"""
    )
