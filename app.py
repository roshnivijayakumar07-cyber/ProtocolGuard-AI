import os
import streamlit as st
from typing import List
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(
    page_title="ProtocolGuard-AI | Clinical QA Copilot",
    page_icon="🛡️",
    layout="wide"
)

st.caption(
    "⚖️ **Regulatory Notice:** ProtocolGuard-AI is an investigational Clinical Decision Support Tool. "
    "All risk classifications, protocol citations, and CAPA drafts must be independently reviewed and approved "
    "by a qualified Clinical Research Associate (CRA) or Principal Investigator before submission under FDA/ICH-GCP guidelines."
)

st.title("🛡️ ProtocolGuard-AI")
st.subheader("GCP Clinical Protocol Deviation & Regulatory CAPA Engine")

class CAPAResponse(BaseModel):
    severity: str = Field(description="Must be strictly: Minor, Major, or Critical")
    risk_score: int = Field(description="Calculated risk score integer between 0 and 100")
    violated_clauses: List[str] = Field(description="Exact protocol sections and clause identifiers cited")
    safety_impact: str = Field(description="Direct clinical impact on subject health and patient safety")
    data_integrity_impact: str = Field(description="Impact on clinical trial data validity and statistical endpoints")
    root_cause: str = Field(description="Identified operational, systemic, or human error root cause")
    capa_immediate: str = Field(description="Immediate containment step required within 24 hours")
    capa_corrective: str = Field(description="Process corrective action to resolve the current non-compliance")
    capa_preventive: str = Field(description="Long-term systemic safeguard to prevent recurrence across sites")

@st.cache_resource(show_spinner=False)
def initialize_knowledge_base():
    data_path = os.path.join("data", "sample_protocol.txt")
    if not os.path.exists(data_path):
        return None
    
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=40,
        separators=["\n\n", "\n", "- ", " "]
    )
    chunks = splitter.split_documents(documents)
    
    # FastEmbed runs purely local ONNX inference: zero torch errors, zero API 404s
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_db = Chroma.from_documents(chunks, embeddings)
    return vector_db

with st.spinner("Initializing Clinical Protocol Vector Store..."):
    vector_db = initialize_knowledge_base()

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password",
    help="Enter your Google AI Studio API key"
)

st.sidebar.markdown("---")
st.sidebar.header("🧪 Test Presets")

preset_choice = st.sidebar.radio(
    "Select an incident scenario:",
    (
        "Custom Incident",
        "Case 1: Routine Visit Window Shift (Minor)",
        "Case 2: Primary Biomarker & ECG Omission (Major)",
        "Case 3: Ineligible Dosing Above Hepatic Threshold (Critical)"
    )
)

preset_texts = {
    "Case 1: Routine Visit Window Shift (Minor)": (
        "Subject SUBJ-042 attended Cycle 3 Day 1 routine visit on Day 23 instead of Day 21 due to public transit strikes. "
        "All vital signs and lab results were normal before medication was dispensed."
    ),
    "Case 2: Primary Biomarker & ECG Omission (Major)": (
        "At the Week 12 primary endpoint evaluation, study medication was dispensed to Subject SUBJ-118, but the coordinator "
        "forgot to collect the fasting plasma blood draw and missed the 12-lead safety ECG prior to patient discharge."
    ),
    "Case 3: Ineligible Dosing Above Hepatic Threshold (Critical)": (
        "Subject SUBJ-089 was randomized and administered the initial 50 mg dose of Compound XT-9. Post-dosing review of "
        "screening labs revealed baseline serum ALT was 4.2 times the Upper Limit of Normal (ULN)."
    )
}

default_text = preset_texts.get(preset_choice, "")

st.markdown("### 📋 Clinical Deviation Incident Log")
incident_input = st.text_area(
    "Enter raw site deviation notes or use a preset from the sidebar:",
    value=default_text,
    height=120,
    placeholder="Describe the clinical deviation (e.g., missed procedures, dosage anomalies, eligibility breaches)..."
)

audit_button = st.button("🚀 Run GCP Compliance Audit", type="primary")

if audit_button:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif vector_db is None:
        st.error("Database initialization failed. Please make sure `data/sample_protocol.txt` exists.")
    elif not incident_input.strip():
        st.warning("Please enter or select an incident description to evaluate.")
    else:
        with st.spinner("Analyzing protocol specifications and generating regulatory CAPA..."):
            try:
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(incident_input)
                context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                parser = JsonOutputParser(pydantic_object=CAPAResponse)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash-lite",
                    google_api_key=api_key,
                    temperature=0.0
                )
                
                prompt_template = ChatPromptTemplate.from_template(
                    """You are a Senior Clinical Quality Assurance Auditor adhering to ICH-GCP E6(R2) standards.
Evaluate the reported clinical deviation against the retrieved clinical trial protocol context.

Protocol Context:
{context}

Reported Incident:
{incident}

Evaluation Rules:
1. Determine the severity tier strictly as 'Minor', 'Major', or 'Critical':
   - Minor: Administrative or slight scheduling delay with zero patient safety risk and negligible endpoint compromise.
   - Major: Compromises endpoint data integrity, violates diagnostic testing sequences, or introduces potential safety risks.
   - Critical: Directly endangers patient safety (e.g., dosing an ineligible patient, severe organ toxicity risks) or invalidates trial cohorts.
2. Calculate a Risk Score integer between 0 and 100.
3. Cite the exact protocol section and clause codes found in the context.
4. Formulate a concrete, auditable three-step CAPA plan: Immediate containment, Root-cause corrective action, and Long-term preventive safeguard.

{format_instructions}
"""
                )
                
                chain = prompt_template | llm | parser
                result = chain.invoke({
                    "context": context_str,
                    "incident": incident_input,
                    "format_instructions": parser.get_format_instructions()
                })
                
                st.markdown("---")
                st.markdown("### 📊 Regulatory Audit Results")
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                severity = result.get("severity", "Unknown")
                risk_score = result.get("risk_score", 0)
                
                with col1:
                    if severity.lower() == "critical":
                        st.error(f"### Severity: {severity.upper()}")
                    elif severity.lower() == "major":
                        st.warning(f"### Severity: {severity.upper()}")
                    else:
                        st.info(f"### Severity: {severity.upper()}")
                
                with col2:
                    st.metric("GCP Compliance Risk Score", f"{risk_score} / 100")
                
                with col3:
                    st.markdown("**Cited Protocol Clauses:**")
                    for clause in result.get("violated_clauses", []):
                        st.write(f"- `{clause}`")
                
                st.markdown("#### 🔬 Clinical Impact & Root-Cause Breakdown")
                imp_col1, imp_col2 = st.columns(2)
                with imp_col1:
                    st.markdown("**Patient Safety Impact:**")
                    st.write(result.get("safety_impact", "N/A"))
                with imp_col2:
                    st.markdown("**Data Integrity Impact:**")
                    st.write(result.get("data_integrity_impact", "N/A"))
                
                st.markdown("**Identified Root Cause:**")
                st.info(result.get("root_cause", "N/A"))
                
                st.markdown("#### 🛠️ Corrective & Preventive Action (CAPA) Plan")
                capa1, capa2, capa3 = st.columns(3)
                with capa1:
                    st.markdown("**1. Immediate Containment**")
                    st.success(result.get("capa_immediate", "N/A"))
                with capa2:
                    st.markdown("**2. Corrective Action**")
                    st.warning(result.get("capa_corrective", "N/A"))
                with capa3:
                    st.markdown("**3. Preventive Safeguard**")
                    st.info(result.get("capa_preventive", "N/A"))
                
                with st.expander("🔍 View Retrieved Protocol Context Chunks"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(doc.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")