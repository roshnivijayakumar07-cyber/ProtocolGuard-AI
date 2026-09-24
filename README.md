# 🛡️ ProtocolGuard-AI

> **GCP Clinical Protocol Deviation & Regulatory CAPA Engine**

ProtocolGuard-AI is an investigational Clinical Decision Support Tool designed for Clinical Research Associates (CRAs), Quality Assurance (QA) auditors, and Principal Investigators. It automates clinical deviation triage, assesses patient safety and data integrity risks against protocol specifications, and drafts auditable Corrective and Preventive Action (CAPA) plans adhering to **ICH-GCP E6(R2)** and **FDA 21 CFR Part 312** standards.

---

## 🚀 Key Features

- **Local Vector Retrieval:** Protocol specifications are embedded and indexed locally using `FastEmbed` (`BAAI/bge-small-en-v1.5`) and `ChromaDB` for fast, offline similarity matching.
- **Deterministic AI Auditing:** Uses Google Gemini models with LangChain structured output parsers (`Pydantic`) to strictly categorize deviations into Minor, Major, or Critical tiers.
- **Automated CAPA Generation:** Formulates a 3-step compliance action plan:
  1. **Immediate Containment** (Action within 24 hours)
  2. **Corrective Action** (Root-cause remediation)
  3. **Preventive Safeguard** (Long-term systemic control)
- **Interactive Clinical Dashboard:** Built with Streamlit, including pre-configured clinical scenario presets (visit window shifts, biomarker omissions, hepatic exclusion thresholds).

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/) / [langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- **AI Models:** Google Gemini (`gemini-3.5-flash-lite`)
- **Embeddings & Vector Store:** [FastEmbed](https://github.com/qdrant/fastembed) & [ChromaDB](https://www.trychroma.com/)
- **Schema Validation:** [Pydantic v2](https://docs.pydantic.dev/)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone git@github.com:roshnivijayakumar07-cyber/ProtocolGuard-AI.git
cd ProtocolGuard-AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file or enter your API key directly in the Streamlit sidebar:
```bash
cp .env.example .env
```
Add your Google Gemini API key in `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
ProtocolGuard-AI/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules for credentials & artifacts
├── README.md                 # Project documentation
├── app.py                    # Streamlit clinical audit application
├── requirements.txt          # Python dependencies
└── data/
    └── sample_protocol.txt   # Example Phase 2 oncology protocol specifications
```

---

## ⚖️ Regulatory Notice

*ProtocolGuard-AI is an investigational Clinical Decision Support Tool. All risk classifications, protocol citations, and CAPA drafts must be independently reviewed and approved by a qualified Clinical Research Associate (CRA) or Principal Investigator before submission under FDA/ICH-GCP guidelines.*
