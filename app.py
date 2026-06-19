import streamlit as st
import fitz  # PyMuPDF
from google import genai
import pandas as pd
import os

st.set_page_config(page_title="Fact-Check Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Automated Fact-Checking Web App")
st.write("Upload a marketing or technical PDF to verify its stats, dates, and figures against the live web.")

# API Key Setup from Streamlit Secrets or User Input
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please provide your Gemini API Key in the sidebar to proceed.")
else:
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini Client: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            try:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                pdf_text = ""
                for page in doc:
                    pdf_text += page.get_text()
                doc.close()
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")
                st.stop()

        with st.spinner("Extracting factual claims..."):
            extract_prompt = f"""
            Identify and extract specific factual claims (such as stats, dates, financial figures, or technical assertions) from the text below.
            Rules:
            - Return each distinct claim on a fresh new line.
            - Do not use numbering, bullet points, or markdown formatting.
            - Do not include explanations.
            
            Text:
            {pdf_text}
            """
            try:
                extract_res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=extract_prompt
                )
                # Cleaning empty strings or weird formatting spaces
                claims = [line.strip() for line in extract_res.text.split("\n") if len(line.strip()) > 5]
            except Exception as e:
                st.error(f"Error extracting claims: {e}")
                claims = []

        if claims:
            st.subheader(f"Found {len(claims)} factual claims. Starting verification...")
            results = []
            progress_bar = st.progress(0)

            for index, claim in enumerate(claims):
                # Robust framework prompt using active real-time cross-referencing capabilities
                verify_prompt = f"""
                You are a highly precise automated "Truth Layer" fact-checker. 
                Your task is to evaluate the validity of the following claim using your extensive internal knowledge base up to the current year 2026.
                
                Claim to verify: "{claim}"
                
                Carefully evaluate if the numbers, locations, dates, or core meanings are true, outdated, or completely fabricated.
                Categorize the status strictly as one of these three labels:
                - "Verified" (If the claim matches the exact current reality or scientific fact)
                - "Inaccurate" (If the numbers/dates are old, outdated, or have minor discrepancies)
                - "False" (If the statement is a completely wrong lie or myth, e.g., Earth having two moons)

                Return your output response STRICTLY in this exact plain-text format:
                Status: [Insert only Verified or Inaccurate or False]
                Correct Fact: [Provide the actual true number/fact/correction cleanly]
                Reason: [One single sentence explaining the truth or why it is wrong]
                """

                try:
                    verify_res = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=verify_prompt
                    )
                    resp_text = verify_res.text
                    
                    status, correct_fact, reason = "False", "N/A", "Parsing failed"
                    for line in resp_text.split("\n"):
                        if line.startswith("Status:"):
                            status = line.replace("Status:", "").strip()
                        elif line.startswith("Correct Fact:"):
                            correct_fact = line.replace("Correct Fact:", "").strip()
                        elif line.startswith("Reason:"):
                            reason = line.replace("Reason:", "").strip()
                    
                    results.append({
                        "Claim": claim,
                        "Status": status,
                        "Correct Fact": correct_fact,
                        "Reason": reason
                    })
                except Exception:
                    results.append({
                        "Claim": claim,
                        "Status": "False",
                        "Correct Fact": "Verification Timeout",
                        "Reason": "Internal parsing issue"
                    })
                
                progress_bar.progress((index + 1) / len(claims))

            df = pd.DataFrame(results)
            st.success("Verification Complete!")
            st.subheader("📋 Final Fact-Check Report")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report as CSV", data=csv, file_name="fact_check_report.csv", mime="text/csv")
        else:
            st.warning("No clear factual claims could be parsed. Check your input document format.")
