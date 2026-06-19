import streamlit as st
import fitz  # PyMuPDF
from google import genai
from duckduckgo_search import DDGS
import pandas as pd
import os

st.set_page_config(page_title="Fact-Check Agent", page_icon="🛡️", layout="wide")

st.title("🛡️ Automated Fact-Checking Web App")
st.write("Upload a marketing or technical PDF to verify its stats, dates, and figures against the live web.")

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
            Extract only specific factual claims (stats, dates, financial, or technical figures) from the text below.
            Rules:
            - Return exactly one claim per line.
            - Do not include bullet points, numbers, or headers.
            - Do not add explanations.
            
            Text:
            {pdf_text}
            """
            try:
                extract_res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=extract_prompt
                )
                claims = [line.strip() for line in extract_res.text.split("\n") if line.strip()]
            except Exception as e:
                st.error(f"Error extracting claims via Gemini: {e}")
                claims = []

        if claims:
            st.subheader(f"Found {len(claims)} factual claims. Starting live verification...")
            results = []
            progress_bar = st.progress(0)

            for index, claim in enumerate(claims):
                evidence = ""
                try:
                    with DDGS() as ddgs:
                        search_results = list(ddgs.text(claim, max_results=3))
                        if search_results:
                            evidence = " ".join([r['body'] for r in search_results if 'body' in r])
                except Exception:
                    evidence = "Live search limit reached. Verifying using core knowledge."

                verify_prompt = f"""
                You are a precise "Truth Layer" fact-checker. 
                Verify the given Claim using the provided Web Search Evidence.
                
                Claim: {claim}
                Web Search Evidence: {evidence}
                
                Categorize the status strictly as one of these:
                - "Verified" (If the claim matches the live web data)
                - "Inaccurate" (If the claim contains outdated statistics or slight mismatches)
                - "False" (If the statement is completely wrong or no evidence supports it)

                Return the response STRICTLY in this exact format:
                Status: [Verified or Inaccurate or False]
                Correct Fact: [Provide the actual true fact/number based on evidence]
                Reason: [One sentence explanation]
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
                        "Correct Fact": "Error",
                        "Reason": "API Call Failed"
                    })
                
                progress_bar.progress((index + 1) / len(claims))

            df = pd.DataFrame(results)
            st.success("Verification Complete!")
            st.subheader("📋 Final Fact-Check Report")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report as CSV", data=csv, file_name="fact_check_report.csv", mime="text/csv")
        else:
            st.warning("No factual claims could be extracted from this PDF.")
