import streamlit as st
import fitz  # PyMuPDF
import os
import anthropic # Import the core anthropic client library
from langchain_core.runnables import RunnableLambda # For wrapping the direct API call
from langchain_core.messages import HumanMessage, SystemMessage # Still useful for message formatting
from dotenv import load_dotenv # Optional: for loading token from .env file

# Load environment variables from a .env file (optional)
# This line might not be strictly necessary if you are relying solely on st.secrets
# but can be kept for local development if you prefer .env for non-Streamlit runs.
load_dotenv()

def initialize_llm_runnable():
    """
    Initializes and returns a RunnableLambda that directly calls the Anthropic API.
    This bypasses ChatAnthropic to gain full control over API parameters for custom endpoints.
    Expects LLMFOUNDRY_TOKEN to be set in st.secrets.
    """
    try:
        # --- CHANGE STARTS HERE ---
        # Access the token using st.secrets
        token = st.secrets.get("LLMFOUNDRY_TOKEN")
        # --- CHANGE ENDS HERE ---

        if not token:
            st.error("LLMFOUNDRY_TOKEN not found in st.secrets. Please set it in your Streamlit secrets.")
            return None

        # Initialize the core Anthropic client directly with the custom base_url
        client = anthropic.Anthropic(
            api_key=f'{token}:my-test-project', # Your LLMFoundry API key
            base_url="https://llmfoundry.straive.com/anthropic/", # Your custom base URL
        )

        # Define a function to make the direct API call
        def invoke_anthropic_api(messages):
            
            system_prompt_content = ""
            final_messages_for_anthropic = []

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    # For Messages API, the system prompt is a top-level parameter
                    system_prompt_content = msg.content
                elif isinstance(msg, HumanMessage):
                    final_messages_for_anthropic.append({"role": "user", "content": msg.content})
                # Add logic for AIMessage if you're processing conversation history
                # elif isinstance(msg, AIMessage):
                #     final_messages_for_anthropic.append({"role": "assistant", "content": msg.content})

            # Ensure there is at least one user message to send
            if not final_messages_for_anthropic:
                raise ValueError("No user message content found to send to the Anthropic API.")

            # Make the direct API call using the core anthropic client
            api_response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                messages=final_messages_for_anthropic,
                system=system_prompt_content if system_prompt_content else None # Pass system prompt separately
            )
            
            # --- START OF FIX FOR 'list' object has no attribute 'encode' ---
            # api_response.content is a list of content blocks (e.g., TextBlock, ImageBlock)
            # We need to extract the 'text' from these blocks.
            if isinstance(api_response.content, list):
                # Concatenate text from all TextBlock objects in the response
                full_response_text = "".join([
                    block.text for block in api_response.content
                    if hasattr(block, 'text') and block.type == 'text'
                ])
            else:
                # Fallback, though for Claude 3 Messages API, it's typically a list
                full_response_text = str(api_response.content)

            return full_response_text # Return the extracted, single string content
            # --- END OF FIX ---

        # Wrap the direct API call function in a RunnableLambda
        # This allows it to be invoked similarly to a LangChain LLM object.
        return RunnableLambda(invoke_anthropic_api)

    except Exception as e:
        st.error(f"Error setting up LLM integration: {e}. "
                 f"Please ensure `LLMFOUNDRY_TOKEN` is correct in `st.secrets` and you have installed `anthropic` package (`pip install anthropic`).")
        return None

def extract_text_from_pdf(pdf_file):
    """
    Extracts text page by page from an uploaded PDF file-like object.
    Returns a string with text formatted as "Page X:\n[text content]\n\nPage Y:\n[text content]".
    """
    document_text_pages = []
    temp_file_path = "temp_uploaded_pdf.pdf"
    try:
        # Save the uploaded file to a temporary location to be read by fitz
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        doc = fitz.open(temp_file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            document_text_pages.append(f"Page {page_num + 1}:\n{text}\n")
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    finally:
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return "\n".join(document_text_pages)


INDEXING_PROMPT = """You are an expert indexer. Your task is to create a comprehensive subject index for the given document content. Follow these strict guidelines to ensure a high-quality, professional index:

**I. Overall Process & Initial Considerations:**

1.  **Understand the Document:**
    * You will receive the full text of the document, formatted with page numbers.
    * If a "Preface" or "Introduction" section is explicitly provided, identify and read it first to understand the book's outline and assist in forming an indexing structure.
2.  **Identify Relevant Information:**
    * Locate and identify all relevant concepts, themes, basic concepts, and important terms within the document.
    * **Discriminate:** Clearly distinguish between significant information on a subject and a mere passing mention.
    * **Exclude:** Do not include passing mentions of subjects that offer nothing significant or substantive to a potential user.
3.  **Concept Analysis:**
    * Analyze the concepts treated in the document so as to produce a series of headings based on its terminology.
    * Indicate relationships between concepts where appropriate.
    * Group together information on subjects that may be scattered throughout the document.

**II. Term Selection & Entry Formulation:**

1.  **Identifying Indexable Topics:** Determine which words, phrases, and concepts are suitable for inclusion in the index.
2.  **Presenting the Topic:** Decide on the most effective way to phrase and present each topic in the index.
3.  **Main Entries (Headings):**
    * **Type:** Main entries **must be nouns** as much as possible.
        * *Example:* Instead of "characteristics of algae", use "Algae, characteristics of".
    * **Relevance:** Must be relevant to the needs of the reader, pertinent, specific, and comprehensive.
    * **Wording:** Be concise and logical. As far as possible, choose terms according to the author's usage.
    * **Exclusions:**
        * Do NOT start index entries with an **article** (e.g., "a", "an", "the").
        * Do NOT start index entries with a **preposition** (e.g., "in", "on", "below").
        * **Verbs** and **adjectives** will **not** be stand-alone main entries. (e.g., "abused", "absorbs" are invalid as main entries).
        * Do NOT index entries that are too general, too narrow, or improbable to be searched by a reader.
4.  **Subheadings (Sub-entries):**
    * **Purpose:** Subheadings are extensions of the main entries.
    * **Qualities:** Should be concise, informative, and have the most important word at the beginning.
        * *Example:* For "Banks", use "Reserve bank regulation" (preferred) over "Regulation of reserve bank".
    * **Allowed:** Prepositions and conjunctions are allowed in sub-levels but should **not** be considered for sorting.
    * **Exclusions:** Avoid terms with acronyms in sub-entries.
5.  **Page Mapping:** Map identified keywords/entries back to the page numbers where they appear in the source document. You will be provided with page-numbered text. Maintain a hierarchical structure, and for each entry, list the corresponding page numbers.

**III. Cross-References:**

Cross-references are crucial for internal navigation within the index.

1.  **"See" Cross-References (Vocabulary Control):**
    * **Function:** Directs users from a term *not used* in the index to the term that *is used* as a heading.
    * **Purpose:** Control scattering of information; anticipate index users' language; reconcile document language with user language.
    * **Usage:** Use whenever it's reasonable that a reader might look up a topic using terminology not chosen for the index (e.g., synonyms, alternative phrasing, abbreviations).
        * *Example:* "American Civil War. See Civil War"
        * *Example:* "War between the States. See Civil War"
        * *Example:* "ABA. See American Bar Association"
2.  **"See also" Cross-References (Related/Additional Information):**
    * **Function:** Guides users to *related* and *additional* information at another heading.
    * **Usage:** Use when users may expect to find additional or related information under a different but connected term. These may be "two-way".
        * *Example:* "drug trafficking. See also narcotics"
        * *Example:* "narcotics. See also drug trafficking"
    * **Considerations:** Include for: Abbreviations and spelled-out forms; Synonyms, name and pseudonym; Equally important halves of a headword (e.g., "breeding, fish" and "fish breeding"); When multiple synonyms exist, use the most well-known synonym.

**IV. Double Postings:**

* **Rationale:** Provide multiple access points for the same information to cater to different reader search behaviors.
* **Implementation:** If readers may look up a topic in more than one way, create entries for both terms pointing to the same page numbers.
    * *Example 1:*
        * automobiles, 55–60
        * cars, 55–60
    * *Example 2:*
        * Roe v. Wade, 78
        * Wade, Roe v., 78
    * *Example 3:*
        * book contracts / of trade publishers, 34–39
        * trade publishers / book contracts of, 34–39
    * Consider more subtle double postings where information isn't a direct inversion but related concepts that warrant separate entries.

**V. Formatting & Compilation:**

1.  **Page Range Style:** When a topic spans multiple contiguous pages, represent them as a range (e.g., 55–60) instead of listing each page individually (e.g., 55, 56, 57, 58, 59, 60).
2.  **Capitalization:** Follow standard indexing capitalization rules (e.g., capitalize the first letter of main entries, proper nouns).
3.  **Deletion of Repetitive Entries:** Ensure there are no exact duplicate entries (except for intentional double postings as described above).
4.  **Sorting & Alphabetization:** Arrange all entries alphabetically.
5.  **Illustrative Matters:** Note page numbers for tables, figures, boxes, and notes if they are relevant illustrative matters for an index entry (style specific means to align with the document's internal style for referencing these).

**VI. Final Output Format:**

Present the index as a clear, alphabetically sorted list of main entries, with subheadings indented appropriately, and page numbers/ranges listed for each. Include cross-references as specified.

---
**Document Content to Index:**
"""


# --- Streamlit Application ---
st.set_page_config(page_title="LLM-Powered Subject Index Creator", layout="centered")

st.title("🧠 LLM-Powered Subject Index Creator")
st.markdown("Upload your PDF document to generate a subject index using an advanced language model based on custom guidelines.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Initialize the runnable (callable) for the LLM
    llm_runnable = initialize_llm_runnable()

    if llm_runnable:
        with st.spinner("Extracting text from PDF and generating index with LLM... This may take a moment."):
            # 1. Extract text from PDF
            document_content = extract_text_from_pdf(uploaded_file)

            if document_content:
                st.success("Text extracted successfully. Sending to LLM for indexing.")

                # 2. Prepare the messages for the LLM (System and Human message)
                messages = [
                    SystemMessage(content="You are an expert indexer generating a subject index based on provided text and strict guidelines."),
                    HumanMessage(content=INDEXING_PROMPT + document_content)
                ]

                # 3. Call the LLM using the runnable
                try:
                    response_content = llm_runnable.invoke(messages) # Invoke the runnable directly
                    st.subheader("Generated Subject Index (from LLM)")
                    st.markdown(response_content)

                    # Option to download the index
                    st.download_button(
                        label="Download LLM-Generated Index as Text File",
                        data=response_content.encode("utf-8"), # Now response_content is a string and can be encoded
                        file_name="llm_subject_index.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to generate index with LLM: {e}")
                    st.info("Please ensure your `LLMFOUNDRY_TOKEN` is correctly set in `st.secrets` and the API service is accessible.")
            else:
                st.error("Could not extract any content from the PDF. The file might be scanned or corrupted, or there was an issue with text extraction.")
    else:
        st.warning("LLM integration could not be initialized. Please check error messages above.")
else:
    st.info("Please upload a PDF file to begin the indexing process.")

st.markdown("---")
