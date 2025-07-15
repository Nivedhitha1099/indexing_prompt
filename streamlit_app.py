import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import os
import streamlit as st
import fitz  # PyMuPDF
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from langchain_core.tools import BaseTool

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# Additional imports
import anthropic
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Data Models
@dataclass
class IndexEntry:
    term: str
    page_numbers: List[int]
    subentries: List['IndexEntry'] = None
    cross_references: List[str] = None
    entry_type: str = "main"  # main, sub, name, company

@dataclass
class IndexStructure:
    main_entries: List[IndexEntry]
    cross_references: Dict[str, str]
    style_guide: Dict[str, Any]

# Custom Tools with proper Pydantic v2 compatibility
class PDFProcessorTool(BaseTool):
    name: str = "PDF Processor"
    description: str = "Extracts text from PDF files with page numbers"
    
    def _run(self, pdf_path: str, start_page: int = None, end_page: int = None) -> str:
        """Extract text from PDF with page numbering, optionally for a page range"""
        try:
            doc = fitz.open(pdf_path)
            document_text_pages = []
            
            start = start_page if start_page is not None else 0
            end = end_page if end_page is not None else doc.page_count
            
            for page_num in range(start, min(end, doc.page_count)):
                page = doc.load_page(page_num)
                text = page.get_text()
                document_text_pages.append(f"Page {page_num + 1}:\n{text}\n")
            
            doc.close()
            return "\n".join(document_text_pages)
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

# Fixed LLM Configuration
def initialize_llm():
    """Initialize Claude-3-haiku from LLMFoundry (Anthropic-compatible) using CrewAI-compatible LLM wrapper."""
    token = os.getenv("LLMFOUNDRY_TOKEN")
    if not token:
        raise ValueError("LLMFOUNDRY_TOKEN not found in environment variables")

    return LLM(
        model="claude-3-haiku-20240307",
        api_key=f"{token}:my-test-project",
        base_url="https://llmfoundry.straive.com/anthropic/",
        temperature=0.3,
        max_tokens=4096
    )

def create_subject_index_agent(llm):
    """Agent responsible for generating only subject index"""
    return Agent(
        role='Subject Index Generator',
        goal='Generate a clean, comprehensive subject index with accurate page numbers',
        backstory="""You are a professional indexer specializing in creating subject indexes 
        for academic and professional publications. You focus only on subject terms and concepts,
        creating clean, alphabetically organized indexes with accurate page references. 
        You exclude names, companies, and other non-subject entries.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_subject_index_task(agent, document_content, llm_guidelines=""):
    """Task for generating only the subject index"""
    return Task(
        description=f"""{llm_guidelines}

        Generate a clean subject index ONLY from the document content. 
        
        Document content: {document_content[:1000]}...
        
        STRICT REQUIREMENTS:
        1. Generate ONLY subject index entries (concepts, topics, themes)
        2. Do NOT include names of people, companies, or organizations
        3. Use this exact format for each entry:
           Term, page numbers
           Example: "Artificial intelligence, 15, 23, 45"
        4. Sort entries alphabetically with letter headers (A, B, C, etc.)
        5. Format: Letter header on its own line, followed by entries
        6. Use accurate page numbers from the document
        7. Output should be clean with no additional explanations or text
        8. One entry per line under each letter section
        
        OUTPUT FORMAT EXAMPLE:
        A
        Algorithms, 12, 34, 56
        Artificial intelligence, 15, 23, 45
        
        B
        Behavioral analysis, 67, 89
        
        D
        Data processing, 78, 90
        
        M
        Machine learning, 101, 123, 145
        
        Generate the subject index now:
        """,
        agent=agent,
        expected_output="Clean subject index with terms and page numbers only, no additional text or formatting"
    )

# Main Indexing Automation Class
class SubjectIndexAutomation:
    def __init__(self):
        self.llm = initialize_llm()
        if self.llm:
            self.agent = create_subject_index_agent(self.llm)
        else:
            self.agent = None
        self.results = {}

    def process_document(self, pdf_path: str, llm_guidelines: str = "") -> Dict[str, Any]:
        """Main processing pipeline for subject index only"""
        try:
            if not self.agent:
                return {'error': 'Agent not initialized. Check LLM configuration.'}

            # Open PDF to get page count
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()

            page_chunk_size = 20  # Larger chunks since we only need subject index
            aggregated_subject_index = []

            # Use the PDFProcessorTool instance directly for extraction
            pdf_processor_tool = PDFProcessorTool()

            for start_page in range(0, total_pages, page_chunk_size):
                end_page = min(start_page + page_chunk_size, total_pages)
                st.info(f"Processing pages {start_page + 1} to {end_page} of {total_pages}")

                # Extract text for page chunk
                pdf_text_chunk = pdf_processor_tool._run(pdf_path, start_page=start_page, end_page=end_page)

                if pdf_text_chunk.startswith("Error processing PDF"):
                    st.warning(f"PDF processing error on pages {start_page + 1}-{end_page}: {pdf_text_chunk}")
                    continue

                # Create subject index task
                index_task = create_subject_index_task(
                    self.agent,
                    pdf_text_chunk,
                    llm_guidelines
                )

                crew = Crew(
                    agents=[self.agent],
                    tasks=[index_task],
                    process=Process.sequential,
                    verbose=True
                )

                try:
                    results = crew.kickoff()
                    subject_index_content = str(results.tasks_output[0].raw) if results.tasks_output else ""
                    
                    # Clean the output further
                    subject_index_content = self._clean_index_output(subject_index_content)
                    
                    if subject_index_content.strip():
                        aggregated_subject_index.append(subject_index_content)
                        
                except Exception as e:
                    st.warning(f"Processing error on pages {start_page + 1}-{end_page}: {e}")
                    continue

            # Combine and clean final results
            if aggregated_subject_index:
                combined_subject_index = self._merge_index_entries(aggregated_subject_index)
            else:
                combined_subject_index = "No subject index entries generated"

            self.results = {
                'subject_index': combined_subject_index,
                'timestamp': datetime.now().isoformat()
            }

            return self.results

        except Exception as e:
            st.error(f"Error in processing pipeline: {str(e)}")
            return {'error': str(e)}

    def _clean_index_output(self, content: str) -> str:
        """Clean the index output to remove unwanted text"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, headers, explanations
            if not line:
                continue
            if line.lower().startswith(('subject index', 'index:', '===', '---', 'generated:', 'output:', 'here')):
                continue
            if 'example:' in line.lower() or 'format:' in line.lower():
                continue
            # Keep lines that look like index entries (have comma and numbers)
            if ',' in line and any(char.isdigit() for char in line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _merge_index_entries(self, index_chunks: List[str]) -> str:
        """Merge and deduplicate index entries from multiple chunks"""
        all_entries = {}
        
        for chunk in index_chunks:
            lines = chunk.split('\n')
            for line in lines:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                # Parse entry
                if line.startswith('    '):  # Subentry
                    continue  # Handle subentries separately
                
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                
                term = parts[0].strip()
                page_numbers = []
                
                for part in parts[1:]:
                    part = part.strip()
                    if part.isdigit():
                        page_numbers.append(int(part))
                
                if term and page_numbers:
                    if term in all_entries:
                        all_entries[term].extend(page_numbers)
                    else:
                        all_entries[term] = page_numbers
        
        # Sort and format final output with alphabetical headers
        sorted_entries = []
        current_letter = ''
        
        for term in sorted(all_entries.keys(), key=str.lower):
            # Get first letter of term
            first_letter = term[0].upper()
            
            # Add letter header if it's a new letter
            if first_letter != current_letter:
                if sorted_entries:  # Add blank line before new letter (except for first)
                    sorted_entries.append('')
                sorted_entries.append(first_letter)
                current_letter = first_letter
            
            # Format page numbers
            unique_pages = sorted(list(set(all_entries[term])))
            page_str = ', '.join(map(str, unique_pages))
            sorted_entries.append(f"{term}, {page_str}")
        
        return '\n'.join(sorted_entries)
    
    def export_clean_index(self) -> str:
        """Export only the clean subject index"""
        return self.results.get('subject_index', 'No subject index generated')

# Streamlit Application
def main():
    st.set_page_config(
        page_title="Subject Index Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Subject Index Generator")
    st.markdown("Generate clean subject indexes from PDF documents")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment check
    has_llm = os.getenv("LLMFOUNDRY_TOKEN")
    
    if not has_llm:
        st.sidebar.error("Please set LLMFOUNDRY_TOKEN")
        st.error("No LLM configuration found. Please set up your API key.")
        return
    
    st.sidebar.info("✅ LLMFoundry configured")
    
    # File upload
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    
    # LLM guidelines input
    st.header("✍️ LLM Guidelines (Optional)")
    llm_guidelines = st.text_area(
        "Enter additional instructions for the subject index generation:",
        height=100,
        placeholder="e.g., Focus on technical terms, Include methodology concepts, etc."
    )
    
    if uploaded_file is not None:
        # Initialize automation system
        automation = SubjectIndexAutomation()
        
        if automation.llm is None:
            st.error("LLM initialization failed. Please check your configuration.")
            return
        
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("PDF uploaded successfully!")
        
        # Show preview
        pdf_tool = PDFProcessorTool()
        preview_text = pdf_tool._run(temp_path, start_page=0, end_page=3)
        preview_text = preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text
        with st.expander("📄 Document Preview"):
            st.text_area("Document Content Preview", preview_text, height=200)
        
        # Process document
        if st.button("🚀 Generate Subject Index"):
            with st.spinner("Generating subject index..."):
                results = automation.process_document(temp_path, llm_guidelines)
                
                if 'error' in results:
                    st.error(f"Processing failed: {results['error']}")
                else:
                    st.success("Subject index generated successfully!")
                    
                    # Display results
                    st.subheader("📋 Generated Subject Index")
                    subject_index = results.get('subject_index', '')
                    st.text_area("Subject Index", subject_index, height=400)
                    
                    # Export option
                    st.header("📥 Download Clean Index")
                    clean_output = automation.export_clean_index()
                    st.download_button(
                        label="📄 Download Subject Index",
                        data=clean_output,
                        file_name=f"subject_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass

if __name__ == "__main__":
    main()
