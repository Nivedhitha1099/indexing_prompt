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
from langchain_openai import ChatOpenAI

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
    
    def _run(self, pdf_path: str) -> str:
        """Extract text from PDF with page numbering"""
        try:
            doc = fitz.open(pdf_path)
            document_text_pages = []
            
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                document_text_pages.append(f"Page {page_num + 1}:\n{text}\n")
            
            doc.close()
            return "\n".join(document_text_pages)
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

class IndexValidatorTool(BaseTool):
    name: str = "Index Validator"
    description: str = "Validates index entries against document content"
    
    def _run(self, index_data: str, document_content: str) -> str:
        """Validate index entries against document content"""
        try:
            # Parse index data (assuming JSON format)
            index_entries = json.loads(index_data)
            validation_results = []
            
            for entry in index_entries:
                term = entry.get('term', '')
                pages = entry.get('pages', [])
                
                # Check if term actually appears on specified pages
                validation_result = {
                    'term': term,
                    'pages': pages,
                    'valid': True,
                    'issues': []
                }
                
                # Simple validation - check if term appears in document
                if term.lower() not in document_content.lower():
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Term '{term}' not found in document")
                
                validation_results.append(validation_result)
            
            return json.dumps(validation_results, indent=2)
        except Exception as e:
            return f"Error validating index: {str(e)}"

class GlossaryExtractorTool(BaseTool):
    name: str = "Glossary Extractor"
    description: str = "Extracts glossary terms and definitions from document"
    
    def _run(self, document_content: str) -> str:
        """Extract glossary terms and definitions"""
        try:
            # Look for glossary sections
            glossary_pattern = r'(?i)glossary|definition|terms'
            lines = document_content.split('\n')
            
            glossary_terms = []
            in_glossary = False
            
            for line in lines:
                if re.search(glossary_pattern, line):
                    in_glossary = True
                    continue
                
                if in_glossary and line.strip():
                    # Simple term extraction (can be enhanced)
                    if ':' in line or '–' in line or '-' in line:
                        parts = re.split(r'[:\-–]', line, 1)
                        if len(parts) == 2:
                            term = parts[0].strip()
                            definition = parts[1].strip()
                            glossary_terms.append({
                                'term': term,
                                'definition': definition
                            })
            
            return json.dumps(glossary_terms, indent=2)
        except Exception as e:
            return f"Error extracting glossary: {str(e)}"

# Fixed LLM Configuration
def initialize_llm():
    """Initialize the LLM with proper CrewAI compatibility"""
    try:
        # Try LLMFoundry first
        token = os.getenv("LLMFOUNDRY_TOKEN")
        if token:
            # Use CrewAI's LLM wrapper for proper compatibility
            return LLM(
                model="anthropic/claude-3-haiku-20240307",
                base_url="https://llmfoundry.straive.com/anthropic/",
                api_key=f'{token}:my-test-project'
            )
        
        # Fallback to standard Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            return LLM(
                model="anthropic/claude-3-haiku-20240307",
                api_key=anthropic_key
            )
        
        return None
        
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        return None

# Alternative LLM initialization for LangChain compatibility
def initialize_lanchain_llm():
    """Initialize the primary LLM (Anthropic Claude), with OpenAI-compatible fallback."""
    try:
        # === Primary Anthropic Setup ===
        token = os.getenv("LLMFOUNDRY_TOKEN")
        if not token:
            raise ValueError("LLMFOUNDRY_TOKEN not set")

        client = anthropic.Anthropic(
            api_key=f"{token}:my-test-project",
            base_url="https://llmfoundry.straive.com/anthropic/",
        )

        def invoke_anthropic_api(messages):
            try:
                system_prompt = ""
                user_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        system_prompt = msg.content
                    elif isinstance(msg, HumanMessage):
                        user_messages.append({"role": "user", "content": msg.content})

                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096,
                    messages=user_messages,
                    system=system_prompt or None
                )

                if isinstance(response.content, list):
                    return "".join([
                        block.text for block in response.content
                        if hasattr(block, 'text') and block.type == 'text'
                    ])
                return str(response.content)

            except anthropic.AnthropicError as e:
                print(f"[Anthropic error - fallback triggered] {e}")
                raise e

        return RunnableLambda(invoke_anthropic_api)

    except Exception as anthropic_error:
        print(f"[Anthropic failed, falling back to gpt-4o-mini] {anthropic_error}")

        try:
            return ChatOpenAI(
                openai_api_base="https://llmfoundry.straive.com/openai/v1/",
                openai_api_key=f'{os.environ["LLMFOUNDRY_TOKEN"]}:my-test-project',
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=4096
            )
        except Exception as fallback_error:
            st.error(f"Both Anthropic and fallback OpenAI LLMs failed: {fallback_error}")
            return None
# Agents with proper LLM configuration
def create_pdf_processor_agent(llm):
    """Agent responsible for PDF processing and text extraction"""
    return Agent(
        role='PDF Processing Specialist',
        goal='Extract and structure text content from PDF documents with accurate page numbering',
        backstory="""You are an expert in document processing and text extraction. 
        You specialize in converting PDF documents into structured text while maintaining 
        page references and document structure. You ensure high-quality text extraction 
        even from complex layouts and scanned documents.""",
        tools=[PDFProcessorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_index_generator_agent(llm):
    """Agent responsible for generating subject indexes"""
    return Agent(
        role='Subject Index Generator',
        goal='Generate comprehensive subject indexes following professional indexing standards',
        backstory="""You are a professional indexer with expertise in creating subject indexes 
        for academic and professional publications. You follow strict indexing guidelines, 
        understand the difference between main entries and subentries, and can create 
        appropriate cross-references. You ensure indexes are user-friendly and comprehensive.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_glossary_agent(llm):
    """Agent responsible for glossary extraction and validation"""
    return Agent(
        role='Glossary Specialist',
        goal='Extract, validate, and organize glossary terms and definitions',
        backstory="""You are a glossary and terminology expert who specializes in 
        identifying key terms and their definitions within documents. You can extract 
        existing glossaries, identify missing terms, and ensure consistency across 
        editions. You understand the importance of accurate terminology in educational materials.""",
        tools=[GlossaryExtractorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_qa_reviewer_agent(llm):
    """Agent responsible for quality assurance and review"""
    return Agent(
        role='QA Review Specialist',
        goal='Review and validate index entries for accuracy and completeness',
        backstory="""You are a quality assurance expert specializing in index validation. 
        You meticulously check index entries against source documents, validate page 
        references, identify inconsistencies, and ensure adherence to style guidelines. 
        You provide detailed feedback for improvements.""",
        tools=[IndexValidatorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

def create_structure_analyst_agent(llm):
    """Agent responsible for analyzing document structure and previous indexes"""
    return Agent(
        role='Structure Analysis Expert',
        goal='Analyze document structure and previous index patterns to maintain consistency',
        backstory="""You are an expert in document structure analysis and indexing patterns. 
        You can identify different types of index entries (names, companies, subjects), 
        analyze previous edition indexes, and establish consistent formatting and 
        organizational rules for new indexes.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Tasks
def create_pdf_processing_task(agent, pdf_content):
    """Task for processing PDF and extracting structured text"""
    return Task(
        description=f"""
        Process the uploaded PDF document and extract structured text with page numbering.
        
        Document content: {pdf_content[:1000]}...
        
        Requirements:
        1. Extract all text content with accurate page numbering
        2. Maintain document structure and formatting
        3. Handle any OCR requirements for scanned content
        4. Prepare text for indexing analysis
        
        Output: Structured text with page numbers clearly marked
        """,
        agent=agent,
        expected_output="Structured text content with page numbers and document hierarchy"
    )

def create_structure_analysis_task(agent, document_content, previous_index=None):
    """Task for analyzing document structure and previous index patterns"""
    return Task(
        description=f"""
        Analyze the document structure and identify indexing patterns.
        
        Document content: {document_content[:1000]}...
        Previous index (if available): {previous_index or 'None provided'}
        
        Requirements:
        1. Identify different types of content (names, companies, subjects)
        2. Analyze previous index structure and patterns
        3. Establish consistent formatting rules
        4. Identify key themes and concepts
        
        Output: Structure analysis report with indexing guidelines
        """,
        agent=agent,
        expected_output="Document structure analysis and indexing pattern recommendations"
    )

def create_index_generation_task(agent, document_content, structure_analysis):
    """Task for generating the subject index"""
    return Task(
        description=f"""
        Generate a comprehensive subject index based on the document content and structure analysis.
        
        Document content: {document_content[:1000]}...
        Structure analysis: {structure_analysis}
        
        Follow these indexing guidelines:
        1. Create main entries as nouns
        2. Include appropriate subentries
        3. Add cross-references where needed
        4. Ensure accurate page numbering
        5. Follow alphabetical ordering
        6. Include double postings for important terms
        
        Output: Complete subject index in standard format
        """,
        agent=agent,
        expected_output="Complete subject index with main entries, subentries, and cross-references"
    )

def create_glossary_extraction_task(agent, document_content):
    """Task for extracting and organizing glossary terms"""
    return Task(
        description=f"""
        Extract and organize glossary terms from the document.
        
        Document content: {document_content[:1000]}...
        
        Requirements:
        1. Identify all glossary terms and definitions
        2. Find terms that should be in glossary but aren't
        3. Check for term consistency across chapters
        4. Organize terms alphabetically
        5. Validate definitions for accuracy
        
        Output: Complete glossary with terms and definitions
        """,
        agent=agent,
        expected_output="Organized glossary with terms, definitions, and validation notes"
    )

def create_qa_review_task(agent, index_content, document_content):
    """Task for quality assurance review"""
    return Task(
        description=f"""
        Review and validate the generated index for accuracy and completeness.
        
        Index content: {index_content}
        Document content: {document_content[:1000]}...
        
        Requirements:
        1. Validate all page references
        2. Check for missing important terms
        3. Verify cross-references are accurate
        4. Ensure consistent formatting
        5. Identify any errors or improvements needed
        
        Output: QA report with validation results and recommendations
        """,
        agent=agent,
        expected_output="Comprehensive QA report with validation results and improvement recommendations"
    )

# Main Indexing Automation Class
class IndexingAutomation:
    def __init__(self):
        self.llm = initialize_llm()
        if self.llm:
            self.agents = self._initialize_agents()
        else:
            self.agents = None
        self.results = {}
    
    def _initialize_agents(self):
        """Initialize all agents"""
        return {
            'pdf_processor': create_pdf_processor_agent(self.llm),
            'structure_analyst': create_structure_analyst_agent(self.llm),
            'index_generator': create_index_generator_agent(self.llm),
            'glossary_specialist': create_glossary_agent(self.llm),
            'qa_reviewer': create_qa_reviewer_agent(self.llm)
        }
    
    def process_document(self, pdf_content: str, previous_index: str = None) -> Dict[str, Any]:
        """Main processing pipeline with error handling"""
        try:
            if not self.agents:
                return {'error': 'Agents not initialized. Check LLM configuration.'}
            
            # Phase 1: PDF Processing and Structure Analysis
            pdf_task = create_pdf_processing_task(
                self.agents['pdf_processor'], 
                pdf_content
            )
            
            structure_task = create_structure_analysis_task(
                self.agents['structure_analyst'], 
                pdf_content, 
                previous_index
            )
            
            # Phase 1 Crew
            crew_phase1 = Crew(
                agents=[
                    self.agents['pdf_processor'],
                    self.agents['structure_analyst']
                ],
                tasks=[pdf_task, structure_task],
                process=Process.sequential,
                verbose=True
            )
            
            try:
                phase1_results = crew_phase1.kickoff()
                
                # Extract results safely
                document_content = str(phase1_results.tasks_output[0].raw) if phase1_results.tasks_output else pdf_content
                structure_analysis = str(phase1_results.tasks_output[1].raw) if len(phase1_results.tasks_output) > 1 else "Basic structure analysis"
                
            except Exception as e:
                st.warning(f"Phase 1 error: {e}. Using fallback processing.")
                document_content = pdf_content
                structure_analysis = "Basic structure analysis due to processing error"
            
            # Phase 2: Index and Glossary Generation
            index_task = create_index_generation_task(
                self.agents['index_generator'],
                document_content,
                structure_analysis
            )
            
            glossary_task = create_glossary_extraction_task(
                self.agents['glossary_specialist'],
                document_content
            )
            
            crew_phase2 = Crew(
                agents=[
                    self.agents['index_generator'],
                    self.agents['glossary_specialist']
                ],
                tasks=[index_task, glossary_task],
                process=Process.sequential,  # Changed from parallel for better stability
                verbose=True
            )
            
            try:
                phase2_results = crew_phase2.kickoff()
                
                # Extract results safely
                index_content = str(phase2_results.tasks_output[0].raw) if phase2_results.tasks_output else "Index generation failed"
                glossary_content = str(phase2_results.tasks_output[1].raw) if len(phase2_results.tasks_output) > 1 else "Glossary extraction failed"
                
            except Exception as e:
                st.warning(f"Phase 2 error: {e}. Using fallback processing.")
                index_content = "Index generation failed due to processing error"
                glossary_content = "Glossary extraction failed due to processing error"
            
            # Phase 3: Quality Assurance
            qa_task = create_qa_review_task(
                self.agents['qa_reviewer'],
                index_content,
                document_content
            )
            
            crew_phase3 = Crew(
                agents=[self.agents['qa_reviewer']],
                tasks=[qa_task],
                process=Process.sequential,
                verbose=True
            )
            
            try:
                phase3_results = crew_phase3.kickoff()
                qa_report = str(phase3_results.tasks_output[0].raw) if phase3_results.tasks_output else "QA review failed"
                
            except Exception as e:
                st.warning(f"Phase 3 error: {e}. Using fallback processing.")
                qa_report = "QA review failed due to processing error"
            
            # Compile final results
            self.results = {
                'document_content': document_content,
                'structure_analysis': structure_analysis,
                'index_content': index_content,
                'glossary_content': glossary_content,
                'qa_report': qa_report,
                'timestamp': datetime.now().isoformat()
            }
            
            return self.results
            
        except Exception as e:
            st.error(f"Error in processing pipeline: {str(e)}")
            return {'error': str(e)}
    
    def export_results(self, format_type: str = 'json') -> str:
        """Export results in specified format"""
        if format_type == 'json':
            return json.dumps(self.results, indent=2)
        elif format_type == 'txt':
            output = f"""
INDEXING AUTOMATION RESULTS
Generated: {self.results.get('timestamp', 'Unknown')}

=== SUBJECT INDEX ===
{self.results.get('index_content', 'No index generated')}

=== GLOSSARY ===
{self.results.get('glossary_content', 'No glossary generated')}

=== QA REPORT ===
{self.results.get('qa_report', 'No QA report generated')}

=== STRUCTURE ANALYSIS ===
{self.results.get('structure_analysis', 'No structure analysis generated')}
"""
            return output
        else:
            return str(self.results)

# Streamlit Application
def main():
    st.set_page_config(
        page_title="CrewAI Indexing Automation",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 CrewAI Indexing Automation System")
    st.markdown("Multi-agent system for automated document indexing and glossary generation")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment check
    has_llm = (os.getenv("LLMFOUNDRY_TOKEN") or 
               os.getenv("ANTHROPIC_API_KEY"))
    
    if not has_llm:
        st.sidebar.error("Please set one of: LLMFOUNDRY_TOKEN or ANTHROPIC_API_KEY")
        st.error("No LLM configuration found. Please set up your API keys.")
        return
    
    # Display LLM status
    llm_status = "✅ LLMFoundry" if os.getenv("LLMFOUNDRY_TOKEN") else "✅ Anthropic"
    st.sidebar.info(f"LLM Provider: {llm_status}")
    
    # File upload
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    
    # Previous index upload (optional)
    previous_index_file = st.file_uploader("Upload Previous Index (Optional)", type=['txt', 'json'])
    
    if uploaded_file is not None:
        # Initialize automation system
        automation = IndexingAutomation()
        
        if automation.llm is None:
            st.error("LLM initialization failed. Please check your configuration.")
            return
        
        # Process previous index if provided
        previous_index = None
        if previous_index_file is not None:
            previous_index = previous_index_file.read().decode('utf-8')
        
        # Extract PDF content
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text
                pdf_tool = PDFProcessorTool()
                pdf_content = pdf_tool._run(temp_path)
                
                # Clean up temp file
                os.remove(temp_path)
                
                if "Error" in pdf_content:
                    st.error(f"PDF processing failed: {pdf_content}")
                    return
                
                st.success("PDF processed successfully!")
                
                # Show preview
                with st.expander("📄 Document Preview"):
                    st.text_area("Document Content", pdf_content[:2000] + "..." if len(pdf_content) > 2000 else pdf_content, height=200)
                
                # Process with CrewAI
                if st.button("🚀 Start Indexing Automation"):
                    with st.spinner("Running multi-agent indexing system..."):
                        results = automation.process_document(pdf_content, previous_index)
                        
                        if 'error' in results:
                            st.error(f"Processing failed: {results['error']}")
                        else:
                            st.success("Indexing automation completed!")
                            
                            # Display results in tabs
                            tab1, tab2, tab3, tab4 = st.tabs([
                                "📋 Subject Index", 
                                "📚 Glossary", 
                                "🔍 QA Report", 
                                "📊 Structure Analysis"
                            ])
                            
                            with tab1:
                                st.subheader("Generated Subject Index")
                                st.text_area("Index Content", results.get('index_content', ''), height=400, key="index_content")
                            
                            with tab2:
                                st.subheader("Extracted Glossary")
                                st.text_area("Glossary Content", results.get('glossary_content', ''), height=400, key="glossary_content")
                            
                            with tab3:
                                st.subheader("Quality Assurance Report")
                                st.text_area("QA Report", results.get('qa_report', ''), height=400, key="qa_report")
                            
                            with tab4:
                                st.subheader("Document Structure Analysis")
                                st.text_area("Structure Analysis", results.get('structure_analysis', ''), height=400, key="structure_analysis")
                            
                            # Export options
                            st.header("📥 Export Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                json_output = automation.export_results('json')
                                st.download_button(
                                    label="📄 Download as JSON",
                                    data=json_output,
                                    file_name=f"indexing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            
                            with col2:
                                text_output = automation.export_results('txt')
                                st.download_button(
                                    label="📝 Download as Text",
                                    data=text_output,
                                    file_name=f"indexing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()
