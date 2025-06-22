# python attack_models/research_papers/pdf_to_text_markdown.py --input path/to/paper.pdf

#!/usr/bin/env python3
"""
PDF to Text Markdown Converter
This script extracts all text content from a PDF (including text from tables, equations, diagrams)
and creates a single markdown file.
"""

import os
import argparse
import fitz  # PyMuPDF
import re
import camelot  # For table extraction
import tabulate  # For table formatting

def extract_text_by_pages(pdf_path):
    """Extract text from PDF page by page"""
    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages_text.append({
                "page": page_num + 1,
                "content": text,
                "type": "text"
            })
        
        return pages_text
    except Exception as e:
        print(f"Error extracting text by pages: {e}")
        return []

def extract_tables_as_text(pdf_path):
    """Extract tables from PDF and convert to text format"""
    table_texts = []
    
    try:
        # Extract tables using camelot
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        
        if len(tables) == 0:
            # Try stream flavor if lattice didn't find any tables
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        
        print(f"Found {len(tables)} tables in PDF")
        
        for i, table in enumerate(tables):
            # Get table page number
            page_num = table.page
            
            # Convert table to markdown
            md_table = tabulate.tabulate(table.df.values.tolist(), headers=table.df.columns.tolist(), tablefmt="pipe")
            
            # Create a reference for markdown
            caption = f"Table {i+1} from page {page_num}"
            table_texts.append({
                "page": page_num,
                "content": f"**{caption}**\n\n{md_table}\n\n",
                "type": "table"
            })
        
        return table_texts
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def create_single_markdown(pdf_path, output_path):
    """Create a single markdown file with all text content from the PDF"""
    try:
        # Get base filename without extension
        base_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        
        # Extract text by pages
        pages_text = extract_text_by_pages(pdf_path)
        
        # Extract tables as text
        table_texts = extract_tables_as_text(pdf_path)
        
        # Combine all elements
        all_elements = []
        
        # Add text elements
        if pages_text:
            all_elements.extend(pages_text)
        
        # Add table elements
        if table_texts:
            all_elements.extend(table_texts)
        
        # Sort all elements by page number
        all_elements.sort(key=lambda x: x["page"])
        
        # Create the unified markdown file
        with open(output_path, "w", encoding="utf-8") as f:
            # Add metadata header
            f.write(f"# {base_name}\n\n")
            f.write("## Document Information\n\n")
            f.write(f"- **Source**: {pdf_path}\n")
            f.write(f"- **Pages**: {len(pages_text)}\n")
            f.write(f"- **Tables**: {len(table_texts)}\n\n")
            
            # Add content by page
            current_page = 0
            
            for element in all_elements:
                # Add page header if we're on a new page
                if element["page"] > current_page:
                    current_page = element["page"]
                    f.write(f"\n\n## Page {current_page}\n\n")
                
                # Add element content based on type
                if element["type"] == "text":
                    f.write(element["content"])
                    f.write("\n\n")
                elif element["type"] == "table":
                    f.write(element["content"])
        
        print(f"Created single markdown file at {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating markdown: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to a single text-only markdown file')
    parser.add_argument('--input', type=str, default="/home/016649880@SJSUAD/Multi-modal-Self-instruct/attack_models/research_papers/1_Towards_Deep_Learning_Models_Resistant_to_Adversarial_Attacks.pdf",
                        help='Path to input PDF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output markdown file (default: input_filename_text_only.md)')
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.basename(args.input).rsplit('.', 1)[0]
        output_dir = os.path.dirname(args.input)
        args.output = os.path.join(output_dir, f"{base_name}_text_only.md")
    
    print("Starting PDF to text-only markdown conversion...")
    
    # Create single markdown file with all text content
    output_path = create_single_markdown(args.input, args.output)
    
    if output_path:
        print(f"\nConversion complete! Text-only markdown available at: {output_path}")
    else:
        print("\nConversion process encountered errors.")

if __name__ == "__main__":
    main()
