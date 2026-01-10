"""
Enhanced Document Processor with PaddleOCR
Handles: Images, Tables, Scanned PDFs
"""

import os
import fitz  # PyMuPDF
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import pdfplumber
from typing import List, Dict, Tuple
import logging
import numpy as np
import io

logger = logging.getLogger(__name__)

class EnhancedDocumentProcessor:
    """
    Multi-modal document processor that combines:
    - Standard text extraction
    - OCR for images and scanned pages
    - Table detection and parsing
    """
    
    def __init__(self, use_gpu=True, lang='en'):
        """
        Initialize OCR and table processors
        
        Args:
            use_gpu: Whether to use GPU for OCR (faster)
            lang: Language code ('en', 'vi', or 'ch' for mixed EN+VI)
        """
        logger.info("Initializing Enhanced Document Processor...")
        
        # PaddleOCR supports: 'en', 'vi', 'ch' (Chinese+English), etc.
        # For medical docs with mixed EN+VI, use 'ch' (multi-language mode)
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # Enable rotation detection
            lang=lang,  # Language
            use_gpu=use_gpu,  # GPU acceleration
            show_log=False,  # Reduce verbose output
            det_db_score_mode='fast',  # Faster detection
            use_space_char=True,  # Preserve spaces
        )
        
        logger.info(f"✅ PaddleOCR initialized (lang={lang}, gpu={use_gpu})")
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract all content from PDF using hybrid approach:
        1. Try standard text extraction
        2. Use OCR for image-heavy/scanned pages
        3. Extract tables separately
        
        Returns:
            List of dicts with: {page, text, tables, images, method}
        """
        results = []
        
        try:
            # Open PDF with PyMuPDF for image detection
            pdf_doc = fitz.open(pdf_path)
            
            # Also open with pdfplumber for table extraction
            with pdfplumber.open(pdf_path) as pdf_plumber:
                
                for page_num in range(len(pdf_doc)):
                    page_data = {
                        'page': page_num + 1,
                        'text': '',
                        'tables': [],
                        'images_text': [],
                        'method': 'hybrid'
                    }
                    
                    # === 1. STANDARD TEXT EXTRACTION ===
                    fitz_page = pdf_doc[page_num]
                    standard_text = fitz_page.get_text()
                    
                    # === 2. TABLE EXTRACTION ===
                    plumber_page = pdf_plumber.pages[page_num]
                    tables = plumber_page.extract_tables()
                    
                    if tables:
                        for table_idx, table in enumerate(tables):
                            # Convert table to markdown format
                            table_md = self._table_to_markdown(table)
                            page_data['tables'].append({
                                'index': table_idx,
                                'content': table_md,
                                'raw': table
                            })
                        logger.info(f"  📊 Page {page_num+1}: Found {len(tables)} table(s)")
                    
                    # === 3. IMAGE OCR ===
                    # Check if page has images or is image-based (scanned)
                    image_list = fitz_page.get_images()
                    
                    if image_list or self._is_scanned_page(standard_text, fitz_page):
                        # Extract page as image and run OCR
                        ocr_text = self._ocr_page(fitz_page)
                        page_data['images_text'].append(ocr_text)
                        logger.info(f"  🖼️ Page {page_num+1}: OCR extracted {len(ocr_text)} chars")
                    
                    # === 4. COMBINE ALL CONTENT ===
                    # Priority: Standard text > OCR text > Tables
                    combined_parts = []
                    
                    if standard_text.strip():
                        combined_parts.append(f"=== TEXT CONTENT ===\n{standard_text}")
                    
                    if page_data['images_text']:
                        combined_parts.append(f"\n=== OCR EXTRACTED TEXT ===\n" + 
                                            "\n".join(page_data['images_text']))
                    
                    if page_data['tables']:
                        tables_text = "\n\n".join([
                            f"=== BẢNG {t['index']+1} ===\n{t['content']}" 
                            for t in page_data['tables']
                        ])
                        combined_parts.append(f"\n{tables_text}")
                    
                    page_data['text'] = "\n".join(combined_parts)
                    results.append(page_data)
            
            pdf_doc.close()
            logger.info(f"✅ Processed {len(results)} pages from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise
        
        return results
    
    def _ocr_page(self, fitz_page) -> str:
        """
        Convert PDF page to image and run OCR
        """
        try:
            # Render page to image (higher DPI = better quality but slower)
            pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom = 144 DPI
            img_data = pix.tobytes("png")
            
            # Convert to numpy array for PaddleOCR
            img = Image.open(io.BytesIO(img_data))
            img_np = np.array(img)
            
            # Run OCR
            result = self.ocr.ocr(img_np, cls=True)
            
            # Extract text from result
            if result and result[0]:
                text_lines = [line[1][0] for line in result[0]]
                return "\n".join(text_lines)
            return ""
            
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ""
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table array to markdown format
        
        Example input:
        [['Header1', 'Header2'], ['Value1', 'Value2']]
        
        Output:
        | Header1 | Header2 |
        |---------|---------|
        | Value1  | Value2  |
        """
        if not table:
            return ""
        
        # Clean None values
        table = [[str(cell) if cell else '' for cell in row] for row in table]
        
        # Build markdown
        md_lines = []
        
        # Header row
        if table:
            md_lines.append("| " + " | ".join(table[0]) + " |")
            md_lines.append("|" + "|".join(['---' for _ in table[0]]) + "|")
        
        # Data rows
        for row in table[1:]:
            md_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md_lines)
    
    def _is_scanned_page(self, text: str, fitz_page) -> bool:
        """
        Heuristic to detect if a page is scanned (image-based)
        """
        # If very little text but page has size, likely scanned
        if len(text.strip()) < 50 and fitz_page.rect.width > 0:
            return True
        
        # Check text-to-image ratio
        images = fitz_page.get_images()
        if len(images) > 0 and len(text.strip()) < 200:
            return True
        
        return False


