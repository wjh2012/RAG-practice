import uuid
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


class PDFProcessor:
    def __init__(self):
        self._ocr_reader = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def ocr_reader(self):
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
        return self._ocr_reader

    def process(self, file_path: Path) -> tuple[list[dict], int, str]:
        document_id = uuid.uuid4().hex[:12]
        filename = file_path.name
        raw_blocks: list[dict] = []

        doc = fitz.open(str(file_path))
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            meta_base = {
                "document_id": document_id,
                "filename": filename,
                "page": page_num + 1,
            }

            text = page.get_text("text").strip()
            if text:
                raw_blocks.append(
                    {"text": text, "metadata": {**meta_base, "content_type": "text"}}
                )

            images = page.get_images(full=True)
            for img_info in images:
                xref = img_info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                    ocr_text = self._ocr_image(img_bytes)
                    if ocr_text.strip():
                        raw_blocks.append(
                            {
                                "text": f"[이미지 OCR]\n{ocr_text}",
                                "metadata": {**meta_base, "content_type": "image_ocr"},
                            }
                        )
                except Exception:
                    continue

        doc.close()

        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    md = self._table_to_markdown(table)
                    if md.strip():
                        raw_blocks.append(
                            {
                                "text": f"[테이블]\n{md}",
                                "metadata": {
                                    "document_id": document_id,
                                    "filename": filename,
                                    "page": i + 1,
                                    "content_type": "table",
                                },
                            }
                        )

        chunks: list[dict] = []
        for block in raw_blocks:
            if block["metadata"]["content_type"] == "text":
                splits = self.text_splitter.split_text(block["text"])
                for split in splits:
                    chunks.append({"text": split, "metadata": {**block["metadata"]}})
            else:
                chunks.append(block)

        return chunks, page_count, document_id

    def _ocr_image(self, img_bytes: bytes) -> str:
        results = self.ocr_reader.readtext(img_bytes, detail=0)
        return "\n".join(results)

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        if not table or not table[0]:
            return ""
        cleaned = [
            [str(cell) if cell is not None else "" for cell in row] for row in table
        ]
        header = "| " + " | ".join(cleaned[0]) + " |"
        separator = "| " + " | ".join("---" for _ in cleaned[0]) + " |"
        rows = "\n".join("| " + " | ".join(row) + " |" for row in cleaned[1:])
        return f"{header}\n{separator}\n{rows}"
