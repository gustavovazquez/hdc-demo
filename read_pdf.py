from pypdf import PdfReader
import os
import sys

# Set stdout to handle utf-8
sys.stdout.reconfigure(encoding='utf-8')

def read_pdf():
    path = os.path.join("papers", "2025___Clei_Final (11).pdf")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
        
    print(text)

if __name__ == "__main__":
    read_pdf()
