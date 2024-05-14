import PyPDF2

def count_words_in_pdf(pdf_path):
    word_count = 0
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            word_count += len(text.split())
    return word_count

pdf_path = 'your_pdf_file.pdf'  #
word_count = count_words_in_pdf(pdf_path)
print(f'Total words in the PDF: {word_count}')
