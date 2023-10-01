import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PyPDF2 import PdfReader
import io
import os
import cv2
import pytesseract
import regex as re

url = 'https://www.sec.gov/Archives/edgar/vprr/index.html'
def fetchdata(url):
    base_url = 'https://www.sec.gov/Archives/edgar/vprr/'
    if not os.path.exists('downloaded_pdfs_2'):
        os.makedirs('downloaded_pdfs_2')
    files = []
    headers = {'User-Agent': "PostmanRuntime/7.32.1"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and (href.startswith("/Archives/edgar/vprr/00") or href.startswith("/Archives/edgar/vprr/01")):
            file_url = urljoin(base_url, href)
            files.append(file_url)
    #uncomment the below line if you are using windows
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    for pdf_url in files:
        response = requests.get(pdf_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and (href.endswith(".pdf")):
                pdf_response = requests.get(urljoin(base_url, href), headers=headers)
                pdf_file_name = os.path.basename(urljoin(base_url, href))
                on_fly_mem_obj = io.BytesIO(pdf_response.content)
                pdf_file = PdfReader(on_fly_mem_obj)

                for count, image_file_object in enumerate(pdf_file.pages[0].images):
                    image_data = image_file_object.data
                    image_name = f"{pdf_file_name}_image_{count}.jpg"
                    image_path = os.path.join('downloaded_pdfs_2', image_name)
                    with open(image_path, "wb") as fp:
                        fp.write(image_data)
                    image = cv2.imread(image_path)
                    extracted_text = pytesseract.image_to_string(image, lang='eng')
                    if "form" in extracted_text.lower():
                        match = re.search(r'form\s+(\S+)', extracted_text, re.IGNORECASE)
                        if match:
                            form_name = match.group(1)
                            max_folder_name_length = 500
                            form_folder = os.path.join('downloaded_pdfs_withmatch', f"form_{form_name[:max_folder_name_length]}")
                            if not os.path.exists(form_folder):
                                os.makedirs(form_folder)
                            os.rename(image_path, os.path.join(form_folder, image_name))
                    else:
                        form_folder = os.path.join('un_downloaded_pdfs_withmatch')
    return "data was sucessfull added"