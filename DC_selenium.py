from openpyxl import load_workbook
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import requests
import pdfplumber
import os


# Function to extract hyperlink from a cell
def extract_hyperlink(cell):
    if cell.hyperlink:
        return cell.hyperlink.target
    else:
        return None


# Function to extract text from the top right of the first page of a PDF
def extract_top_right_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        first_page = pdf.pages[0]
        top_right = (
            first_page.width / 2,
            0,
            first_page.width,
            first_page.height / 4
        )
        text = first_page.crop(top_right).extract_text()
        return text


# Load the workbook and the specific sheet
excel_file_path = 'C:/Users/gsanr/PycharmProjects/LDA_topic_modeling/export (1) completely sorted - year wise updated-2 - Copy.xlsx'
workbook = load_workbook(filename=excel_file_path)
sheet = workbook.active  # or workbook['SheetName']

# Extract hyperlinks from the 'Titel' column (assuming it's in the first column)
urls = [extract_hyperlink(cell) for cell in sheet['A']]

# Set up the WebDriver
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

top_right_text_data = []

for url in urls:
    try:
        if url is None:
            raise ValueError("No URL found")

        driver.get(url)

        # Find the PDF link element
        pdf_link_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h2.result--title.icon.icon--large.icon--publication > a"))
        )
        pdf_link = pdf_link_element.get_attribute('href')

        response = requests.get(pdf_link)
        temp_pdf_path = 'C:/Users/gsanr/PycharmProjects/LDA_topic_modeling/temp_pdf.pdf'
        with open(temp_pdf_path, 'wb') as f:
            f.write(response.content)

        top_right_text = extract_top_right_text(temp_pdf_path)
        top_right_text_data.append(top_right_text)

        # Optionally delete the temporary PDF file
        os.remove(temp_pdf_path)

    except (NoSuchElementException, TimeoutException):
        print(f"PDF link not found for URL {url}")
        top_right_text_data.append("PDF Link Not Found")
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        top_right_text_data.append("Error")

    # Sleep to prevent too rapid requests
    time.sleep(1)

# Add the extracted text data to a DataFrame
df = pd.DataFrame({
    'Titel': urls,
    'TopRightText': top_right_text_data
})

# Save the updated DataFrame to a new Excel file
updated_excel_file_path = 'C:/Users/gsanr/PycharmProjects/LDA_topic_modeling/updated_excel.xlsx'
df.to_excel(updated_excel_file_path, index=False)

# Close the WebDriver
driver.quit()




