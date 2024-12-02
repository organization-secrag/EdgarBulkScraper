import json
import os
import traceback
import requests
import re
import tarfile
from lxml import etree
from edgar._markdown import html_to_markdown
from edgar.xbrl import XBRLData, XBRLInstance
from edgar.xbrl.statements import BalanceSheet, IncomeStatement, CashFlowStatement, StatementOfChangesInEquity, StatementOfComprehensiveIncome
from edgar.financials import Financials
import pandas as pd
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from markdownify import markdownify as md
import concurrent.futures
# URL of the website to send the request to
import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="A simple command-line tool with year and quarter arguments")

# Optional argument for year (e.g., 2024)
parser.add_argument("-y", "--year", type=int, help="Year for processing", required=True)

# Optional argument for quarter (1-4)
parser.add_argument("-q", "--quarter", type=int, choices=[1, 2, 3, 4], help="Quarter for processing (1-4)", required=True)

parser.add_argument("-t", "--threads", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], help="Threads for Filing processor (1-8) Default: 4", required=False)

parser.add_argument('--debug', action='store_true', help="Run once and then exit. Do not override existing filings")


args = parser.parse_args()



year = str(args.year)
quarter = str(args.quarter)
threads = args.threads if args.threads else 4

temp_path = "temp/"

url = f"https://www.sec.gov/Archives/edgar/Feed/{year}/QTR{quarter}/"

headers = {
    'User-Agent': 'SECRAG admin@secrag.com'
}




from bs4 import BeautifulSoup

def table_to_markdown(table):
    # Initialize the markdown string
    markdown = ""
    table = BeautifulSoup(table, 'html.parser')
    # Extract table headers if available
    headers = table.find_all('th')
    if headers:
        # Add headers to markdown (pipe-separated)
        header_row = '| ' + ' | '.join([header.text.strip() for header in headers]) + ' |'
        markdown += header_row + '\n'
        # Add separator row
        markdown += '| ' + ' | '.join(['---' for _ in headers]) + ' |\n'

    # Extract table rows (excluding headers)
    rows = table.find_all('tr')
    for row in rows:
        # Extract columns (td)
        cols = row.find_all('td')
        if cols:
            # Add data to markdown (pipe-separated)
            row_data = '| ' + ' | '.join([col.text.strip() for col in cols]) + ' |'
            markdown += row_data + '\n'

    return markdown

def extract_table_to_markdown(data, tag_prefix, attribute_name, attribute_value):
    # Send GET request to fetch the HTML content of the page
    soup = BeautifulSoup(data, 'html.parser')

    # Find all tables in the HTML content
    tables = soup.find_all('table')
    target_table = None

    # Regular expression pattern for tags starting with 'ix:'
    tag_pattern = re.compile(f'^{tag_prefix}:')

    # Iterate over all tables to find the one containing the desired attribute and tag
    for table in tables:
        # Check if any element inside the table has the specific attribute with the desired value
        matching_element = table.find_all(lambda tag: tag_pattern.match(tag.name) and tag.get(attribute_name) == attribute_value)

        if matching_element:
            target_table = table
            break  # Stop once the table is found

    if not target_table:
        print(f"No table found containing '{tag_prefix}:' tags with {attribute_name}='{attribute_value}' inside it.")
        return None

    # Convert the found table to markdown
    markdown = table_to_markdown(target_table)

    return markdown


def download_archive(file_name):
    # Send GET request to download the file with streaming enabled
    if os.path.exists(f"temp/{file_name}"):
        print(f"Skipping {file_name} download. Already downloaded!\n")
        return
    print(f"Downloading {file_name}...\n")
    response = requests.get(url + file_name, stream=True, headers=headers)
    downloaded_size = 0
    # Check if the request was successful
    if response.status_code == 200:
        total_size = int(response.headers.get('Content-Length', 0))
        # Open a local file to save the content
        with open("temp/"+file_name, "wb") as file:
            # Write the file content in chunks
            for chunk in response.iter_content(chunk_size=8192):  # 8KB chunks
                downloaded_size += len(chunk)  # Update downloaded size
                file.write(chunk)

                # Calculate and print progress
                if total_size > 0:  # Only show progress if total size is known
                    progress = (downloaded_size / total_size) * 100
                    print(f"\rDownloading {file_name} - {progress:.2f}% ({downloaded_size:,}/{total_size:,} bytes)", end="", flush=True)
                else:
                    print(f"\rDownloading {file_name} - {downloaded_size} bytes", end="", flush=True)
    else:
        raise Exception(response.status_code, response.text)

def extract_archive(file_name):
    extract_path = temp_path+file_name.replace(".tar.gz", "")
    if os.path.exists(extract_path):
        print(f"{extract_path} extract path already exists! Delete the path to trigger extraction.")
        return
    print(f"Extracting {file_name}...\n")
    # extract_path = temp_path+file_name.replace(".tar.gz", "")
    with tarfile.open(temp_path+file_name, "r:gz") as archive:
        # Extract all files to the specified directory
        members = archive.getmembers()
        total_files = len(members)
        
        # Extract files one by one and print progress
        for index, member in enumerate(members):
            archive.extract(member, path=extract_path)
            
            # Print progress: (index + 1) gives the number of files extracted so far
            progress = (index + 1) / total_files * 100
            print(f"\rProgress: {round(progress, 2)}%", end="", flush=True)

def check_type_in_xml(file_path, target_values):
    try:
        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [file.readline().strip() for _ in range(20)]  # Strip newline characters
            content = '\n'.join(lines) 

        # Parse the content as XML using lxml, which is more lenient with malformed XML
        parser = etree.XMLParser(recover=True)  # recover=True helps handle malformed XML
        root = etree.fromstring(content, parser=parser)

        # Iterate through all elements and check if TYPE tag exists
        for elem in root.iter('TYPE'):
            
            value = elem.text.strip() 
            if value in target_values:
                # print(f"Found {target_values} in {file_path}")
                return value
        # print(f"{target_values} not found in {file_path}")
        return False
    except Exception:
        print(f"Error parsing {file_path}: Malformed XML")
        return False

def check_file_in_directory(directory, target_values, dir_size, i, filename):
    file_path = os.path.join(directory, filename)
    progress = round(i * 100 / dir_size, 2)
    # Check if the file is not a directory and is not empty
    if os.path.isfile(file_path):

        print(f"\rReading {filename} | {i+1}/{dir_size} | {progress}%         ", end="", flush=True)
        found_str = check_type_in_xml(file_path, target_values)
        return found_str, file_path
    return None, None

def check_files_in_directory(directory, target_values):
    # Iterate through all files in the directory
    dir_files = os.listdir(directory)
    dir_size = len(dir_files)
    docs_of_interest = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit the tasks to the executor
        results = [executor.submit(check_file_in_directory, directory, target_values, dir_size, i, file) for i, file in enumerate(dir_files)]
        
        # Retrieve the results as they are completed
        for future in concurrent.futures.as_completed(results):
            found_str, file_path = future.result()
            if found_str:
                docs_of_interest.append([file_path, found_str])

            # print(future.result(), display_counter)
    # for i, filename in enumerate(dir_files):
    #     file_path = os.path.join(directory, filename)
    #     progress = round(i * 100 / dir_size, 2)
    #     # Check if the file is not a directory and is not empty
    #     if os.path.isfile(file_path):

    #         print(f"\rReading {filename} | {i+1}/{dir_size} | {progress}%         ", end="", flush=True)
    #         found_str = check_type_in_xml(file_path, target_values)
    #         if (found_str):
    #             docs_of_interest.append([file_path, found_str])

    print(f"\nFound {target_values} in {len(docs_of_interest)} files!")
    return docs_of_interest

def clean_directory(dir_path, allowed_filepaths):
    allowed_files = [file.split("\\")[-1] for file, file_type in allowed_filepaths]
    try:
        # List all files in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            # Check if it's a file and not in the allowed files list
            if os.path.isfile(file_path) and filename not in allowed_files:
                print(f"Removing file: {filename}")
                os.remove(file_path)
            else:
                print(f"NOT REMOVING file: {filename}")
        
        print(f"Directory '{dir_path}' cleaned successfully.")
    except Exception as e:
        print(f"Error cleaning directory: {e}")

xbrl_doc_types = ['instance', 'schema', 'label', 'calculation', 'presentation']
xbrl_doc_codes = ['EX-101.INS', 'EX-101.SCH', 'EX-101.LAB', 'EX-101.CAL', 'EX-101.PRE']


def financials_mds(fin_statements: dict, return_empty: bool = False) -> dict:
    output = {}
    # Generate chunks for financials

    for key, _ in fin_statements.items():
        # Sometimes it returns tuples
        if isinstance(fin_statements[key], tuple):
            # Process each element of a tuple
            for damn_tuple in fin_statements[key]:
                # If item is a dataframe, convert it to markdown
                # Otherwise, just return "no data"
                text = damn_tuple.to_markdown() if type(damn_tuple) is pd.DataFrame and not damn_tuple.empty  else "no data"
                if text == "no data" and not return_empty:
                    continue
                output[key] = text
        else:
            # If item is a dataframe, convert it to markdown
            # Otherwise, just return "no data"
            text = fin_statements[key].to_markdown() if type(fin_statements[key]) is pd.DataFrame and not fin_statements[key].empty  else "no data"
            if text == "no data" and not return_empty:
                continue
            output[key] = text
    
    return output

def try_except(func, soup, concepts, default=None, expected_exc=(Exception,)):
    try: return func()
    except expected_exc: 
        return extract_financial_table_using_concepts(soup, concepts)
    
def extract_financial_table_using_concepts(soup, concepts):
    for i in concepts:
        element = soup.find(lambda tag: tag.name.startswith('ix:') and tag.get('name') == i.concept.replace("_",":"))

        # If the element is found, find its parent table
        if element:
            table = element.find_parent('table')
            
            # Check if the table was found and print it out
            if table:
                return md(str(table))  # .prettify() for formatted output
                # return df.to_markdown()
                # break
    return None

filings_of_interest = ["10-K", "10-Q", "10-K/A", "10-Q/A"]

def convert_to_mds(file_path, filing_type):

    doc_pattern = r'<DOCUMENT.*?>.*?</DOCUMENT>'
    html_pattern = r'<html.*?>.*?</html>'
    html_stupid_pattern = r'<HTML.*?>.*?</HTML>'
    xbrl_exhibit_pattern = r'<TEXT>.*?</TEXT>' # Relaxed
    xbrl_instance_pattern = r'<XBRL>.*?</XBRL>' # Now we process instance XBRL as well
    xbrl_pattern = r'<\?xml.*?>.*?</xbrl>'
    xbrl_schema_pattern = r'<\?xml.*?>.*?</.*?schema>'
    xbrl_parsed_docs = {}
    with open(file_path, "r") as file:
        data = file.read()

    parser = etree.XMLParser(recover=True)  # recover=True helps handle malformed XML
    root = etree.fromstring(data, parser=parser)
    documents = re.findall(doc_pattern, data, re.DOTALL)

    cik = root.xpath('//CIK')[0].text.strip()
    company_name = root.xpath('//CONFORMED-NAME')[0].text.strip()
    filing_date = root.xpath('//FILING-DATE')[0].text.strip()
    filtered_filing_type = filing_type.replace("/", "")
    out_dir = f"output/{cik}_{filing_date}_{filtered_filing_type}.json"
    file_postfix = 1

    if os.path.exists(out_dir):
        out_dir_w_postfix = f"output/{cik}_{filing_date}_{filtered_filing_type}_{file_postfix}.json"
        if args.debug:
            print("Skipping via debug...")
            return
        while os.path.exists(out_dir_w_postfix):
            file_postfix += 1 
            out_dir_w_postfix = f"output/{cik}_{filing_date}_{filtered_filing_type}_{file_postfix}.json"

        out_dir = out_dir_w_postfix


    for document in documents:
        # print(document)
        document_tree = etree.fromstring(document, parser=parser)
        for description in document_tree.iter('TYPE'):
            # print(description.text)
            if description.text == None:
                continue
            processed_description = description.text.strip() 
            if description is not None and processed_description in filings_of_interest:
                # filing_type = processed_description
                # print(f"{cik} | {company_name} | {filing_type} | {filing_date}")

                # Find the "XBRL" tag and print its value
                useful_data = re.findall(html_pattern, document, re.DOTALL)
                # print(type(useful_data))
                if len(useful_data) > 0:
                    # print("FOUND stuff")
                    # print(useful_data[0][:200])
                    md_main_data = html_to_markdown(useful_data[0])
                    main_html = BeautifulSoup(useful_data[0], 'lxml')
                else:
                    useful_data = re.findall(html_stupid_pattern, document, re.DOTALL)
                    md_main_data = html_to_markdown(useful_data[0])
                    main_html = BeautifulSoup(useful_data[0], 'lxml')


                # quit()
            elif description is not None and processed_description in xbrl_doc_codes:

                exhibit_name = xbrl_doc_types[xbrl_doc_codes.index(processed_description)]
                # print("XBRL found:",exhibit_name)
                if exhibit_name == "schema":  
                    xbrl_parsed_docs[exhibit_name] = re.findall(xbrl_schema_pattern, document, re.DOTALL)[0]
                elif exhibit_name == "instance": # First time Instance exhibit occurence resulted in an infinite loop?
                    xbrl_parsed_docs[exhibit_name] = re.findall(xbrl_instance_pattern, document, re.DOTALL)[0]
                else:
                    xbrl_parsed_docs[exhibit_name] = re.findall(xbrl_exhibit_pattern, document, re.DOTALL)[0]
            
            elif processed_description == "XML":
                # print(filing_date)
                # if re.findall(rf"<FILENAME>.*?_htm.xml", document, re.DOTALL):
                #     print("AAAAA") 
                if len(re.findall("XMLSchema-instance", document, re.DOTALL)) != 0 or len(re.findall(rf"<FILENAME>.*?_htm\.xml", document, re.DOTALL)) != 0: 
                    # print("XML XBRL found: instance")
                    xbrl_parsed_docs["instance"] = re.findall(xbrl_pattern, document, re.DOTALL)[0]

            # print(description)                   
    # print(xbrl_parsed_docs.keys())
    if xbrl_parsed_docs and xbrl_parsed_docs.get("presentation"):
        try:

            xbrl_data = XBRLData.from_input(
                instance_xml = xbrl_parsed_docs.get("instance"),
                presentation_xml = xbrl_parsed_docs.get("presentation"),
                label_xml = xbrl_parsed_docs.get("label"),
                calculation_xml = xbrl_parsed_docs.get("calculation")
            )

            financials_obj = Financials(xbrl_data)
            balance_sheet = try_except(lambda: financials_obj.get_balance_sheet().to_dataframe(), main_html, BalanceSheet.concepts)
            income_statement = try_except(lambda: financials_obj.get_income_statement().to_dataframe(), main_html, IncomeStatement.concepts)
            cash_flow_statement = try_except(lambda: financials_obj.get_cash_flow_statement().to_dataframe(), main_html, IncomeStatement.concepts)
            statement_of_comprehensive_income = try_except(lambda: financials_obj.get_statement_of_comprehensive_income().to_dataframe(), main_html, IncomeStatement.concepts)
            statement_of_changes_in_equity = try_except(lambda: financials_obj.get_statement_of_changes_in_equity().to_dataframe(), main_html, IncomeStatement.concepts)

            fin_statements = {
                "balance_sheet": balance_sheet,
                "income_statement": income_statement,
                "cash_flow_statement": cash_flow_statement,
                "statement_of_comprehensive_income": statement_of_comprehensive_income,
                "statement_of_changes_in_equity": statement_of_changes_in_equity
            }

            fin_mds = financials_mds(fin_statements)
        except OutOfBoundsDatetime as e:
            print("\nFailed to process XBRL: ", str(e))
            fin_mds = {}
    else:
        # print(xbrl_instance.facts)
        soup = BeautifulSoup(useful_data[0], "lxml")
        # print("Balance sheet:")

        # print(BalanceSheet.concepts[0].concept)
        # Search for the ix:nonfraction element by its 'name' attribute

        fin_mds = {
            "balance_sheet": str(extract_financial_table_using_concepts(soup, BalanceSheet.concepts)),
            "income_statement": str(extract_financial_table_using_concepts(soup, IncomeStatement.concepts)),
            "cash_flow_statement": str(extract_financial_table_using_concepts(soup, CashFlowStatement.concepts)),
            "statement_of_comprehensive_income": str(extract_financial_table_using_concepts(soup, StatementOfComprehensiveIncome.concepts)),
            "statement_of_changes_in_equity": str(extract_financial_table_using_concepts(soup, StatementOfChangesInEquity.concepts))
        } 

    output_data = {}
    output_data["type"] = filing_type
    output_data["main"] = md_main_data
    output_data["financials"] = fin_mds

    with open(out_dir, "w") as f:
        json.dump(output_data, f, indent=4)
    return out_dir


def try_convert_to_mds(file_path, filing_type):
    try:
        out_dir = convert_to_mds(file_path, filing_type)
        return f"Processed {file_path} to {out_dir}"
    except Exception as e:
        with open("failed_to_process.txt", "a") as f:
            f.write(f"{file_path}\n")
        return f"Failed to process {file_path}: \n{traceback.format_exc()}\n{str(e)}"



if __name__ == "__main__":

    # Send a GET request to the website
    response = requests.get(url, headers=headers)

    files = re.findall("[0-9]*.nc.tar.gz",response.text)

    for file_name in files:
        with open("archive_bookkeeping.txt", "r") as f:
                archives_processed = f.read()
        
        # print(archives_processed)
        
        if file_name in archives_processed:
            continue
        try:
            download_archive(file_name)
            print("\n")
        except Exception as e:
            os.remove(f"temp/{file_name}")
            raise e
        
        
        debug = False
        debug_file = "temp/20240229.nc/0001758766-24-000029.nc"
        debug_file_type = "10-K"
        if debug:
            convert_to_mds(debug_file, debug_file_type)

        else:
            extract_archive(file_name)
            extracted_archive_path = temp_path+file_name.replace(".tar.gz", "")
            docs = check_files_in_directory(extracted_archive_path, filings_of_interest)
            clean_directory(extracted_archive_path, docs)
            
            total_docs = len(docs)
            print(f"Files to process: {total_docs}...")
            display_counter = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                    # Submit the tasks to the executor
                    results = [executor.submit(try_convert_to_mds, doc[0], doc[1]) for doc in docs]
                    
                    # Retrieve the results as they are completed
                    for future in concurrent.futures.as_completed(results):
                        display_counter +=1
                        print(future.result(), display_counter)

            print(f"Processed number of docs: {display_counter} / {total_docs}")
            with open("archive_bookkeeping.txt", "a") as f:
                f.write(file_name+"\n")

            for i in os.listdir(f"{extracted_archive_path}"):
                os.remove(f"{extracted_archive_path}/{i}")
            os.remove(f"temp/{file_name}")

            print(f"Processing {file_name} complete!")
            if args.debug:
                print("Exiting due to debug mode.")
                quit()
            
