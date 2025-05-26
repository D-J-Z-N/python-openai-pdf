import camelot
import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-nano")

def get_ai_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN environment variable is required")
    return OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token
    )

def extract_all_tables_from_pdf(file_path):
    tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
    all_rows = []
    for table in tables:
        df = table.df
        headers = df.iloc[0].tolist()
        for index, row in df.iterrows():
            if index == 0:
                continue  # Skip header row
            row_data = {headers[i]: row[i] for i in range(len(headers))}
            all_rows.append(row_data)
    return all_rows

def format_table_csv(data):
    if not data:
        return ""
    headers = list(data[0].keys())
    lines = [" ,".join(headers)]
    for row in data:
        lines.append(",".join(row.get(h, "") or "" for h in headers))
    return "\n".join(lines)

def main():
    pdf_path = input("Enter the path to your PDF file: ")
    formatted_data = extract_all_tables_from_pdf(pdf_path)

    if not formatted_data:
        print("No tables found in the PDF.")
        return

    print("Tables loaded successfully. You can now ask questions about them. Type 'exit' to quit.")

    ai_client = get_ai_client()

    csv_table = format_table_csv(formatted_data)
    conversation = [
        {"role": "system", "content": "You are a helpful assistant who answers questions about tabular data."},
        {"role": "user", "content": "Here is tabular data extracted from a PDF:\n" + csv_table}
    ]

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break

        conversation.append({"role": "user", "content": question})

        response = ai_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=conversation,
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()
        print("Answer:", answer)
        conversation.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
