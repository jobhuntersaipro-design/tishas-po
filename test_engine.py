# --- Library Imports ---
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from io import BytesIO

import google.generativeai as genai
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st  # Critical for Secrets
from google.cloud import storage
from google.oauth2 import service_account  # Critical for GCS Auth
from pdf2image import convert_from_path
from PIL import Image

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

# 1. Configure Gemini AI
# Try fetching from secrets (Cloud), fallback to hardcoded string (Local) if needed
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    # Fallback for local testing if secrets.toml isn't set up
    # Replace this string with your actual key if testing locally
    genai.configure(api_key="YOUR_LOCAL_API_KEY_HERE")

# 2. Database Configuration (SQLite)
DB_NAME = "tishas_demo.db"

# 3. GCS Configuration
GCS_BUCKET_NAME = "tishas-po-storage"
STORAGE_FOLDER = "po_storage"  # Local fallback folder


def init_db():
    """Initializes the SQLite database if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS po_staging (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debtor_code TEXT,
            retailer_name TEXT,
            branch_name TEXT,
            branch_code TEXT,
            delivery_address TEXT,
            buyer_name TEXT,
            po_number TEXT,
            po_date TEXT,
            delivery_date TEXT,
            currency TEXT,
            total_amount REAL,
            tax_id TEXT,
            article_code TEXT,
            barcode TEXT,
            article_description TEXT,
            qty REAL,
            uom TEXT,
            unit_price REAL,
            line_total REAL,
            status TEXT,
            file_hash TEXT,
            file_storage_url TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# Initialize DB immediately on load
init_db()


def get_gcs_client():
    """
    Creates a GCS client using Streamlit Secrets.
    """
    try:
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            creds = service_account.Credentials.from_service_account_info(key_dict)
            return storage.Client(credentials=creds, project=key_dict["project_id"])
        else:
            return None
    except Exception as e:
        print(f"Error creating GCS client from secrets: {e}")
        return None


# ==========================================
# 1. RETAILER SPECIFIC PROMPTS
# ==========================================
RETAILER_PROMPT_MAP = {
    "TFP_GROUP": "Refer to the Ship-to Address to know which branches. | Quantity using packet | Tax id is in after SST Registration No. at the right side, example: 'W10-1808-32000353'.",
    "MYDIN": "Refer to the Address on the PO to know which branches.",
    "CHECKERS_SAM": "Refer to the Delivery Address to know which branches. Extract UOM from U/M column. Example: 'KTK (36)' should be written as 'ktk (36 units each)'",
    "TUNAS MANJA": "Refer to the text in parentheses () beside the retailer name for different branches.",
    "GIANT": "Look for 'Store Code' or 'Site Code' near the address. | 'branch_name' should be the specific facility name (e.g., 'KAJANG FRESH DISTRIBUTION CENTRE'). | CRITICAL: Extract 'qty' from Total Order Qty and set the 'uom' to a descriptive string like 'case (X units each)' where X is from Case Qty.",
    "CS_GROCER": "Refer to top of the address block to know which branches.",
    "LOTUS": "This PO may be multi-page, ensure all items are extracted. | CRITICAL: Extract 'branch_name' from the line starting with 'STORE NAME:' (e.g., 'Lotus's Shah Alam DC...'). | Extract 'qty' as the NUMBER OF CASES and set the 'uom' to a descriptive string like 'case (X units each)' where X is the packing size.",
    "PASARAYA_ANGKASA": "Quantity must be calculated as packet count.",
    "ST_ROSYAM": "Refer to the text beside the retailer name for branches.",
    "SOGO": "Quantity must be calculated as packet count.",
    "SELECTION_GROCERIES": "Quantity must be calculated as packet count.",
    "SUPER_SEVEN": "Refer to the Address on the PO to know which branches. | Article Code is NOT present in this file, return null for article_code (do not use Barcode).",
    "PASARAYA_DARUSSALAM": "Refer to the text in parentheses () beside the retailer name for different branches.",
    "URBAN_MARKETPLACE": "Refer to top of the address block to know which branches.",
    "GLOBAL_JAYA": "NO Barcode is expected for this retailer.",
    "RAMLY_MART": "NO Barcode is expected for this retailer.",
    "PELANGI": "NO PO number for this retailer. Always leave it as blank, even if 'one' or 'zero' is detected.",
    "UNKNOWN": "",
}

# --- Load Retailer CSV ---
RETAILERS_DF = None
try:
    RETAILERS_DF = pd.read_csv("retailers.csv")
    RETAILERS_DF["debtor_code"] = RETAILERS_DF["debtor_code"].astype(str)
    RETAILERS_DF["debtor_code"] = RETAILERS_DF["debtor_code"].where(
        pd.notnull(RETAILERS_DF["debtor_code"]), None
    )

    def clean_branch_code_val(val):
        if pd.isna(val) or str(val).strip() == "":
            return None
        s = str(val).strip()
        if s.endswith(".0"):
            return s[:-2]
        return s

    if "branch_code" in RETAILERS_DF.columns:
        RETAILERS_DF["branch_code"] = RETAILERS_DF["branch_code"].apply(
            clean_branch_code_val
        )

    def clean_text(text):
        return re.sub(r"[^A-Z0-9]", "", str(text).upper())

    RETAILERS_DF["clean_name"] = RETAILERS_DF["retailers_name"].apply(clean_text)
    RETAILERS_DF["clean_branch"] = RETAILERS_DF["branch"].apply(clean_text)
    RETAILERS_DF["clean_group"] = RETAILERS_DF["retailers_group_name"].apply(clean_text)
    print(">> Successfully loaded retailers.csv for mapping.")
except Exception:
    print("!! [WARNING] retailers.csv not found or invalid. Mapper will be disabled.")


# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================


def empty_to_none(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s == "" or s in ["null", "none", "na", "n/a", "unknown", "one", "zero", "0"]:
        return None
    return value


def clean_tax_id(tax_id):
    if tax_id is None:
        return None
    s = str(tax_id).strip()
    if (
        s == ""
        or s.replace("0", "") == ""
        or s.upper() in ["NA", "N/A", "NONE", "NULL"]
    ):
        return None
    return s


def parse_date(date_string):
    formats = [
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d %b %Y",
        "%d-%b-%Y",
        "%d. %B %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%b-%y",
        "%d/%m/%y",
        "%d %B %Y",
        "%Y/%m/%d",
    ]
    if not date_string:
        return None
    date_string = str(date_string).strip().replace(",", "")
    date_string_clean = date_string.replace(".", "").replace("-", " ")
    for fmt in formats:
        try:
            return datetime.strptime(date_string_clean, fmt).strftime("%Y-%m-%d")
        except:
            continue
    match_iso = re.search(r"(\d{4}-\d{2}-\d{2})", date_string)
    if match_iso:
        return match_iso.group(1)
    return None


def clean_nan_values(data):
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif pd.isna(data):
        return None
    return data


def check_if_exists(file_hash):
    if not file_hash:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM po_staging WHERE file_hash = ? LIMIT 1", (file_hash,)
        )
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"DB Check Error: {e}")
        return False


def fetch_all_pos_from_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        query = "SELECT * FROM po_staging ORDER BY id DESC LIMIT 2000"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return pd.DataFrame()


# ==========================================
# 3. STORAGE & UPLOAD FUNCTIONS
# ==========================================


def save_local_copy(temp_file_path, original_filename):
    """Fallback if cloud upload fails."""
    if not os.path.exists(STORAGE_FOLDER):
        os.makedirs(STORAGE_FOLDER)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", original_filename)
    new_filename = f"{timestamp}_{clean_name}"
    dest_path = os.path.join(STORAGE_FOLDER, new_filename)

    try:
        shutil.copy(temp_file_path, dest_path)
        return os.path.abspath(dest_path)
    except Exception as e:
        print(f"Error saving local copy: {e}")
        return None


def upload_to_gcs(local_file_path, original_filename):
    """
    Uploads to GCS using the original filename.
    Organizes files into folders named by their content hash to prevent conflicts.
    """
    try:
        client = get_gcs_client()
        if not client:
            print("GCS Client unavailable. Using local save.")
            return save_local_copy(local_file_path, original_filename)

        bucket = client.bucket(GCS_BUCKET_NAME)

        # 1. Generate Hash to ensure unique folder path
        with open(local_file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # 2. Create a clean filename (remove spaces/special characters)
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", original_filename)

        # 3. Use the hash as a folder, but keep original name as the file
        # Result: "a1b2c3d4.../my_po_file.pdf"
        blob_name = f"{file_hash}/{clean_name}"
        blob = bucket.blob(blob_name)

        # 4. Upload if it doesn't exist
        if not blob.exists():
            print(f"Uploading: {blob_name}")
            blob.upload_from_filename(local_file_path)
        else:
            print(f"File already exists at: {blob_name}")

        return blob.public_url

    except Exception as e:
        print(f"GCS Upload Error: {e}")
        return save_local_copy(local_file_path, original_filename)


# ==========================================
# 4. RETAILER MAPPING LOGIC
# ==========================================


def get_standard_retailer_info(
    extracted_name, extracted_address, extracted_branch_name=None
):
    if RETAILERS_DF is None:
        return extracted_name, None, None, extracted_address, None, 0

    po_name_upper = str(extracted_name).upper()
    po_address_upper = str(extracted_address).upper()
    po_branch_name_upper = (
        str(extracted_branch_name).upper() if extracted_branch_name else ""
    )

    po_name_clean = re.sub(r"[^A-Z0-9]", "", po_name_upper)
    po_address_clean = re.sub(r"[^A-Z0-9]", "", po_address_upper)
    po_branch_name_clean = re.sub(r"[^A-Z0-9]", "", po_branch_name_upper)

    potential_matches = pd.DataFrame()

    if "GIANT" in po_name_upper or "GCH RETAIL" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_group"].str.contains("GIANT", na=False)
            | RETAILERS_DF["clean_name"].str.contains("GCHRETAIL", na=False)
        ]
    elif "CS GROCER" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains("CSGROCER", na=False)
        ]
    elif "MYDIN" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_group"].str.contains("MYDIN", na=False)
        ]
    elif "LOTUS" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_group"].str.contains("LOTUS", na=False)
        ]
    elif "SUPER SEVEN" in po_name_upper or "SUPER 7" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains("SUPERSEVEN", na=False)
        ]
    elif "CHECKERS" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains("CHECKERS", na=False)
        ]
    else:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains(po_name_clean, na=False)
        ]

    if potential_matches.empty:
        return extracted_name, None, None, extracted_address, None, 0

    best_match = None
    max_score = 0
    po_address_tokens = set(re.findall(r"\w+", po_address_upper))

    for index, row in potential_matches.iterrows():
        score = 0
        csv_address_raw = str(row["delivery_address"]).upper()
        csv_branch_clean = row["clean_branch"]

        if po_branch_name_clean and len(po_branch_name_clean) > 5:
            if (
                po_branch_name_clean in csv_branch_clean
                or csv_branch_clean in po_branch_name_clean
            ):
                score += 60

        if "LOTUS" in po_name_upper and "(" in row["branch"]:
            branch_code_in_name = re.search(r"\((\d+)\)", row["branch"])
            if branch_code_in_name:
                code_val = branch_code_in_name.group(1)
                if code_val in po_branch_name_upper or code_val in po_address_upper:
                    score += 55

        unique_location = str(row.get("branch", "")).upper()
        unique_location = (
            re.sub(r"[^A-Z0-9]", "", unique_location)
            .replace(row["clean_name"], "")
            .strip()
        )
        if len(unique_location) > 4 and unique_location in (
            po_name_clean + po_address_clean
        ):
            score += 50

        csv_tokens = set(re.findall(r"\w+", csv_address_raw))
        common = po_address_tokens.intersection(csv_tokens)
        sig_csv = {t for t in csv_tokens if len(t) > 3}
        sig_common = {t for t in common if len(t) > 3}

        if len(sig_csv) > 0:
            ratio = len(sig_common) / len(sig_csv)
            if ratio > 0.4:
                score += 40
            elif ratio > 0.2:
                score += 20

        if score > max_score:
            max_score = score
            best_match = row

    final_score = min(100, max_score)

    if max_score < 20 or best_match is None:
        return extracted_name, None, None, extracted_address, None, 0

    code = best_match["debtor_code"]
    final_debtor_code = str(code) if pd.notnull(code) else None
    csv_branch_code = (
        str(best_match.get("branch_code"))
        if pd.notnull(best_match.get("branch_code"))
        else None
    )

    return (
        best_match["retailers_name"],
        final_debtor_code,
        csv_branch_code,
        best_match["delivery_address"],
        best_match["branch"],
        final_score,
    )


def enrich_po_data(po_data, file_hash=None):
    extracted_name = po_data.get("retailer") or "UNKNOWN"
    extracted_branch_addr = po_data.get("delivery_address")
    extracted_branch_name = po_data.get("branch_name")
    extracted_buyer_name_raw = po_data.get("buyer_name")

    (
        standard_name,
        debtor_code,
        csv_branch_code,
        official_address,
        standard_branch_name,
        reliability_score,
    ) = get_standard_retailer_info(
        extracted_name, extracted_branch_addr, extracted_branch_name
    )

    if debtor_code:
        po_data["debtor_code"] = debtor_code
    if standard_name:
        po_data["retailer_name"] = standard_name
        po_data["retailer_name_standardized"] = standard_name
    else:
        po_data["retailer_name"] = extracted_name
        po_data["retailer_name_standardized"] = extracted_name

    if standard_branch_name:
        po_data["branch_name"] = standard_branch_name
    else:
        if extracted_branch_name:
            po_data["branch_name"] = extracted_branch_name
        elif extracted_branch_addr:
            po_data["branch_name"] = extracted_branch_addr

    if csv_branch_code:
        po_data["branch_code"] = csv_branch_code

    if official_address:
        po_data["delivery_address"] = official_address
    elif not po_data.get("delivery_address"):
        po_data["delivery_address"] = extracted_branch_addr

    po_data["reliability_score"] = reliability_score
    po_data["file_hash"] = file_hash
    po_data["buyer_name"] = extracted_buyer_name_raw
    po_data["tax_id"] = clean_tax_id(po_data.get("tax_id"))

    try:
        po_data["total_amount"] = float(po_data.get("total_amount", 0.0))
    except:
        po_data["total_amount"] = 0.0

    raw_po = str(po_data.get("po_number", ""))
    po_data["po_number"] = raw_po.lstrip("O").lstrip("0").replace("ONO", "").strip()

    if po_data.get("items"):
        for item in po_data["items"]:
            if item.get("uom"):
                item["uom"] = str(item["uom"]).lower()
            try:
                item["unit_price"] = float(item.get("unit_price", 0))
            except:
                item["unit_price"] = 0.0
            try:
                item["total_price"] = float(
                    item.get("total_price", item.get("total", 0))
                )
            except:
                item["total_price"] = 0.0

    return po_data


# ==========================================
# 5. DATABASE SAVE LOGIC (SQLite)
# ==========================================


def save_to_db(doc):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        f_hash = doc.get("file_hash")
        po_num = doc.get("po_number")

        # Prevent duplicates: delete old version if exists
        if f_hash and po_num:
            cursor.execute(
                "DELETE FROM po_staging WHERE file_hash = ? AND po_number = ?",
                (f_hash, po_num),
            )
            conn.commit()

        insert_query = """
            INSERT INTO po_staging (
                debtor_code, retailer_name, branch_name, branch_code,
                delivery_address, buyer_name, po_number, po_date,
                delivery_date, currency, total_amount, tax_id,
                article_code, barcode, article_description, qty, uom,
                unit_price, line_total, status, file_hash, file_storage_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        items = doc.get("items", [])
        d_code = doc.get("debtor_code")
        r_name = doc.get("retailer_name") or doc.get("retailer_name_standardized")
        b_name = doc.get("branch_name") or doc.get("branch_name_standardized")
        b_code = doc.get("branch_code")
        d_addr = doc.get("delivery_address") or doc.get("delivery_address_standardized")
        buyer = doc.get("buyer_name")
        p_date = parse_date(doc.get("po_date"))
        d_date = parse_date(doc.get("delivery_date"))
        curr = doc.get("currency", "MYR")
        tot_amt = doc.get("total_amount")

        if isinstance(tot_amt, str) and " " in tot_amt:
            parts = tot_amt.split(" ")
            if len(parts) >= 2:
                curr = parts[0]
                try:
                    tot_amt = float(parts[1].replace(",", ""))
                except:
                    tot_amt = 0.0

        tax = clean_tax_id(doc.get("tax_id"))
        f_url = doc.get("file_path_url")

        for item in items:
            a_code = (
                item.get("Article Code") or item.get("article_code") or item.get("sku")
            )
            a_desc = (
                item.get("Article Description")
                or item.get("article_description")
                or item.get("description")
            )
            qty_val = (
                item.get("Quantity")
                or item.get("quantity")
                or item.get("qty")
                or item.get("Qty")
            )
            uom_val = item.get("UOM") or item.get("uom")
            price_val = item.get("Unit Price") or item.get("unit_price")
            total_val = (
                item.get("Line Total") or item.get("total_price") or item.get("total")
            )
            barcode_val = item.get("Barcode") or item.get("barcode")

            val = (
                d_code,
                r_name,
                b_name,
                b_code,
                d_addr,
                buyer,
                po_num,
                p_date,
                d_date,
                curr,
                tot_amt,
                tax,
                a_code,
                barcode_val,
                a_desc,
                qty_val,
                uom_val,
                price_val,
                total_val,
                "PENDING",
                f_hash,
                f_url,
            )
            cursor.execute(insert_query, val)

        conn.commit()
        conn.close()
        print(f">> Saved PO {po_num} to DB.")
        return True

    except Exception as e:
        print(f"!! Database Save Error: {e}")
        return False


# ==========================================
# 6. AI PARSER
# ==========================================


def parse_with_gemini(data, extra_instruction=""):
    json_schema = """
        {
            "documents": [
                {
                    "document_type": "Purchase Order",
                    "retailer": "DETECTED_NAME",
                    "po_number": "EXTRACTED_NUM",
                    "po_date": "YYYY-MM-DD",
                    "delivery_date": "YYYY-MM-DD",
                    "currency": "CURRENCY_CODE",
                    "total_amount": NUMBER,
                    "buyer_name": "BUYER_NAME_LEGAL_ENTITY",
                    "delivery_address": "FULL_DELIVERY_ADDRESS",
                    "branch_name": "EXTRACTED_STORE_NAME",
                    "branch_code": "STORE_CODE",
                    "tax_id": "TAX_ID",
                    "items": [
                        {
                            "article_code": "EXTRACTED_CODE",
                            "article_description": "PRODUCT_NAME",
                            "barcode": "EXTRACTED_CODE",
                            "qty": NUMBER,
                            "uom": "UOM_CODE",
                            "unit_price": PRICE_FLOAT,
                            "total_price": TOTAL_PRICE_FLOAT
                        }
                    ]
                }
            ]
        }
    """

    prompt_base = f"""
        You are an expert PO data extractor. Extract ALL distinct POs.
        CRITICAL RULES:
        1. **VENDOR vs RETAILER:** "Tishas Food Marketing" is the VENDOR. Do not extract it as retailer.
        2. **DATES:** Extract 'po_date'. Look for "Delivery Date", "Deliver By", "Required Date", or "Ship Date" for 'delivery_date'. If NOT found, return null.
        3. **BUYER vs BRANCH:**
           - 'buyer_name': Legal Entity (Bill To).
           - 'branch_name': Specific Store/Facility (Ship To).
        4. **SPACING:** Fix mashed words in 'article_description' (e.g. "TISHASROTI" -> "TISHAS ROTI").
        5. **ITEMS:** Extract EVERY row.
        6. - **Unit of Measure (UOM):** If the file doesn't mention it, write 'unit'.
        7. **PRICES:** Return prices as Numbers (floats), not strings.
        8. **BARCODES:** Extract barcode if present (often EAN-13 or similar numbers near description).

        Return JSON only:
        {json_schema}
        IMPORTANT RETAILER INSTRUCTION: {extra_instruction}
    """

    parts = []
    if isinstance(data, list):
        parts = data
        prompt = f"{prompt_base}\n\nAnalyze the image."
    elif isinstance(data, str):
        parts = [data]
        prompt = f"{prompt_base}\n\nDocument Text:\n{data}"

    parts.insert(0, prompt)

    try:
        generation_config = {"response_mime_type": "application/json"}
        model = genai.GenerativeModel(
            "gemini-2.5-flash", generation_config=generation_config
        )
        response = model.generate_content(parts)

        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        if not clean_json.startswith("{") and "{" in clean_json:
            clean_json = clean_json[clean_json.find("{") : clean_json.rfind("}") + 1]

        data = json.loads(clean_json)
        extracted = data.get("documents", [])
        for doc in extracted:
            if not doc.get("currency"):
                doc["currency"] = "MYR"
        return extracted

    except Exception as e:
        print(f"  !! [AI Error]: {e}")
        return None


# ==========================================
# 7. PROCESSORS
# ==========================================


def process_pdf(file_path, file_hash=None):
    text_fragments = []
    is_scanned = False
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2)
                if text:
                    text_fragments.append(text)
        full_text = "\n\n".join(text_fragments)
        if not full_text.strip() or len(full_text) < 50:
            is_scanned = True
    except:
        is_scanned = True

    if is_scanned:
        try:
            images = convert_from_path(file_path)
            image_parts = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_parts.append(
                    {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(buffered.getvalue()).decode("utf-8"),
                    }
                )
            raw_docs = parse_with_gemini(image_parts, "Scanned Document")
        except:
            return None
    else:
        t = full_text.upper()
        curr = "UNKNOWN"
        if "TFP" in t:
            curr = "TFP_GROUP"
        elif "GLOBAL JAYA" in t:
            curr = "GLOBAL_JAYA"
        elif "GCH" in t:
            curr = "GIANT"
        elif "CS GROCER" in t:
            curr = "CS_GROCER"
        elif "SUPER SEVEN" in t:
            curr = "SUPER_SEVEN"
        elif "SAM'S GROCERIA" in t or "CHECKERS" in t:
            curr = "CHECKERS_SAM"
        elif "LOTUSS" in t:
            curr = "LOTUS"
        raw_docs = parse_with_gemini(full_text, RETAILER_PROMPT_MAP.get(curr, ""))

    if not raw_docs:
        return None

    final_results = []
    for doc in raw_docs:
        doc = enrich_po_data(doc, file_hash)
        doc = clean_nan_values(doc)
        final_results.append(doc)
    return final_results


def process_image(file_path, file_hash=None):
    try:
        img = Image.open(file_path)
        mime = Image.MIME.get(img.format, "image/jpeg")
        buffered = BytesIO()
        img.save(buffered, format=img.format or "JPEG")
        image_part = {
            "mime_type": mime,
            "data": base64.b64encode(buffered.getvalue()).decode("utf-8"),
        }
        raw_docs = parse_with_gemini([image_part], "")
        if raw_docs:
            final_results = []
            for doc in raw_docs:
                doc = enrich_po_data(doc, file_hash)
                doc = clean_nan_values(doc)
                final_results.append(doc)
            return final_results
        return None
    except:
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_engine.py <file_path>")
        sys.exit(0)
    file_path = sys.argv[1]
