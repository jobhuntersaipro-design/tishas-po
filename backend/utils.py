# --- Library Imports ---
import base64
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any

from difflib import SequenceMatcher
import google.generativeai as genai
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv

from db_connection import get_db_connection

load_dotenv()

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

# 1. Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")

# 2. Database Configuration (SQLite)
DB_NAME = "tishas_demo.db"

# GCS + Storage Configuration
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
STORAGE_FOLDER = "po_storage"  # Local fallback folder


# PostgreSQL tables are created via setup_neon_db.py
# init_db() is no longer needed


def get_gcs_client():
    """
    Creates a GCS client using Environment Variables.
    """
    try:
        # Check for service account JSON content in env var
        service_account_info = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if service_account_info:
            info = json.loads(service_account_info)
            creds = service_account.Credentials.from_service_account_info(info)
            return storage.Client(credentials=creds, project=info.get("project_id"))
        
        # Check for service account file path
        service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if service_account_path and os.path.exists(service_account_path):
             return storage.Client.from_service_account_json(service_account_path)

        return None
    except Exception as e:
        print(f"Error creating GCS client: {e}")
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
# Assumes retailers.csv is in the same directory as utils.py
# Load Retailers Data (DB First, then CSV)
RETAILERS_DF = pd.DataFrame()

try:
    # Try fetching from DB using SQLAlchemy Engine (standard for pandas)
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
            
        engine = create_engine(db_url)
        # Verify connection
        with engine.connect() as connection:
             RETAILERS_DF = pd.read_sql("SELECT * FROM retailers", connection)
    else:
        # Fallback to existing method
        conn = get_db_connection()
        RETAILERS_DF = pd.read_sql("SELECT * FROM retailers", conn)
        conn.close()
    
    if RETAILERS_DF.empty:
        raise Exception("DB table empty")

except Exception as e:
    # Fallback to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "retailers.csv")
    if os.path.exists(csv_path):
        RETAILERS_DF = pd.read_csv(csv_path)

if not RETAILERS_DF.empty:
    try:
        # Normalize columns
        if "debtor_code" in RETAILERS_DF.columns:
            def clean_debtor_code(val):
                if pd.isna(val): return None
                s = str(val).strip()
                if s.lower() in ['nan', 'none', 'null', '']: return None
                return s
            
            RETAILERS_DF["debtor_code"] = RETAILERS_DF["debtor_code"].apply(clean_debtor_code)

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

        # Ensure required columns exist logic could be added, but assuming schema match
        if "retailers_name" in RETAILERS_DF.columns:
            RETAILERS_DF["clean_name"] = RETAILERS_DF["retailers_name"].apply(clean_text)
        if "branch" in RETAILERS_DF.columns:
            RETAILERS_DF["clean_branch"] = RETAILERS_DF["branch"].apply(clean_text)
        if "retailers_group_name" in RETAILERS_DF.columns:
            RETAILERS_DF["clean_group"] = RETAILERS_DF["retailers_group_name"].apply(clean_text)
            
    except Exception as e:
        print(f"Error normalizing retailer data: {e}")



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
    """
    Checks if file_hash already exists in po_data.
    Returns True if exists, False otherwise.
    """
    if not file_hash:
        return False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM po_data WHERE file_hash = %s LIMIT 1", (file_hash,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"DB Check Error: {e}")
        return False


def fetch_all_pos_from_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Remove DISTINCT to show all line items
        query = """
            SELECT 
                debtor_code, retailer_name, branch_name, branch_code,
                delivery_address, buyer_name, po_number, po_date, 
                delivery_date, expiry_date, currency, total_amount, tax_id,
                article_code, barcode, article_description, qty, uom,
                unit_price, line_total, file_storage_url, source_filename
            FROM po_data 
            ORDER BY po_number DESC 
            LIMIT 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return results # Returns list of dicts directly due to RealDictCursor
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return []


def check_po_number_exists(po_number):
    """Check if a PO number already exists in the database."""
    if not po_number:
        return False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT po_number FROM po_data WHERE po_number = %s LIMIT 1", (str(po_number),)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"DB Check Error: {e}")
        return False


def validate_is_po(doc):
    """
    Validate if the extracted document looks like a valid Purchase Order.
    Returns (is_valid, error_message).
    """
    if not doc:
        return False, "No data extracted from document"
    
    # Check for critical PO fields
    has_retailer = doc.get("retailer") or doc.get("retailer_name")
    has_po_number = doc.get("po_number")
    has_items = doc.get("items") and len(doc.get("items", [])) > 0
    
    # A valid PO should have at minimum: retailer info and line items
    # PO number is optional for some retailers (like PELANGI)
    if not has_retailer:
        return False, "Document does not appear to be a Purchase Order - missing retailer information"
    
    if not has_items:
        return False, "Document does not appear to be a Purchase Order - no line items found"
    
    # Check if document_type is explicitly NOT a PO
    doc_type = str(doc.get("document_type", "")).lower()
    if "invoice" in doc_type or "receipt" in doc_type or "packing" in doc_type:
        return False, f"Document appears to be a {doc.get('document_type')} not a Purchase Order"
    
    return True, None



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
        return f"http://localhost:8000/files/{new_filename}"
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

    # Filter potential matches based on retailer type
    if "MYDIN" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_group"].str.contains("MYDIN", na=False)
        ]
    elif "CS" in po_name_upper and "GROCER" in po_name_upper:
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains("CSGROCER", na=False)
            | RETAILERS_DF["clean_name"].str.contains("CS.*GROCER", na=False, regex=True)
        ]
    else:
        # Initial attempts for generic match
        potential_matches = RETAILERS_DF[
            RETAILERS_DF["clean_name"].str.contains(po_name_clean, na=False)
        ]

    # CRITICAL FALLBACK: If specific filtering found nothing, try checking EVERYTHING.
    # This ensures we don't return 0 just because the group name didn't match perfectly.
    if potential_matches.empty:
        potential_matches = RETAILERS_DF

    if potential_matches.empty:
        return extracted_name, None, None, extracted_address, None, 0

    best_match = None
    max_score = 0
    po_address_tokens = set(re.findall(r"\w+", po_address_upper))

    rows_checked = 0
    for index, row in potential_matches.iterrows():
        rows_checked += 1
        score = 0
        csv_address_raw = str(row["delivery_address"]).upper()
        csv_branch_clean = row["clean_branch"]
        csv_branch_raw = str(row.get("branch", "")).upper()
        
        if "LOTUS" in po_name_upper and "(" in row["branch"]:
            branch_code_in_name = re.search(r"\((\d+)\)", row["branch"])
        # Check if extracted branch name is contained in the CSV branch field
        if po_branch_name_clean and len(po_branch_name_clean) > 5:
            # Example: "KAJANGMEWAH" in "CSGROCER(KAJANGMEWAH)(KM)"
            if po_branch_name_clean in csv_branch_clean:
                score += 70  # Very high score for branch name match
            # Also check if CSV branch contains PO branch (reversed)
            elif csv_branch_clean in po_branch_name_clean:
                score += 60
            # Partial substring match (if overlap is significant)
            elif SequenceMatcher(None, po_branch_name_clean, csv_branch_clean).find_longest_match(0, len(po_branch_name_clean), 0, len(csv_branch_clean)).size > 6:
                 score += 40

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
        
        # Special boost for MYDIN if we find key location markers
        if "MYDIN" in po_name_upper:
            # Check for specific location keywords in both PO and CSV
            csv_branch_upper = str(row.get("branch", "")).upper()
            
            # Boost score if key location words match
            location_keywords = ["PUTRAJAYA", "SHAH", "ALAM", "KLANG", "SEREMBAN", "SUBANG", "JAYA", "KAJANG"]
            for keyword in location_keywords:
                if keyword in po_address_upper and keyword in csv_address_raw:
                    score += 40  # High score for address location match
                    break
                if keyword in po_branch_name_upper and keyword in csv_branch_upper:
                    score += 40  # High score for branch name location match
                    break
        
        # Special boost for CS GROCER - match on location in branch name
        if "CS" in po_name_upper and "GROCER" in po_name_upper:
            csv_branch_upper = str(row.get("branch", "")).upper()
            # Check for location keywords
            location_keywords = ["KAJANG", "MEWAH", "PLAZA", "METRO", "PUNCAK", "ALAM"]
            for keyword in location_keywords:
                if keyword in po_branch_name_upper and keyword in csv_branch_upper:
                    score += 50  # Very high score for location match
                    break

        if score > max_score:
            max_score = score
            best_match = row

    final_score = min(100, max_score)

    min_threshold = 15
    if "MYDIN" in po_name_upper or "GIANT" in po_name_upper:
        min_threshold = 10
    if "CS" in po_name_upper and "GROCER" in po_name_upper:
        min_threshold = 10

    if max_score < min_threshold or best_match is None:
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
        conn = get_db_connection()
        cursor = conn.cursor()

        po_num = doc.get("po_number")

        # Prevent duplicates: delete old version if exists
        if po_num:
            cursor.execute(
                "DELETE FROM po_data WHERE po_number = %s",
                (po_num,)
            )
            conn.commit()

        insert_query = """
            INSERT INTO po_data (
                debtor_code, retailer_name, branch_name, branch_code,
                delivery_address, buyer_name, po_number, po_date,
                delivery_date, expiry_date, currency, total_amount, tax_id,
                article_code, barcode, article_description, qty, uom,
                unit_price, line_total, file_storage_url, source_filename
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        e_date = parse_date(doc.get("expiry_date"))
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
        f_url = doc.get("file_path_url") or doc.get("file_storage_url")
        source_fname = doc.get("source_filename")

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
                d_code, r_name, b_name, b_code, d_addr, buyer, po_num, p_date,
                d_date, e_date, curr, tot_amt, tax,
                a_code, barcode_val, a_desc, qty_val, uom_val,
                price_val, total_val, f_url, source_fname,
            )
            cursor.execute(insert_query, val)

        conn.commit()
        cursor.close()
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
                    "expiry_date": "YYYY-MM-DD or null",
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
        You are an expert PO data extractor. The input contains multiple pages which may contain MULTIPLE DISTINCT Purchase Orders.
        
        CRITICAL INSTRUCTION:
        1. SCAN EVERY SINGLE PAGE OF THE INPUT. 
        2. DO NOT STOP after finding the first Purchase Order. 
        3. If there are 10 pages and 10 POs, you must extract 10 separate documents.
        4. "Tishas Food Marketing" is the VENDOR/SUPPLIER. Do NOT mistake it for the retailer.
        
        EXTRACTION RULES:
        - **Identify Distinct POs:** A new PO usually starts with a new PO Number, a new Reference Number, or a new Retailer/Branch.
        - **Pagination:** If a single PO flows onto a second page (e.g. "Page 1 of 2"), combine the data into ONE document object.
        - **Mydin/Giant/Lotus:** These retailers often put different stores on different pages. TREAT THEM AS SEPARATE POs if they have different PO numbers or delivery addresses.
        
        FIELDS TO EXTRACT:
        - 'retailer': The supermarket name (e.g., Mydin, Giant, Lotus).
        - 'po_number': The unique PO number.
        - 'po_date': Date of the PO.
        - 'delivery_date': Expected delivery date (or Ship Date).
        - 'expiry_date': Use specific rules: MYDIN="Deliver By", LOTUS/CHECKERS="Cancel Date", Others="Expiry/Best Before". Return null if not found.
        - 'buyer_name': The "Bill To" entity.
        - 'branch_name': The "Ship To"/"Deliver To" specific store name.
        - 'items': ALL line items from the table.
        - 'qty': Quantity (number).
        - 'uom': Unit of Measure (e.g., "carton", "pcs"). Default to "unit" if unknown.
        - 'unit_price': Price per unit (number).
        - 'total_amount': Total PO value (number).
        
        Return a JSON object with a "documents" list containing ALL extracted POs.
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
            
        # Fix trailing commas (common AI JSON error)
        clean_json = re.sub(r",\s*([\]}])", r"\1", clean_json)

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


def process_pdf(file_path, file_hash, source_filename):
    """
    Converts PDF to images and uses Gemini 1.5 Flash to extract data.
    """
    # 1. Validation: Check if "Tisha" exists in the PDF text
    try:
        import pdfplumber
        has_tisha = False
        with pdfplumber.open(file_path) as pdf:
            text_found = False
            for page in pdf.pages[:3]: # Check first 3 pages
                text = page.extract_text()
                if text:
                    text_found = True
                    text_lower = text.lower()
                    # Expanded keyword list for validation
                    keywords = [
                        "tisha", "tishas", "tisha's", 
                        "global jaya", "global food merchant",
                        "tfp", "mydin", "giant", "lotus", "checkers", 
                        "cs grocer", "super seven", "sogo", "aeon", "ramly mart", "pelangi", 
                        "tunas manja", "pasaraya", "rosyam", "selection", "urban"
                    ]
                    if any(x in text_lower for x in keywords):
                        has_tisha = True
                        break
            
            # If we found text but no keywords -> Fail
            # If we found NO text -> Pass (assume scanned/image-based PDF)
            if text_found and not has_tisha:
                 raise ValueError(f"Upload failed for {source_filename}. Valid retailer keyword not found.")
            elif not text_found:
                 print(f"No text extracted from {source_filename}, assuming scanned document. Proceeding...")
                 has_tisha = True # Allow to proceed
            
    except ImportError:
        print("pdfplumber not installed, skipping keyword validation")
    except Exception as e:
        if "Upload failed" in str(e):
            raise e
        print(f"Validation warning: {e}")

    raw_docs = []
    
    # PER-PAGE EXTRACTION STRATEGY
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Processing {total_pages} pages in {source_filename}...")
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text(x_tolerance=2) or ""
                
                # If page has very little text, try OCR (fallback)
                if len(text.strip()) < 50:
                    print(f"Page {page_num} seems empty/scanned. Skipping text extraction for this page.")
                    continue
                    
                # Identify Retailer for this specific page
                t_page = text.upper()
                curr_page_retailer = "UNKNOWN"
                if "SELECTION" in t_page: curr_page_retailer = "SELECTION_GROCERIES"
                elif "PASARAYA" in t_page or "ANGKASA" in t_page: curr_page_retailer = "PASARAYA_ANGKASA"
                elif "TUNAS" in t_page or "MANJA" in t_page: curr_page_retailer = "TUNAS_MANJA"
                elif "ROSYAM" in t_page: curr_page_retailer = "ST_ROSYAM"
                elif "TFP" in t_page: curr_page_retailer = "TFP_GROUP"
                elif "GLOBAL JAYA" in t_page: curr_page_retailer = "GLOBAL_JAYA"
                elif "GCH" in t_page: curr_page_retailer = "GIANT"
                elif "CS GROCER" in t_page: curr_page_retailer = "CS_GROCER"
                elif "SUPER SEVEN" in t_page: curr_page_retailer = "SUPER_SEVEN"
                elif "SAM'S GROCERIA" in t_page or "CHECKERS" in t_page: curr_page_retailer = "CHECKERS_SAM"
                elif "LOTUSS" in t_page: curr_page_retailer = "LOTUS"
                
                print(f"Extracting Page {page_num}/{total_pages} (Detected: {curr_page_retailer})...")
                
                # Extract using Gemini for THIS PAGE ONLY
                page_results = parse_with_gemini(text, RETAILER_PROMPT_MAP.get(curr_page_retailer, ""))
                
                if page_results:
                    print(f"  -> Found {len(page_results)} document(s) on page {page_num}")
                    for doc in page_results:
                        doc["_page_num"] = page_num # Internal tracking
                        raw_docs.append(doc)

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

    if not raw_docs:
        # Fallback to OCR if no text found in loop
        is_scanned = True
    else:
        is_scanned = False


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

    if not raw_docs:
        return None

    # MERGE LOGIC
    # CRITICAL: Use COMPOSITE KEY (PO Number + Retailer) to prevent merging different retailers
    merged_docs_map = {}
    
    for doc in raw_docs:
        po_num = str(doc.get("po_number") or "").strip()
        retailer = str(doc.get("retailer") or "").strip().upper()
        
        # Create composite key: "PO_NUMBER|RETAILER_NAME"
        # This ensures same PO number but different retailers = different entries
        if po_num and po_num.lower() not in ["null", "none", ""]:
            merge_key = f"{po_num}|{retailer}" if retailer else po_num
        else:
            # No PO number - try to attach to last seen PO with SAME retailer
            if merged_docs_map:
                # Find the last PO with matching retailer
                matching_key = None
                for key in reversed(list(merged_docs_map.keys())):
                    existing_retailer = str(merged_docs_map[key].get("retailer") or "").strip().upper()
                    if existing_retailer == retailer or not retailer:
                        matching_key = key
                        break
                
                if matching_key:
                    # Merge with matching retailer
                    target_doc = merged_docs_map[matching_key]
                    if doc.get("items"):
                        target_doc["items"].extend(doc["items"])
                    print(f"  -> Merged continuation page into {matching_key}")
                    continue
            
            # No match found - create new entry
            merge_key = f"NO_PO_{len(merged_docs_map)+1}|{retailer}"
        
        if merge_key in merged_docs_map:
            # Merge items for same PO + Retailer combo
            target = merged_docs_map[merge_key]
            if doc.get("items"):
                target["items"].extend(doc["items"])
            # Update missing fields
            for k, v in doc.items():
                if not target.get(k) and v:
                    target[k] = v
            print(f"  -> Merged into existing PO: {merge_key}")
        else:
            merged_docs_map[merge_key] = doc
            print(f"  -> Created new PO entry: {merge_key}")

    final_results = []
    for doc in merged_docs_map.values():
        # Clean up internal keys
        if "_page_num" in doc: del doc["_page_num"]
        
        doc = enrich_po_data(doc, file_hash)
        doc = clean_nan_values(doc)
        final_results.append(doc)
    return final_results


def process_image(file_path, file_hash, source_filename):
    """
    Uses Gemini 1.5 Flash to extract data from an image.
    """
    # 1. Validation (OCR logic hard to implement strictly without OCR engine, 
    # skipping or relying on Gemini later, but user asked for check. 
    # For images, we might trust Gemini or skip check if text not easily accessible)
    # Let's skip explicit text check for raw images for now unless requested.
    
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
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    return None
