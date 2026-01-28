import os
import shutil
import hashlib
import uuid
import threading
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import utils
google_sheets_service = None
google_service_available = False

try:
    import google_sheets_service
    google_service_available = True
    print("SUCCESS: Google Sheets Service loaded.")
except ImportError as e:
    print(f"CRITICAL WARNING: Google Library missing. Service disabled. Error: {e}")
except Exception as e:
    print(f"CRITICAL WARNING: Error loading Google Service: {e}")

app = FastAPI(title="Tishas PO Extractor API")

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://tishas-po.vercel.app"
]

# Add production domains from env
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded PDFs
if not os.path.exists("po_storage"):
    os.makedirs("po_storage")
app.mount("/files", StaticFiles(directory="po_storage"), name="files")

# In-memory storage for batch processing
batch_storage: Dict[str, Dict[str, Any]] = {}
batch_lock = threading.Lock()

# --- Pydantic Models for Input Validation ---

class Item(BaseModel):
    article_code: Optional[str] = None
    article_description: Optional[str] = None
    barcode: Optional[str] = None
    qty: Optional[float] = 0.0
    uom: Optional[str] = None
    unit_price: Optional[float] = 0.0
    total_price: Optional[float] = 0.0
    # Allow loose matching for flexible incoming data
    class Config:
        extra = "allow"

class PODocument(BaseModel):
    id: Optional[int] = None
    retailer_name: Optional[str] = None
    po_number: Optional[str] = None
    po_date: Optional[str] = None
    delivery_date: Optional[str] = None
    expiry_date: Optional[str] = None
    currency: Optional[str] = "MYR"
    total_amount: Optional[Any] = 0.0
    buyer_name: Optional[str] = None
    delivery_address: Optional[str] = None
    billing_address: Optional[str] = None  # NEW: Billing address
    branch_name: Optional[str] = None
    branch_code: Optional[str] = None
    debtor_code: Optional[str] = None
    tax_id: Optional[str] = None
    file_hash: Optional[str] = None
    file_path_url: Optional[str] = None
    source_filename: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = []
    is_flagged: Optional[bool] = False  # NEW: Flag for issues
    flag_reason: Optional[str] = None  # NEW: Reason for flagging
    
    class Config:
        extra = "allow"


def calculate_items_sum(items: List[Dict[str, Any]]) -> float:
    """Calculate the sum of all line item totals."""
    total = 0.0
    for item in items:
        item_total = item.get("total_price") or item.get("Line Total") or item.get("total") or 0
        try:
            total += float(item_total)
        except (ValueError, TypeError):
            pass
    return total


def flag_amount_mismatch(doc: dict) -> dict:
    """Check if line items sum matches total_amount and flag if mismatch."""
    items = doc.get("items", [])
    if not items:
        return doc
    
    items_sum = calculate_items_sum(items)
    total_amount = 0.0
    try:
        total_amount = float(doc.get("total_amount", 0))
    except (ValueError, TypeError):
        pass
    
    # Flag if difference is RM1 or more
    tolerance = 1.0
    difference = abs(items_sum - total_amount)
    
    if difference >= tolerance and total_amount > 0:
        doc["is_flagged"] = True
        doc["flag_reason"] = f"Line items sum ({items_sum:.2f}) differs from total amount ({total_amount:.2f}) by {difference:.2f}"
    else:
        doc["is_flagged"] = False
        doc["flag_reason"] = None
    
    return doc

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Tishas PO Extractor API is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    original_filename = file.filename
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Calculate hash from the saved file to match utils logic expectation or read bytes
        with open(temp_filename, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Upload to Storage (GCS or Local)
        saved_path, storage_existed = utils.upload_to_gcs(temp_filename, original_filename)
        
        # Process File
        file_ext = os.path.splitext(original_filename)[1].lower()
        results = []
        
        if file_ext == ".pdf":
            results = utils.process_pdf(temp_filename, file_hash, original_filename)
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            results = utils.process_image(temp_filename, file_hash, original_filename)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file format")

        if not results:
            raise HTTPException(status_code=400, detail="No data extracted from file. Please make sure it's a valid Purchase Order.")
        
        # Validate each extracted document
        validated_results = []
        for doc in results:
            # Validate if it's a PO
            is_valid, error_msg = utils.validate_is_po(doc)
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg or "Document does not appear to be a valid Purchase Order")
            
            # Add file metadata
            doc["file_path_url"] = saved_path
            doc["source_filename"] = original_filename
            
            # Check for duplicate PO number
            po_number = doc.get("po_number")
            if po_number and utils.check_po_number_exists(po_number):
                doc["already_exists"] = True
                doc["duplicate_message"] = f"PO {po_number} already exists in database"
            else:
                doc["already_exists"] = False
            
            # Flag if line items sum doesn't match total
            doc = flag_amount_mismatch(doc)
            
            validated_results.append(doc)
        
        return {"status": "success", "documents": validated_results}

    except HTTPException as he:
        print(f"HTTP Exception: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/api/save_all")
async def save_all_pos(pos: List[PODocument]):
    """Save multiple PO documents at once"""
    saved_count = 0
    failed_count = 0
    errors = []
    
    for po in pos:
        try:
            doc = po.model_dump()
            success = utils.save_to_db(doc)
            if success:
                saved_count += 1
            else:
                failed_count += 1
                errors.append(f"PO {doc.get('po_number', 'Unknown')} failed to save")
        except Exception as e:
            failed_count += 1
            errors.append(f"Error saving PO: {str(e)}")
    
    if saved_count > 0:
        return {
            "status": "success",
            "message": f"Saved {saved_count} POs successfully" + (f", {failed_count} failed" if failed_count > 0 else ""),
            "saved": saved_count,
            "failed": failed_count,
            "errors": errors if errors else None
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to save any POs. " + "; ".join(errors))

@app.post("/api/save")
async def save_po(po: PODocument):
    # Convert Pydantic model to dict
    doc = po.model_dump()
    
    # Handle potentially nested or non-standard item keys if needed, 
    # but utils.save_to_db expects a dict with keys matching what `utils.parse_with_gemini` creates.
    # The Pydantic model `extra="allow"` helps pass through everything.
    
    success = utils.save_to_db(doc)
    if success:
        return {"status": "success", "message": f"PO {doc.get('po_number')} saved."}
    else:
        raise HTTPException(status_code=500, detail="Failed to save PO to database.")

@app.get("/api/history")
def get_history():
    data = utils.fetch_all_pos_from_db()
    return data


@app.get("/api/export_csv")
def export_csv(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Export POs to CSV with optional date range filter."""
    import csv
    from io import StringIO
    
    data = utils.fetch_pos_by_date_range(start_date, end_date)
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for the specified date range")
    
    # Create CSV in memory
    output = StringIO()
    
    # Get all unique keys from the data
    fieldnames = [
        'po_number', 'retailer_name', 'debtor_code', 'branch_name', 'branch_code',
        'buyer_name', 'delivery_address', 'billing_address', 'po_date', 'delivery_date',
        'expiry_date', 'currency', 'total_amount', 'tax_id', 'article_code', 'barcode',
        'article_description', 'qty', 'uom', 'unit_price', 'line_total', 'source_filename'
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    
    for row in data:
        # Convert row to dict if it's a tuple/list
        if isinstance(row, dict):
            writer.writerow(row)
        else:
            # Assume it's in the same order as fieldnames
            row_dict = dict(zip(fieldnames, row))
            writer.writerow(row_dict)
    
    output.seek(0)
    
    # Generate filename with date range
    filename = "po_export"
    if start_date:
        filename += f"_from_{start_date}"
    if end_date:
        filename += f"_to_{end_date}"
    filename += ".csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/api/check/{file_hash}")
def check_file(file_hash: str):
    exists = utils.check_if_exists(file_hash)
    return {"exists": exists}

@app.get("/api/check_po/{po_number}")
def check_po_number(po_number: str):
    exists = utils.check_po_number_exists(po_number)
    return {"exists": exists}

def process_batch_file(batch_id: str, temp_filename: str, original_filename: str, file_hash: str, saved_path: str):
    """Background task to process file page by page with incremental PO display"""
    
    def finalize_and_add_po(doc, batch_id, file_hash, saved_path, original_filename):
        """Helper to finalize a merged PO and add it to results immediately"""
        # Clean up internal keys
        if "_page_num" in doc:
            del doc["_page_num"]
        
        # DEDUPLICATION: Remove duplicate items based on barcode or article_code
        items = doc.get("items", [])
        if items:
            seen_items = {}
            unique_items = []
            for item in items:
                barcode = str(item.get("barcode") or item.get("Barcode") or "").strip()
                article_code = str(item.get("article_code") or item.get("Article Code") or "").strip()
                item_key = barcode if barcode else article_code
                
                if item_key and item_key in seen_items:
                    print(f"    -> Removed duplicate item: {item_key}")
                    continue
                
                if item_key:
                    seen_items[item_key] = True
                unique_items.append(item)
            
            if len(items) != len(unique_items):
                print(f"    -> Deduplicated: {len(items)} -> {len(unique_items)} items")
            doc["items"] = unique_items
        
        doc = utils.enrich_po_data(doc, file_hash)
        doc = utils.fix_billing_address(doc)
        doc = utils.clean_nan_values(doc)
        doc["file_path_url"] = saved_path
        doc["source_filename"] = original_filename
        
        po_number = doc.get("po_number")
        if po_number and utils.check_po_number_exists(po_number):
            doc["already_exists"] = True
            doc["duplicate_message"] = f"PO {po_number} already exists in database"
        else:
            doc["already_exists"] = False
        
        is_valid, error_msg = utils.validate_is_po(doc)
        if is_valid:
            doc = flag_amount_mismatch(doc)
            # Add to batch results IMMEDIATELY
            with batch_lock:
                batch_storage[batch_id]["pos"].append(doc)
            print(f"  => Added PO {po_number} to results")
            return True
        return False
    
    try:
        import pdfplumber
        
        merged_docs_map = {}  # Track current merging state
        last_merge_key = None  # Track the last PO we're working on
        
        with pdfplumber.open(temp_filename) as pdf:
            total_pages = len(pdf.pages)
            
            with batch_lock:
                batch_storage[batch_id]["total_pages"] = total_pages
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                with batch_lock:
                    batch_storage[batch_id]["current_page"] = page_num
                
                try:
                    text = page.extract_text(x_tolerance=2) or ""
                    
                    if len(text.strip()) < 50:
                        print(f"Page {page_num} seems empty/scanned. Skipping...")
                        continue
                    
                    # Detect retailer
                    t_page = text.upper()
                    curr_retailer = "UNKNOWN"
                    if "MYDIN" in t_page: curr_retailer = "MYDIN"
                    elif "SELECTION" in t_page: curr_retailer = "SELECTION_GROCERIES"
                    elif "DARUSSALAM" in t_page: curr_retailer = "PASARAYA_DARUSSALAM"
                    elif "PASARAYA" in t_page or "ANGKASA" in t_page: curr_retailer = "PASARAYA_ANGKASA"
                    elif "TUNAS" in t_page or "MANJA" in t_page: curr_retailer = "TUNAS_MANJA"
                    elif "ROSYAM" in t_page: curr_retailer = "ST_ROSYAM"
                    elif "GLOBAL JAYA" in t_page: curr_retailer = "GLOBAL_JAYA"
                    elif "GCH" in t_page or "GIANT" in t_page: curr_retailer = "GIANT"
                    elif "CS GROCER" in t_page: curr_retailer = "CS_GROCER"
                    elif "SUPER SEVEN" in t_page: curr_retailer = "SUPER_SEVEN"
                    elif "SAM'S GROCERIA" in t_page or "CHECKERS" in t_page: curr_retailer = "CHECKERS_SAM"
                    elif "LOTUSS" in t_page or "LOTUS" in t_page: curr_retailer = "LOTUS"
                    elif "TFP" in t_page: curr_retailer = "TFP_GROUP"
                    
                    print(f"Extracting Page {page_num}/{total_pages} (Detected: {curr_retailer})...")
                    
                    page_results = utils.parse_with_gemini(text, utils.RETAILER_PROMPT_MAP.get(curr_retailer, ""))
                    
                    if page_results:
                        print(f"  -> Found {len(page_results)} document(s) on page {page_num}")
                        for doc in page_results:
                            print(f"      DEBUG Page {page_num}: PO={doc.get('po_number')}, items={len(doc.get('items', []))}, total={doc.get('total_amount')}")
                            doc["_page_num"] = page_num
                            
                            # Determine merge key
                            po_num = str(doc.get("po_number") or "").strip()
                            retailer = str(doc.get("retailer") or "").strip().upper()
                            
                            if po_num and po_num.lower() not in ["null", "none", ""]:
                                merge_key = f"{po_num}|{retailer}" if retailer else po_num
                            else:
                                # No PO number - attach to last seen PO with same retailer
                                if merged_docs_map:
                                    matching_key = None
                                    for key in reversed(list(merged_docs_map.keys())):
                                        existing_retailer = str(merged_docs_map[key].get("retailer") or "").strip().upper()
                                        if existing_retailer == retailer or not retailer:
                                            matching_key = key
                                            break
                                    if matching_key:
                                        target_doc = merged_docs_map[matching_key]
                                        if doc.get("items"):
                                            target_doc["items"].extend(doc["items"])
                                        print(f"  -> Merged continuation page into {matching_key}")
                                        continue
                                merge_key = f"NO_PO_{len(merged_docs_map)+1}|{retailer}"
                            
                            # Check if this is a NEW PO (different from what we've been merging)
                            if merge_key in merged_docs_map:
                                # Same PO - merge items
                                target = merged_docs_map[merge_key]
                                if doc.get("items"):
                                    target["items"].extend(doc["items"])
                                
                                # Update total_amount if new one is larger
                                new_total = doc.get("total_amount")
                                if new_total:
                                    try:
                                        new_total_float = float(new_total) if isinstance(new_total, str) else new_total
                                        current_total = target.get("total_amount", 0)
                                        current_total_float = float(current_total) if isinstance(current_total, str) else (current_total or 0)
                                        if new_total_float > current_total_float:
                                            target["total_amount"] = new_total_float
                                            print(f"    -> Updated total_amount to {new_total_float}")
                                    except (ValueError, TypeError):
                                        pass
                                
                                for k, v in doc.items():
                                    if k == "total_amount":
                                        continue
                                    if not target.get(k) and v:
                                        target[k] = v
                                print(f"  -> Merged continuation page into: {merge_key}")
                            else:
                                # NEW PO detected - finalize any PREVIOUS POs that are complete
                                # A PO is complete when we encounter a different PO
                                for prev_key in list(merged_docs_map.keys()):
                                    if prev_key != merge_key:
                                        prev_doc = merged_docs_map.pop(prev_key)
                                        finalize_and_add_po(prev_doc, batch_id, file_hash, saved_path, original_filename)
                                
                                # Start tracking the new PO
                                merged_docs_map[merge_key] = doc
                                print(f"  -> Created new PO entry: {merge_key}")
                    else:
                        print(f"  !! No results from Gemini for page {page_num}")
                
                except Exception as page_error:
                    print(f"Error processing page {page_num}: {page_error}")
                    with batch_lock:
                        if "page_errors" not in batch_storage[batch_id]:
                            batch_storage[batch_id]["page_errors"] = []
                        batch_storage[batch_id]["page_errors"].append({
                            "page": page_num,
                            "error": str(page_error)
                        })
                    continue
        
        # Finalize any remaining POs after all pages processed
        for remaining_key, remaining_doc in merged_docs_map.items():
            finalize_and_add_po(remaining_doc, batch_id, file_hash, saved_path, original_filename)
        
        with batch_lock:
            batch_storage[batch_id]["status"] = "complete"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        with batch_lock:
            batch_storage[batch_id]["status"] = "error"
            batch_storage[batch_id]["error"] = str(e)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def process_batch_image(batch_id: str, temp_filename: str, original_filename: str, file_hash: str, saved_path: str):
    """Background task to process an image file"""
    try:
        # Update batch with total pages (1 for image)
        with batch_lock:
            batch_storage[batch_id]["total_pages"] = 1
            batch_storage[batch_id]["current_page"] = 1
        
        # Process the image using existing utility
        results = utils.process_image(temp_filename, file_hash, original_filename)
        
        if results:
            for doc in results:
                doc["file_path_url"] = saved_path
                doc["source_filename"] = original_filename
                
                po_number = doc.get("po_number")
                if po_number and utils.check_po_number_exists(po_number):
                    doc["already_exists"] = True
                    doc["duplicate_message"] = f"PO {po_number} already exists in database"
                else:
                    doc["already_exists"] = False
                
                is_valid, error_msg = utils.validate_is_po(doc)
                if is_valid:
                    # Flag if line items sum doesn't match total
                    doc = flag_amount_mismatch(doc)
                    # Add to batch results
                    with batch_lock:
                        batch_storage[batch_id]["pos"].append(doc)
        
        # Mark as complete
        with batch_lock:
            batch_storage[batch_id]["status"] = "complete"
            
    except Exception as e:
        with batch_lock:
            batch_storage[batch_id]["status"] = "error"
            batch_storage[batch_id]["error"] = str(e)
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/api/upload_batch")
async def upload_batch(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Start batch processing and return batch ID"""
    batch_id = str(uuid.uuid4())
    temp_filename = f"temp_{batch_id}_{file.filename}"
    original_filename = file.filename
    
    try:
        # Save file
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate hash
        with open(temp_filename, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Upload to storage
        saved_path, storage_existed = utils.upload_to_gcs(temp_filename, original_filename)
        
        # Check file type
        file_ext = os.path.splitext(original_filename)[1].lower()
        supported_images = [".jpg", ".jpeg", ".png"]
        
        if file_ext != ".pdf" and file_ext not in supported_images:
            raise HTTPException(status_code=400, detail="Only PDF and image files (JPG, JPEG, PNG) are supported for batch processing")
        
        # Initialize batch storage
        with batch_lock:
            batch_storage[batch_id] = {
                "status": "processing",
                "current_page": 0,
                "total_pages": 0,
                "pos": [],
                "error": None,
                "page_errors": [],
                "storage_existed": storage_existed,
                "storage_url": saved_path
            }
        
        # Start background processing based on file type
        if file_ext == ".pdf":
            thread = threading.Thread(
                target=process_batch_file,
                args=(batch_id, temp_filename, original_filename, file_hash, saved_path)
            )
        else:
            # Image file
            thread = threading.Thread(
                target=process_batch_image,
                args=(batch_id, temp_filename, original_filename, file_hash, saved_path)
            )
        thread.daemon = True
        thread.start()
        
        return {"batch_id": batch_id}
        
    except HTTPException as he:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/batch_status/{batch_id}")
def get_batch_status(batch_id: str):
    """Get current status of batch processing"""
    with batch_lock:
        if batch_id not in batch_storage:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        batch_data = batch_storage[batch_id]
        return {
            "status": batch_data["status"],
            "progress": {
                "current": batch_data["current_page"],
                "total": batch_data["total_pages"]
            },
            "pos": batch_data["pos"],
            "error": batch_data.get("error"),
            "page_errors": batch_data.get("page_errors", []),
            "storage_existed": batch_data.get("storage_existed", False),
            "storage_url": batch_data.get("storage_url")
        }


import asyncio
import json


@app.post("/api/upload_stream")
async def upload_file_stream(file: UploadFile = File(...)):
    """Streaming endpoint that yields POs as they're extracted"""
    temp_filename = f"temp_{file.filename}"
    original_filename = file.filename
    
    async def event_generator():
        try:
            # Save file
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            with open(temp_filename, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            saved_path, storage_existed = utils.upload_to_gcs(temp_filename, original_filename)
            file_ext = os.path.splitext(original_filename)[1].lower()
            
            if file_ext != ".pdf":
                yield f"data: {json.dumps({'type': 'error', 'message': 'Only PDF supported for streaming'})}\\n\\n"
                return
            
            # Process PDF page by page
            import pdfplumber
            with pdfplumber.open(temp_filename) as pdf:
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    text = page.extract_text(x_tolerance=2) or ""
                    
                    if len(text.strip()) < 50:
                        continue
                    
                    # Detect retailer
                    t_page = text.upper()
                    curr_retailer = "UNKNOWN"
                    if "MYDIN" in t_page: curr_retailer = "MYDIN"
                    elif "SELECTION" in t_page: curr_retailer = "SELECTION_GROCERIES"
                    elif "DARUSSALAM" in t_page: curr_retailer = "PASARAYA_DARUSSALAM"
                    elif "PASARAYA" in t_page or "ANGKASA" in t_page: curr_retailer = "PASARAYA_ANGKASA"
                    elif "TUNAS" in t_page or "MANJA" in t_page: curr_retailer = "TUNAS_MANJA"
                    elif "ROSYAM" in t_page: curr_retailer = "ST_ROSYAM"
                    elif "GLOBAL JAYA" in t_page: curr_retailer = "GLOBAL_JAYA"
                    elif "GCH" in t_page or "GIANT" in t_page: curr_retailer = "GIANT"
                    elif "CS GROCER" in t_page: curr_retailer = "CS_GROCER"
                    elif "SUPER SEVEN" in t_page: curr_retailer = "SUPER_SEVEN"
                    elif "SAM'S GROCERIA" in t_page or "CHECKERS" in t_page: curr_retailer = "CHECKERS_SAM"
                    elif "LOTUSS" in t_page or "LOTUS" in t_page: curr_retailer = "LOTUS"
                    elif "TFP" in t_page: curr_retailer = "TFP_GROUP"
                    
                    # Extract POs
                    page_results = utils.parse_with_gemini(text, utils.RETAILER_PROMPT_MAP.get(curr_retailer, ""))
                    
                    if page_results:
                        for doc in page_results:
                            doc = utils.enrich_po_data(doc, file_hash)
                            doc = utils.clean_nan_values(doc)
                            doc["file_path_url"] = saved_path
                            doc["source_filename"] = original_filename
                            
                            po_number = doc.get("po_number")
                            if po_number and utils.check_po_number_exists(po_number):
                                doc["already_exists"] = True
                                doc["duplicate_message"] = f"PO {po_number} already exists in database"
                            else:
                                doc["already_exists"] = False
                            
                            is_valid, error_msg = utils.validate_is_po(doc)
                            if is_valid:
                                # Flag if line items sum doesn't match total
                                doc = flag_amount_mismatch(doc)
                                yield f"data: {json.dumps({'type': 'po', 'data': doc, 'page': page_num, 'total_pages': total_pages})}\\n\\n"
                                await asyncio.sleep(0)  # Allow other tasks to run
                
                yield f"data: {json.dumps({'type': 'complete'})}\\n\\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ==========================================
# GOOGLE SHEETS OAUTH ENDPOINTS
# ==========================================

from fastapi import Query

# In-memory session storage (use Redis/database in production)
user_sessions = {}


@app.get("/api/google/auth/initiate")
async def google_auth_initiate(redirect_uri: str = Query(default="http://localhost:3000/auth/google/callback")):
    """Initiate Google OAuth flow"""
    try:
        flow = google_sheets_service.get_oauth_flow(redirect_uri)
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        
        return JSONResponse({
            "auth_url": authorization_url,
            "state": state
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth initiation failed: {str(e)}")


@app.get("/api/google/auth/callback")
async def google_auth_callback(code: str, state: str, redirect_uri: str = Query(default="http://localhost:3000/auth/google/callback")):
    """Handle Google OAuth callback"""
    try:
        flow = google_sheets_service.get_oauth_flow(redirect_uri)
        flow.fetch_token(code=code)
        
        credentials = flow.credentials
        credentials_dict = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = credentials_dict
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": "Google authentication successful"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OAuth callback failed: {str(e)}")


@app.post("/api/google/sheets/export")
async def export_to_google_sheets(
    session_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    view_mode: str = "item_level"
):
    """Export PO data to Google Sheets in user's Drive"""
    try:
        # Verify session
        if session_id not in user_sessions:
            raise HTTPException(status_code=401, detail="Not authenticated. Please sign in with Google.")
        
        credentials_dict = user_sessions[session_id]
        
        # Refresh credentials if needed
        credentials_dict = google_sheets_service.refresh_credentials(credentials_dict)
        user_sessions[session_id] = credentials_dict
        
        # Fetch PO data
        if start_date or end_date:
            po_data = utils.fetch_pos_by_date_range(start_date, end_date)
        else:
            po_data = utils.fetch_all_pos_from_db()
        
        if not po_data:
            raise HTTPException(status_code=404, detail="No PO data found")
        
        # Convert to list of dicts
        po_list = [dict(row) for row in po_data]
        
        # Create spreadsheet title with date range
        if start_date and end_date:
            title = f"PO Export {start_date} to {end_date}"
        elif start_date:
            title = f"PO Export from {start_date}"
        else:
            from datetime import datetime
            title = f"PO Export {datetime.now().strftime('%Y-%m-%d')}"
        
        # Create spreadsheet with view_mode
        spreadsheet_url = google_sheets_service.create_spreadsheet_from_pos(
            credentials_dict,
            po_list,
            title,
            view_mode
        )
        
        return JSONResponse({
            "success": True,
            "spreadsheet_url": spreadsheet_url,
            "message": "Google Sheet created successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/google/auth/status")
async def google_auth_status(session_id: Optional[str] = None):
    """Check if user is authenticated with Google"""
    if session_id and session_id in user_sessions:
        return JSONResponse({"authenticated": True, "session_id": session_id})
    return JSONResponse({"authenticated": False})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)