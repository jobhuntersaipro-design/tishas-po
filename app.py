import hashlib
import math
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import test_engine  # Backend logic

# --- Page Config ---
st.set_page_config(page_title="Tishas PO Extractor", layout="wide", page_icon="📄")

# --- Session State Initialization ---
if "processed_docs" not in st.session_state:
    st.session_state["processed_docs"] = []
if "processed_hashes" not in st.session_state:
    st.session_state["processed_hashes"] = set()
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 0
if "history_page" not in st.session_state:
    st.session_state["history_page"] = 0
if "all_saved_message" not in st.session_state:
    st.session_state["all_saved_message"] = False
if "accumulated_files" not in st.session_state:
    st.session_state["accumulated_files"] = []
if "process_log" not in st.session_state:
    st.session_state["process_log"] = []
if "log_page" not in st.session_state:  # NEW: Log pagination
    st.session_state["log_page"] = 0

# --- Queue Persistence States ---
if "processing_active" not in st.session_state:
    st.session_state["processing_active"] = False

# Update URL on tab change
query_params = st.query_params
default_tab = query_params.get("view", "Process New POs")

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = default_tab
else:
    if (
        "view" in query_params
        and query_params["view"] != st.session_state["active_tab"]
    ):
        st.session_state["active_tab"] = query_params["view"]


# --- Helper: Confidence & Health ---
def get_field_status(value):
    if (
        value is None
        or str(value).strip() == ""
        or str(value).upper() in ["NO_CODE", "NULL"]
    ):
        return "🔴"
    try:
        if float(value) == 0.0:
            return "🔴"
    except (ValueError, TypeError):
        pass
    return "🟢"


# --- Helper: Render a Single PO Card ---
def render_po_card(doc, index):
    reliability = doc.get("reliability_score", 0)

    # CHECK DB STATUS DYNAMICALLY
    is_saved = test_engine.check_if_exists(doc.get("file_hash"))

    # FILTER: If saved, do NOT render in this view (Requirement #2)
    if is_saved:
        return

    po_title = doc.get("po_number") or "Unknown PO"
    ret_title = (
        doc.get("retailer_name_standardized")
        or doc.get("retailer_name")
        or "Unknown Retailer"
    )

    with st.expander(f"**{po_title}** | {ret_title} ({reliability}%)", expanded=True):

        c_stat, c_info = st.columns([1, 5])
        with c_stat:
            if reliability < 60:
                st.warning("REVIEW")
            else:
                st.info("READY")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Retailer Info**")
            doc["debtor_code"] = st.text_input(
                f"{get_field_status(doc.get('debtor_code'))} Debtor Code",
                value=doc.get("debtor_code") or "",
                key=f"debtor_{index}",
            )
            doc["branch_name"] = st.text_input(
                f"{get_field_status(doc.get('branch_name'))} Branch Name",
                value=doc.get("branch_name") or "",
                key=f"branch_{index}",
            )
            doc["branch_code"] = st.text_input(
                f"{get_field_status(doc.get('branch_code'))} Branch Code",
                value=doc.get("branch_code") or "",
                key=f"bcode_{index}",
            )
            doc["buyer_name"] = st.text_input(
                f"{get_field_status(doc.get('buyer_name'))} Buyer Name",
                value=doc.get("buyer_name") or "",
                key=f"buyer_{index}",
            )

        with c2:
            st.markdown("**Logistics & Tax**")
            doc["po_number"] = st.text_input(
                f"{get_field_status(doc.get('po_number'))} PO Number",
                value=doc.get("po_number") or "",
                key=f"ponum_{index}",
            )
            doc["delivery_address"] = st.text_area(
                f"{get_field_status(doc.get('delivery_address'))} Delivery Address",
                value=doc.get("delivery_address") or "",
                height=100,
                key=f"daddr_{index}",
            )
            doc["tax_id"] = st.text_input(
                f"{get_field_status(doc.get('tax_id'))} Tax ID",
                value=doc.get("tax_id") or "",
                key=f"taxid_{index}",
            )

        with c3:
            st.markdown("**Dates & Financials**")
            doc["po_date"] = st.text_input(
                f"{get_field_status(doc.get('po_date'))} PO Date",
                value=doc.get("po_date") or "",
                key=f"podate_{index}",
            )
            doc["delivery_date"] = st.text_input(
                f"{get_field_status(doc.get('delivery_date'))} Delivery Date",
                value=doc.get("delivery_date") or "",
                key=f"deldate_{index}",
            )

            curr = doc.get("currency", "MYR")
            raw_amt = doc.get("total_amount", 0.0)
            try:
                if isinstance(raw_amt, str):
                    clean_amt = raw_amt.replace(",", "").replace(curr, "").strip()
                    amt = float(clean_amt)
                else:
                    amt = float(raw_amt)
            except:
                amt = 0.0

            combined_val = f"{curr} {amt:,.2f}"

            new_combined = st.text_input(
                "Total Amount", value=combined_val, key=f"comb_{index}"
            )
            doc["total_amount"] = new_combined

            st.caption(f"Match Score: {reliability}%")

        with c4:
            st.markdown("**Actions**")
            if doc.get("file_path_url"):
                fname = os.path.basename(doc.get("file_path_url"))
                if doc.get("file_path_url").startswith("http"):
                    st.markdown(f"📄 [**{fname}**]({doc.get('file_path_url')})")
                else:
                    st.caption(f"📄 {fname}")

            # Save Button
            if st.button("Save Changes", key=f"btn_save_{index}"):
                test_engine.save_to_db(doc)
                st.toast(f"PO {doc.get('po_number')} saved!")
                st.rerun()

        with st.expander("Edit Line Items", expanded=False):
            if doc.get("items"):
                display_items = []
                for item in doc["items"]:
                    display_items.append(
                        {
                            "Article Code": item.get("sku")
                            or item.get("article_code")
                            or item.get("Article Code"),
                            "Barcode": item.get("barcode") or item.get("Barcode"),
                            "Article Description": item.get("description")
                            or item.get("article_description")
                            or item.get("Article Description"),
                            "Quantity": item.get("quantity")
                            or item.get("qty")
                            or item.get("Qty")
                            or item.get("Quantity"),
                            "UOM": item.get("uom") or item.get("UOM"),
                            "Unit Price": float(
                                item.get("unit_price", 0) or item.get("Unit Price", 0)
                            ),
                            "Line Total": float(
                                item.get("total_price", 0)
                                or item.get("line_total", 0)
                                or item.get("Line Total", 0)
                            ),
                        }
                    )

                edited_df = st.data_editor(
                    display_items,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"ed_{index}",
                    column_config={
                        "Unit Price": st.column_config.NumberColumn(format="%.2f"),
                        "Line Total": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                doc["items"] = edited_df


# --- MAIN UI ---
st.title("📄 Tishas PO Extraction Engine")

# --- CSS STYLING ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="stRadio"] {
        margin-top: 20px !important;
        padding-bottom: 30px;
    }

    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 30px;
    }

    .pagination-text {
        text-align: center;
        font-weight: bold;
        padding-top: 5px;
    }

    .stMetric {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def update_url():
    st.query_params["view"] = st.session_state["active_tab"]


selected_tab = st.radio(
    "Navigation",
    ["Process New POs", "Database History"],
    horizontal=True,
    key="active_tab",
    on_change=update_url,
    label_visibility="collapsed",
)

# ==========================================
# VIEW 1: PROCESS NEW FILES
# ==========================================
if selected_tab == "Process New POs":

    # ----------------------------------------------------
    # 1. FILE UPLOAD & ACCUMULATION
    # ----------------------------------------------------
    st.markdown("### Upload Files")
    c_upload, c_spacer = st.columns([6, 1])

    with c_upload:
        uploaded_files_current = st.file_uploader(
            "Upload POs (PDF/Image)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
        )

    if uploaded_files_current:
        current_signatures = {
            f"{f.name}_{f.size}" for f in st.session_state["accumulated_files"]
        }
        for f in uploaded_files_current:
            sig = f"{f.name}_{f.size}"
            if sig not in current_signatures:
                st.session_state["accumulated_files"].append(f)

    # ----------------------------------------------------
    # 2. QUEUE CALCULATION
    # ----------------------------------------------------
    all_files_in_queue = st.session_state["accumulated_files"]
    pending_files = []
    processed_count = 0

    for f in all_files_in_queue:
        file_id = f"{f.name}_{f.size}"
        if file_id in st.session_state["processed_hashes"]:
            processed_count += 1
        else:
            pending_files.append(f)

    # AUTO-START TRIGGER
    if len(pending_files) > 0:
        st.session_state["processing_active"] = True

    # ----------------------------------------------------
    # 3. STATUS BAR & STOP BUTTON (Requirement #1)
    # ----------------------------------------------------
    st.markdown("### Queue Status")

    col_done, col_pending, col_action = st.columns([1, 1, 4])

    with col_done:
        st.metric("Done", processed_count)

    with col_pending:
        st.metric("Pending", len(pending_files))

    with col_action:
        # Stop Button only appears if there are pending files
        if len(pending_files) > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🛑 Stop / Clear Queue", type="secondary"):
                st.session_state["processing_active"] = False
                st.session_state["accumulated_files"] = []
                st.session_state["process_log"] = []
                st.rerun()

    st.divider()

    # ----------------------------------------------------
    # 4. LOGS (Requirement #5: Pagination)
    # ----------------------------------------------------
    if st.session_state["process_log"]:
        with st.expander("Processing Log (Latest at Top)", expanded=False):

            LOGS_PER_PAGE = 10
            total_logs = len(st.session_state["process_log"])
            total_log_pages = max(1, math.ceil(total_logs / LOGS_PER_PAGE))

            # Simple Pagination Controls
            c_prev_log, c_txt_log, c_next_log = st.columns([1, 2, 1])

            if c_prev_log.button(
                "◀", key="log_prev", disabled=st.session_state["log_page"] == 0
            ):
                st.session_state["log_page"] -= 1
                st.rerun()

            c_txt_log.caption(
                f"Page {st.session_state['log_page'] + 1} / {total_log_pages}"
            )

            if c_next_log.button(
                "▶",
                key="log_next",
                disabled=st.session_state["log_page"] >= total_log_pages - 1,
            ):
                st.session_state["log_page"] += 1
                st.rerun()

            start_l = st.session_state["log_page"] * LOGS_PER_PAGE
            end_l = start_l + LOGS_PER_PAGE

            for msg in st.session_state["process_log"][start_l:end_l]:
                st.text(msg)

    # ----------------------------------------------------
    # 5. MAIN PROCESSING LOOP (Daisy Chain)
    # ----------------------------------------------------
    if st.session_state["processing_active"] and len(pending_files) > 0:

        # Pick FIRST file
        uploaded_file = pending_files[0]
        ts = datetime.now().strftime("%H:%M:%S")

        # Requirement #4: Progress bar for THIS file only (can be simulated with spinner)
        # Using spinner as main indicator for single-file focus
        with st.spinner(f"Processing {uploaded_file.name}..."):

            st.session_state["process_log"].insert(
                0, f"[{ts}] ⏳ {uploaded_file.name}..."
            )

            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            temp_filename = f"temp_{uploaded_file.name}"

            try:
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # --- STORAGE UPLOAD ---
                saved_path = None
                if hasattr(test_engine, "upload_to_gcs"):
                    saved_path = test_engine.upload_to_gcs(
                        temp_filename, uploaded_file.name
                    )
                elif hasattr(test_engine, "upload_to_s3"):
                    saved_path = test_engine.upload_to_s3(
                        temp_filename, uploaded_file.name
                    )
                elif hasattr(test_engine, "save_local_copy"):
                    saved_path = test_engine.save_local_copy(
                        temp_filename, uploaded_file.name
                    )

                file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()

                # --- AI EXTRACTION ---
                results = None
                if file_ext == ".pdf":
                    results = test_engine.process_pdf(temp_filename, file_hash)
                else:
                    results = test_engine.process_image(temp_filename, file_hash)

                # --- UPDATE STATE ---
                if results:
                    if isinstance(results, list):
                        for doc in results:
                            doc["file_path_url"] = saved_path
                        st.session_state["processed_docs"].extend(results)
                    else:
                        results["file_path_url"] = saved_path
                        st.session_state["processed_docs"].append(results)

                    ts_end = datetime.now().strftime("%H:%M:%S")
                    st.session_state["process_log"].insert(
                        0, f"[{ts_end}] ✅ {uploaded_file.name}: Done"
                    )
                    st.session_state["processed_hashes"].add(
                        f"{uploaded_file.name}_{uploaded_file.size}"
                    )
                else:
                    st.session_state["process_log"].insert(
                        0, f"[{ts}] ❌ {uploaded_file.name}: No Data"
                    )

            except Exception as e:
                st.session_state["process_log"].insert(
                    0, f"[{ts}] ❌ {uploaded_file.name}: Error ({e})"
                )
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # CRITICAL: Rerun immediately to show result (Requirement #3)
        st.rerun()

    # ----------------------------------------------------
    # 6. RESULTS RENDERER
    # ----------------------------------------------------

    # Filter docs that are NOT saved yet (Requirement #2)
    # We create a display list by filtering the main session state list
    docs_to_display = [
        d
        for d in st.session_state["processed_docs"]
        if not test_engine.check_if_exists(d.get("file_hash"))
    ]

    if docs_to_display:
        st.markdown(f"### Extracted Results ({len(docs_to_display)} unsaved)")

        col_list_title, col_save_action = st.columns([5, 2])
        with col_save_action:
            if st.button(
                "💾 SAVE ALL UNSAVED", type="primary", use_container_width=True
            ):
                saved_count = 0
                for doc in docs_to_display:
                    if test_engine.save_to_db(doc):
                        saved_count += 1
                st.toast(f"Saved {saved_count} new POs!")
                st.rerun()

        # Pagination Logic for Results
        ITEMS_PER_PAGE = 5
        total_items = len(docs_to_display)
        total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))

        if st.session_state["current_page"] >= total_pages:
            st.session_state["current_page"] = max(0, total_pages - 1)

        start_idx = st.session_state["current_page"] * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        current_batch = docs_to_display[start_idx:end_idx]

        if total_pages > 1:
            _, c_prev, c_txt, c_next, _ = st.columns([5, 1, 2, 1, 5])
            with c_prev:
                if st.button(
                    "◀",
                    key="proc_prev",
                    use_container_width=True,
                    disabled=st.session_state["current_page"] == 0,
                ):
                    st.session_state["current_page"] -= 1
                    st.rerun()
            with c_txt:
                st.markdown(
                    f"<div class='pagination-text'>{st.session_state['current_page'] + 1} / {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with c_next:
                if st.button(
                    "▶",
                    key="proc_next",
                    use_container_width=True,
                    disabled=st.session_state["current_page"] == total_pages - 1,
                ):
                    st.session_state["current_page"] += 1
                    st.rerun()

        for i, doc in enumerate(current_batch):
            # Index must be unique relative to the full list to avoid key collision
            real_index = st.session_state["processed_docs"].index(doc)
            render_po_card(doc, real_index)

    elif len(st.session_state["processed_docs"]) > 0:
        # State where docs exist but all are saved
        st.success("✅ All processed documents have been saved to the database.")

# ==========================================
# VIEW 2: HISTORY (Database)
# ==========================================
elif selected_tab == "Database History":
    if st.button("Refresh Data", key="refresh_history"):
        st.cache_data.clear()

    df_history = test_engine.fetch_all_pos_from_db()

    if not df_history.empty:
        # Convert numeric columns explicitly
        numeric_cols = ["total_amount", "line_total", "unit_price", "qty"]
        for col in numeric_cols:
            if col in df_history.columns:
                df_history[col] = pd.to_numeric(df_history[col], errors="coerce")

        search_query = st.text_input(
            "🔍 Search",
            "",
            key="history_search",
            help="Type to search. Matches text anywhere in the row (e.g., 'sd' matches 'sdn').",
        )

        if search_query:
            mask = (
                df_history.astype(str)
                .apply(lambda x: x.str.contains(search_query, case=False, regex=False))
                .any(axis=1)
            )
            df_history = df_history[mask]

        # STRICT COLUMN MAPPING
        col_mapping = {
            "id": "Id",
            "debtor_code": "Debtor Code",
            "retailer_name": "Retailer Name",
            "branch_name": "Branch Name",
            "buyer_name": "Buyer Name",
            "branch_code": "Branch Code",
            "delivery_address": "Delivery Address",
            "tax_id": "Tax ID",
            "po_number": "PO Number",
            "po_date": "PO Date",
            "delivery_date": "Delivery Date",
            "currency": "Currency",
            "total_amount": "Total Amount",
            "article_code": "Article Code",
            "barcode": "Barcode",
            "article_description": "Article Description",
            "qty": "Quantity",
            "uom": "UOM",
            "unit_price": "Unit Price",
            "line_total": "Line Total",
            "status": "Status",
            "file_storage_url": "File Storage Url",
            "file_hash": "File Hash",
        }

        df_history.rename(columns=col_mapping, inplace=True)

        ITEMS_PER_PAGE_HISTORY = 20
        total_rows = len(df_history)
        total_pages_history = max(1, math.ceil(total_rows / ITEMS_PER_PAGE_HISTORY))

        if st.session_state["history_page"] >= total_pages_history:
            st.session_state["history_page"] = max(0, total_pages_history - 1)

        if total_pages_history > 1:
            _, c_prev, c_txt, c_next, _ = st.columns([5, 1, 2, 1, 5])

            with c_prev:
                if st.button(
                    "◀",
                    key="hist_prev",
                    use_container_width=True,
                    disabled=(st.session_state["history_page"] == 0),
                ):
                    st.session_state["history_page"] -= 1
                    st.rerun()
            with c_txt:
                st.markdown(
                    f"<div class='pagination-text'>{st.session_state['history_page'] + 1} / {total_pages_history}</div>",
                    unsafe_allow_html=True,
                )
            with c_next:
                if st.button(
                    "▶",
                    key="hist_next",
                    use_container_width=True,
                    disabled=(
                        st.session_state["history_page"] == total_pages_history - 1
                    ),
                ):
                    st.session_state["history_page"] += 1
                    st.rerun()

        start_idx_hist = st.session_state["history_page"] * ITEMS_PER_PAGE_HISTORY
        end_idx_hist = min(start_idx_hist + ITEMS_PER_PAGE_HISTORY, total_rows)

        desired_columns = [
            "Id",
            "Debtor Code",
            "Retailer Name",
            "Branch Name",
            "Buyer Name",
            "Branch Code",
            "Delivery Address",
            "Tax ID",
            "PO Number",
            "PO Date",
            "Delivery Date",
            "Currency",
            "Total Amount",
            "Article Code",
            "Barcode",
            "Article Description",
            "Quantity",
            "UOM",
            "Unit Price",
            "Line Total",
            "Status",
            "File Storage Url",
            "File Hash",
        ]

        existing_cols = [c for c in desired_columns if c in df_history.columns]
        df_display = df_history[existing_cols].iloc[start_idx_hist:end_idx_hist]

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=750,
            column_config={
                "Total Amount": st.column_config.NumberColumn(format="%.2f"),
                "Line Total": st.column_config.NumberColumn(format="%.2f"),
                "Unit Price": st.column_config.NumberColumn(format="%.2f"),
                "Quantity": st.column_config.NumberColumn("Quantity"),
                "File Storage Url": st.column_config.LinkColumn(
                    "Source File", display_text="🔗 Link"
                ),
                "PO Date": st.column_config.DateColumn("PO Date"),
                "Delivery Date": st.column_config.DateColumn("Delivery Date"),
            },
        )
    else:
        st.info("No records found.")
