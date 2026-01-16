"""
Configuration for New Drug Discovery Data Pipeline

Contains API endpoints, rate limits, and settings for all official data sources.
"""

from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FDA Orange Book - Updated URLs
# ============================================================================
FDA_ORANGE_BOOK = {
    # Main ZIP file with products.txt, patent.txt, exclusivity.txt
    "download_url": "https://www.fda.gov/media/76860/download?attachment",
    "data_url": "https://www.fda.gov/media/76860/download?attachment",
    "products_url": "https://www.accessdata.fda.gov/scripts/cder/ob/search_product.cfm",
    "output_dir": RAW_DATA_DIR / "fda_orange_book",
}

# ============================================================================
# openFDA API
# ============================================================================
OPENFDA_API = {
    "base_url": "https://api.fda.gov",
    "endpoints": {
        "drug_label": "/drug/label.json",
        "drug_event": "/drug/event.json",      # Adverse events
        "drug_ndc": "/drug/ndc.json",          # National Drug Code
        "drug_enforcement": "/drug/enforcement.json",  # Recalls
    },
    "rate_limit": 240,          # Requests per minute (no API key)
    "rate_limit_with_key": 240, # Requests per minute (with API key)
    "daily_limit": 120000,      # Requests per day (with key)
    "max_results_per_query": 1000,
    "api_key": None,            # Set your API key here (optional)
    "output_dir": RAW_DATA_DIR / "openfda",
}

# ============================================================================
# ClinicalTrials.gov API (v2)
# ============================================================================
CLINICALTRIALS_API = {
    "base_url": "https://clinicaltrials.gov/api/v2",
    "endpoints": {
        "studies": "/studies",
        "study_detail": "/studies/{nctId}",
    },
    "max_results_per_page": 1000,
    "fields": [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "OverallStatus",
        "Phase",
        "StudyType",
        "Condition",
        "InterventionName",
        "InterventionType",
        "WhyStopped",
        "StartDate",
        "CompletionDate",
        "PrimaryOutcomeMeasure",
        "EnrollmentCount",
        "LeadSponsorName",
    ],
    "output_dir": RAW_DATA_DIR / "clinicaltrials",
}

# ============================================================================
# PubChem API
# ============================================================================
PUBCHEM_API = {
    "base_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    "compound_by_name": "/compound/name/{name}/JSON",
    "compound_properties": "/compound/name/{name}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChI,InChIKey/JSON",
    "rate_limit": 5,  # Requests per second
    "output_dir": RAW_DATA_DIR / "pubchem",
}

# ============================================================================
# Drug Status Mappings
# ============================================================================
DRUG_STATUS = {
    "approved": 1,
    "withdrawn": 0,
    "terminated": 0,
    "suspended": 0,
    "completed": 1,  # Trial completed - may or may not be approved
    "active": 0.5,   # Still in trial
    "recruiting": 0.5,
}

# ============================================================================
# Processing Settings
# ============================================================================
PROCESSING = {
    "min_smiles_length": 5,
    "max_smiles_length": 500,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
}
