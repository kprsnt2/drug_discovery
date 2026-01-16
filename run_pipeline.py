"""
New Drug Discovery Data Pipeline - Main Orchestrator

Downloads data from all official sources and prepares AI-ready training data.

Usage:
    python run_pipeline.py              # Run complete pipeline
    python run_pipeline.py --download   # Only download data
    python run_pipeline.py --process    # Only process existing data
    python run_pipeline.py --quick      # Quick test with smaller limits
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from downloaders import (
    FDAOrangeBookDownloader,
    OpenFDADownloader,
    ClinicalTrialsDownloader,
    PubChemDownloader,
)
from processors import DataMerger, TrainingDataPreparer

console = Console()


def print_banner():
    """Print a nice banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       NEW DRUG DISCOVERY DATA PIPELINE                       ║
║       Official FDA + ClinicalTrials.gov + PubChem            ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="cyan"))


def run_downloads(quick: bool = False):
    """
    Run all data downloads.
    
    Args:
        quick: If True, use smaller limits for testing
    """
    console.print("\n[bold cyan]PHASE 1: DATA DOWNLOAD[/bold cyan]\n")
    
    results = {}
    
    # 1. FDA Orange Book
    console.print("[bold]1/4 FDA Orange Book[/bold]")
    try:
        fda_downloader = FDAOrangeBookDownloader()
        results["fda_orange_book"] = fda_downloader.run()
    except Exception as e:
        console.print(f"[red]✗ FDA Orange Book failed: {e}[/red]")
    
    # 2. openFDA API
    console.print("\n[bold]2/4 openFDA API[/bold]")
    try:
        openfda_downloader = OpenFDADownloader()
        limit = 1000 if quick else 10000
        results["openfda"] = openfda_downloader.run()
    except Exception as e:
        console.print(f"[red]✗ openFDA failed: {e}[/red]")
    
    # 3. ClinicalTrials.gov
    console.print("\n[bold]3/4 ClinicalTrials.gov[/bold]")
    try:
        ct_downloader = ClinicalTrialsDownloader()
        limit = 1000 if quick else 10000
        results["clinicaltrials"] = ct_downloader.run(limit=limit)
    except Exception as e:
        console.print(f"[red]✗ ClinicalTrials.gov failed: {e}[/red]")
    
    # 4. PubChem (enrichment happens during processing)
    console.print("\n[bold]4/4 PubChem (will run during merge)[/bold]")
    console.print("[yellow]  ℹ PubChem lookups happen during data merge step[/yellow]")
    
    return results


def run_processing():
    """Run data processing pipeline."""
    console.print("\n[bold cyan]PHASE 2: DATA PROCESSING[/bold cyan]\n")
    
    # 1. Merge all data sources
    console.print("[bold]1/2 Merging Data Sources[/bold]")
    merger = DataMerger()
    merger.show_summary()
    unified_df = merger.create_unified_database()
    
    if unified_df.empty:
        console.print("[yellow]⚠ No data to process. Run downloads first.[/yellow]")
        return {}
    
    # 2. Prepare training data
    console.print("\n[bold]2/2 Preparing Training Data[/bold]")
    preparer = TrainingDataPreparer()
    stats = preparer.run()
    
    return stats


def run_pubchem_enrichment():
    """Enrich unified data with PubChem structures."""
    console.print("\n[bold cyan]PUBCHEM ENRICHMENT[/bold cyan]\n")
    
    from config import PROCESSED_DATA_DIR
    import pandas as pd
    
    # Load unified data
    unified_path = PROCESSED_DATA_DIR / "unified_drugs.csv"
    if not unified_path.exists():
        console.print("[yellow]⚠ Unified database not found. Run merge first.[/yellow]")
        return
    
    df = pd.read_csv(unified_path)
    
    # Get drugs without SMILES
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = None
    
    missing_smiles = df[df["canonical_smiles"].isna()]["drug_name"].tolist()
    
    if not missing_smiles:
        console.print("[green]✓ All drugs already have SMILES[/green]")
        return
    
    console.print(f"Looking up {len(missing_smiles)} drugs in PubChem...")
    
    # Look up in batches of 100
    pubchem = PubChemDownloader()
    batch_size = 100
    
    for i in range(0, min(len(missing_smiles), 500), batch_size):
        batch = missing_smiles[i:i+batch_size]
        structures_df = pubchem.lookup_drugs(batch)
        
        # Update unified df
        for _, row in structures_df.iterrows():
            if row.get("canonical_smiles"):
                mask = df["drug_name"] == row["drug_name"]
                df.loc[mask, "canonical_smiles"] = row["canonical_smiles"]
                df.loc[mask, "molecular_weight"] = row.get("molecular_weight")
    
    # Save updated
    df.to_csv(unified_path, index=False)
    console.print(f"[green]✓ Updated {unified_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="New Drug Discovery Data Pipeline"
    )
    parser.add_argument(
        "--download", 
        action="store_true",
        help="Only run data downloads"
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Only run data processing"
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run PubChem enrichment on existing data"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with smaller data limits"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Determine what to run
    run_dl = args.download or (not args.process and not args.enrich)
    run_proc = args.process or (not args.download and not args.enrich)
    
    if args.enrich:
        run_pubchem_enrichment()
    elif run_dl and run_proc:
        # Full pipeline
        run_downloads(quick=args.quick)
        run_processing()
    elif run_dl:
        run_downloads(quick=args.quick)
    elif run_proc:
        run_processing()
    
    console.print("\n[bold green]Pipeline Complete![/bold green]\n")


if __name__ == "__main__":
    main()
