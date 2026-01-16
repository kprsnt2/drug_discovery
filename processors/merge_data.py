"""
Data Merger - Combines data from all sources

Merges FDA Orange Book, openFDA, ClinicalTrials.gov, and PubChem data
into a unified drug database.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

console = Console()


class DataMerger:
    """Merges drug data from multiple sources."""
    
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.output_dir = PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_fda_orange_book(self) -> pd.DataFrame:
        """Load FDA Orange Book products."""
        path = self.raw_dir / "fda_orange_book" / "products.csv"
        if not path.exists():
            console.print("[yellow]⚠ FDA Orange Book data not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} FDA Orange Book products[/green]")
        return df
    
    def load_openfda_labels(self) -> pd.DataFrame:
        """Load openFDA drug labels."""
        path = self.raw_dir / "openfda" / "drug_labels.csv"
        if not path.exists():
            console.print("[yellow]⚠ openFDA labels not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} openFDA drug labels[/green]")
        return df
    
    def load_openfda_adverse_events(self) -> pd.DataFrame:
        """Load openFDA adverse events."""
        path = self.raw_dir / "openfda" / "adverse_events.csv"
        if not path.exists():
            console.print("[yellow]⚠ openFDA adverse events not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} adverse event records[/green]")
        return df
    
    def load_clinical_trials(self) -> pd.DataFrame:
        """Load clinical trials data."""
        path = self.raw_dir / "clinicaltrials" / "clinical_trials.csv"
        if not path.exists():
            console.print("[yellow]⚠ Clinical trials data not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} clinical trials[/green]")
        return df
    
    def load_failed_trials(self) -> pd.DataFrame:
        """Load failed/terminated trials."""
        path = self.raw_dir / "clinicaltrials" / "failed_trials.csv"
        if not path.exists():
            console.print("[yellow]⚠ Failed trials data not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} failed trials[/green]")
        return df
    
    def load_pubchem_structures(self) -> pd.DataFrame:
        """Load PubChem structure data."""
        path = self.raw_dir / "pubchem" / "drug_structures.csv"
        if not path.exists():
            console.print("[yellow]⚠ PubChem structures not found[/yellow]")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        console.print(f"[green]✓ Loaded {len(df)} PubChem structures[/green]")
        return df
    
    def extract_unique_drugs(self) -> pd.DataFrame:
        """
        Extract unique drug names from all sources.
        
        Returns:
            DataFrame with unique drug entries
        """
        console.print("\n[bold blue]Extracting unique drugs from all sources...[/bold blue]")
        
        all_drugs = []
        
        # FDA Orange Book
        fda_df = self.load_fda_orange_book()
        if not fda_df.empty:
            # Look for common drug name columns
            name_cols = ['ingredient', 'trade_name', 'applicant_full_name']
            for col in name_cols:
                if col in fda_df.columns:
                    drugs = fda_df[col].dropna().unique()
                    for drug in drugs:
                        all_drugs.append({
                            "drug_name": str(drug),
                            "source": "fda_orange_book",
                            "status": "approved",
                        })
        
        # openFDA labels
        labels_df = self.load_openfda_labels()
        if not labels_df.empty:
            for col in ["brand_name", "generic_name"]:
                if col in labels_df.columns:
                    drugs = labels_df[col].dropna().unique()
                    for drug in drugs:
                        all_drugs.append({
                            "drug_name": str(drug),
                            "source": "openfda_labels",
                            "status": "approved",
                        })
        
        # Clinical trials
        trials_df = self.load_clinical_trials()
        if not trials_df.empty and "drug_names" in trials_df.columns:
            for _, row in trials_df.iterrows():
                drug_names = str(row.get("drug_names", "")).split(", ")
                status = row.get("status", "unknown")
                
                for drug in drug_names:
                    if drug.strip():
                        all_drugs.append({
                            "drug_name": drug.strip(),
                            "source": "clinicaltrials",
                            "status": status,
                        })
        
        # Create DataFrame and deduplicate
        drugs_df = pd.DataFrame(all_drugs)
        
        if drugs_df.empty:
            console.print("[yellow]No drugs found from any source[/yellow]")
            return pd.DataFrame()
        
        # Group by drug name and aggregate sources
        grouped = drugs_df.groupby("drug_name").agg({
            "source": lambda x: ", ".join(sorted(set(x))),
            "status": "first",
        }).reset_index()
        
        console.print(f"[green]✓ Extracted {len(grouped)} unique drugs[/green]")
        return grouped
    
    def create_unified_database(self) -> pd.DataFrame:
        """
        Create a unified drug database from all sources.
        
        Returns:
            Merged DataFrame with all drug information
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]CREATING UNIFIED DRUG DATABASE[/bold cyan]")
        console.print("="*60 + "\n")
        
        # Get unique drugs
        drugs_df = self.extract_unique_drugs()
        
        if drugs_df.empty:
            return pd.DataFrame()
        
        # Load PubChem structures
        structures_df = self.load_pubchem_structures()
        
        if not structures_df.empty:
            # Merge structures
            drugs_df = drugs_df.merge(
                structures_df,
                on="drug_name",
                how="left"
            )
        
        # Load adverse events and aggregate by drug
        adverse_df = self.load_openfda_adverse_events()
        if not adverse_df.empty and "drug_name" in adverse_df.columns:
            adverse_agg = adverse_df.groupby("drug_name").agg({
                "reactions": lambda x: "; ".join(x.dropna().astype(str).unique()[:5]),
            }).reset_index()
            adverse_agg.columns = ["drug_name", "adverse_reactions"]
            
            drugs_df = drugs_df.merge(
                adverse_agg,
                on="drug_name",
                how="left"
            )
        
        # Load failed trials for failure reasons
        failed_df = self.load_failed_trials()
        if not failed_df.empty and "drug_names" in failed_df.columns:
            # Explode drug names and get why_stopped
            failure_reasons = []
            for _, row in failed_df.iterrows():
                drug_names = str(row.get("drug_names", "")).split(", ")
                why_stopped = row.get("why_stopped", "")
                for drug in drug_names:
                    if drug.strip():
                        failure_reasons.append({
                            "drug_name": drug.strip(),
                            "failure_reason": why_stopped,
                        })
            
            if failure_reasons:
                failures_df = pd.DataFrame(failure_reasons)
                failures_agg = failures_df.groupby("drug_name").agg({
                    "failure_reason": lambda x: "; ".join(x.dropna().astype(str).unique()[:3]),
                }).reset_index()
                
                drugs_df = drugs_df.merge(
                    failures_agg,
                    on="drug_name",
                    how="left"
                )
        
        # Save unified database
        output_path = self.output_dir / "unified_drugs.csv"
        drugs_df.to_csv(output_path, index=False)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ Unified Database Created![/bold green]")
        console.print(f"  Total drugs: {len(drugs_df)}")
        if "canonical_smiles" in drugs_df.columns:
            console.print(f"  With SMILES: {drugs_df['canonical_smiles'].notna().sum()}")
        console.print(f"  Output: {output_path}")
        console.print("="*60 + "\n")
        
        return drugs_df
    
    def show_summary(self):
        """Display a summary of available data."""
        table = Table(title="Data Sources Summary")
        table.add_column("Source", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Records", justify="right")
        table.add_column("Status", style="yellow")
        
        sources = [
            ("FDA Orange Book", "fda_orange_book/products.csv"),
            ("openFDA Labels", "openfda/drug_labels.csv"),
            ("openFDA Adverse Events", "openfda/adverse_events.csv"),
            ("openFDA Recalls", "openfda/drug_recalls.csv"),
            ("Clinical Trials", "clinicaltrials/clinical_trials.csv"),
            ("Failed Trials", "clinicaltrials/failed_trials.csv"),
            ("PubChem Structures", "pubchem/drug_structures.csv"),
        ]
        
        for name, rel_path in sources:
            path = self.raw_dir / rel_path
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    table.add_row(name, rel_path, str(len(df)), "✓ Available")
                except Exception as e:
                    table.add_row(name, rel_path, "-", f"✗ Error: {e}")
            else:
                table.add_row(name, rel_path, "-", "⚠ Not downloaded")
        
        console.print(table)


if __name__ == "__main__":
    merger = DataMerger()
    merger.show_summary()
    merger.create_unified_database()
