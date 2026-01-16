"""
PubChem API Downloader

Retrieves chemical structure data (SMILES, InChI) for drugs by name.
Used to enrich drug data from FDA/clinical trials with molecular structures.

API Documentation: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).parent.parent))
from config import PUBCHEM_API

console = Console()


class PubChemDownloader:
    """Retrieves molecular structures from PubChem for drug names."""
    
    def __init__(self):
        self.base_url = PUBCHEM_API["base_url"]
        self.output_dir = PUBCHEM_API["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = PUBCHEM_API["rate_limit"]  # requests per second
        self._last_request_time = 0
    
    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def get_compound_by_name(self, name: str) -> Optional[dict]:
        """
        Get compound data from PubChem by name.
        
        Args:
            name: Drug name to search
            
        Returns:
            Dictionary with compound properties or None
        """
        self._rate_limit_wait()
        
        # Use requests library's URL encoding
        clean_name = name.strip()
        url = f"{self.base_url}/compound/name/{clean_name}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IsomericSMILES,InChI,InChIKey/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                return None  # Compound not found
            
            response.raise_for_status()
            data = response.json()
            
            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                props = data["PropertyTable"]["Properties"][0]
                # Handle both CanonicalSMILES and ConnectivitySMILES
                smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
                return {
                    "cid": props.get("CID"),
                    "molecular_formula": props.get("MolecularFormula"),
                    "molecular_weight": props.get("MolecularWeight"),
                    "canonical_smiles": smiles,
                    "isomeric_smiles": props.get("IsomericSMILES"),
                    "inchi": props.get("InChI"),
                    "inchi_key": props.get("InChIKey"),
                }
            
        except requests.RequestException as e:
            console.print(f"[yellow]  Request error for {name}: {e}[/yellow]")
        except json.JSONDecodeError:
            pass
        
        return None
    
    def lookup_drugs(self, drug_names: List[str], max_workers: int = 1) -> pd.DataFrame:
        """
        Look up SMILES and molecular properties for a list of drug names.
        
        Args:
            drug_names: List of drug names to look up
            max_workers: Number of concurrent workers (default 1 for rate limiting)
            
        Returns:
            DataFrame with drug names and their molecular properties
        """
        console.print(f"[bold blue]📥 Looking up {len(drug_names)} drugs in PubChem...[/bold blue]")
        
        results = []
        found = 0
        not_found = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Looking up drugs...", total=len(drug_names))
            
            for name in drug_names:
                if not name or pd.isna(name):
                    progress.update(task, advance=1)
                    continue
                
                # Clean the drug name
                clean_name = str(name).strip()
                if not clean_name:
                    progress.update(task, advance=1)
                    continue
                
                result = self.get_compound_by_name(clean_name)
                
                if result:
                    result["drug_name"] = name
                    results.append(result)
                    found += 1
                else:
                    results.append({
                        "drug_name": name,
                        "cid": None,
                        "molecular_formula": None,
                        "molecular_weight": None,
                        "canonical_smiles": None,
                        "isomeric_smiles": None,
                        "inchi": None,
                        "inchi_key": None,
                    })
                    not_found += 1
                
                progress.update(task, advance=1)
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "drug_structures.csv", index=False)
        
        console.print(f"[green]✓ Found structures for {found} drugs[/green]")
        console.print(f"[yellow]  Not found: {not_found} drugs[/yellow]")
        
        return df
    
    def enrich_dataframe(
        self, 
        df: pd.DataFrame, 
        name_column: str = "drug_name"
    ) -> pd.DataFrame:
        """
        Enrich a DataFrame with SMILES and molecular properties.
        
        Args:
            df: Input DataFrame with drug names
            name_column: Name of the column containing drug names
            
        Returns:
            DataFrame with added molecular property columns
        """
        console.print(f"[bold blue]📥 Enriching {len(df)} records with PubChem data...[/bold blue]")
        
        # Get unique drug names
        unique_names = df[name_column].dropna().unique().tolist()
        console.print(f"  Unique drugs to look up: {len(unique_names)}")
        
        # Look up all unique drugs
        structures_df = self.lookup_drugs(unique_names)
        
        # Merge back to original DataFrame
        enriched_df = df.merge(
            structures_df,
            left_on=name_column,
            right_on="drug_name",
            how="left",
            suffixes=("", "_pubchem")
        )
        
        return enriched_df
    
    def run(self, sample_drugs: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run the PubChem lookup with sample drugs or load from file.
        
        Args:
            sample_drugs: Optional list of drug names to look up
            
        Returns:
            DataFrame with drug structures
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]PUBCHEM API DOWNLOADER[/bold cyan]")
        console.print("="*60 + "\n")
        
        if sample_drugs is None:
            # Load some sample approved drugs
            sample_drugs = [
                "Aspirin",
                "Ibuprofen",
                "Metformin",
                "Atorvastatin",
                "Omeprazole",
                "Lisinopril",
                "Amlodipine",
                "Metoprolol",
                "Paracetamol",
                "Simvastatin",
            ]
        
        df = self.lookup_drugs(sample_drugs)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ PubChem Lookup Complete![/bold green]")
        console.print(f"  Total drugs: {len(df)}")
        console.print(f"  With SMILES: {df['canonical_smiles'].notna().sum()}")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return df


if __name__ == "__main__":
    downloader = PubChemDownloader()
    data = downloader.run()
