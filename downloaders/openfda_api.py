"""
openFDA API Downloader

Downloads drug data from the openFDA API including:
- Drug labels (package inserts)
- Adverse event reports
- Drug recalls/enforcement actions

API Documentation: https://open.fda.gov/apis/
"""

import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

sys.path.append(str(Path(__file__).parent.parent))
from config import OPENFDA_API

console = Console()


class OpenFDADownloader:
    """Downloads drug data from openFDA API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = OPENFDA_API["base_url"]
        self.endpoints = OPENFDA_API["endpoints"]
        self.output_dir = OPENFDA_API["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key or OPENFDA_API["api_key"]
        self.rate_limit = OPENFDA_API["rate_limit"]
        self.max_results = OPENFDA_API["max_results_per_query"]
        
        self._last_request_time = 0
    
    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        min_interval = 60 / self.rate_limit  # seconds between requests
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make a rate-limited request to openFDA API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response as dictionary
        """
        self._rate_limit_wait()
        
        url = f"{self.base_url}{endpoint}"
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return {}
    
    def download_drug_labels(self, limit: int = 10000) -> pd.DataFrame:
        """
        Download drug labels (package inserts) from openFDA.
        
        Contains: Brand name, generic name, manufacturer, indications,
                  warnings, dosage, active ingredients, etc.
        
        Args:
            limit: Maximum number of records to download
            
        Returns:
            DataFrame with drug label data
        """
        console.print("[bold blue]📥 Downloading Drug Labels...[/bold blue]")
        
        endpoint = self.endpoints["drug_label"]
        all_records = []
        skip = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading labels...", total=limit)
            
            while skip < limit:
                batch_size = min(self.max_results, limit - skip)
                
                params = {
                    "limit": batch_size,
                    "skip": skip,
                }
                
                data = self._make_request(endpoint, params)
                
                if "results" not in data:
                    break
                
                results = data["results"]
                if not results:
                    break
                
                # Extract key fields from each label
                for label in results:
                    record = {
                        "brand_name": self._get_first(label.get("openfda", {}).get("brand_name", [])),
                        "generic_name": self._get_first(label.get("openfda", {}).get("generic_name", [])),
                        "manufacturer": self._get_first(label.get("openfda", {}).get("manufacturer_name", [])),
                        "substance_name": self._get_first(label.get("openfda", {}).get("substance_name", [])),
                        "product_type": self._get_first(label.get("openfda", {}).get("product_type", [])),
                        "route": self._get_first(label.get("openfda", {}).get("route", [])),
                        "application_number": self._get_first(label.get("openfda", {}).get("application_number", [])),
                        "indications": self._get_first(label.get("indications_and_usage", [])),
                        "warnings": self._get_first(label.get("warnings", [])),
                        "dosage": self._get_first(label.get("dosage_and_administration", [])),
                        "active_ingredients": self._get_first(label.get("active_ingredient", [])),
                        "purpose": self._get_first(label.get("purpose", [])),
                    }
                    all_records.append(record)
                
                skip += len(results)
                progress.update(task, completed=skip)
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.output_dir / "drug_labels.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} drug labels[/green]")
        return df
    
    def download_adverse_events(self, limit: int = 50000) -> pd.DataFrame:
        """
        Download adverse event reports from openFDA.
        
        Contains: Drug name, reaction, outcome, patient info, etc.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            DataFrame with adverse event data
        """
        console.print("[bold blue]📥 Downloading Adverse Events...[/bold blue]")
        
        endpoint = self.endpoints["drug_event"]
        all_records = []
        skip = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading events...", total=limit)
            
            while skip < limit:
                batch_size = min(self.max_results, limit - skip)
                
                params = {
                    "limit": batch_size,
                    "skip": skip,
                }
                
                data = self._make_request(endpoint, params)
                
                if "results" not in data:
                    break
                
                results = data["results"]
                if not results:
                    break
                
                for event in results:
                    # Get drug info from patient record
                    drugs = event.get("patient", {}).get("drug", [])
                    reactions = event.get("patient", {}).get("reaction", [])
                    
                    for drug in drugs:
                        record = {
                            "drug_name": drug.get("medicinalproduct", ""),
                            "drug_indication": drug.get("drugindication", ""),
                            "drug_characterization": drug.get("drugcharacterization", ""),
                            "reactions": ", ".join([r.get("reactionmeddrapt", "") for r in reactions]),
                            "outcome": event.get("patient", {}).get("summary", {}).get("narrativeincludeclinical", ""),
                            "serious": event.get("serious", ""),
                            "receive_date": event.get("receivedate", ""),
                            "country": event.get("occurcountry", ""),
                        }
                        all_records.append(record)
                
                skip += len(results)
                progress.update(task, completed=skip)
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.output_dir / "adverse_events.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} adverse events[/green]")
        return df
    
    def download_drug_recalls(self, limit: int = 5000) -> pd.DataFrame:
        """
        Download drug recall/enforcement data from openFDA.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            DataFrame with recall data
        """
        console.print("[bold blue]📥 Downloading Drug Recalls...[/bold blue]")
        
        endpoint = self.endpoints["drug_enforcement"]
        all_records = []
        skip = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading recalls...", total=limit)
            
            while skip < limit:
                batch_size = min(self.max_results, limit - skip)
                
                params = {
                    "limit": batch_size,
                    "skip": skip,
                }
                
                data = self._make_request(endpoint, params)
                
                if "results" not in data:
                    break
                
                results = data["results"]
                if not results:
                    break
                
                for recall in results:
                    record = {
                        "product_description": recall.get("product_description", ""),
                        "reason_for_recall": recall.get("reason_for_recall", ""),
                        "status": recall.get("status", ""),
                        "classification": recall.get("classification", ""),
                        "recalling_firm": recall.get("recalling_firm", ""),
                        "recall_initiation_date": recall.get("recall_initiation_date", ""),
                        "voluntary_mandated": recall.get("voluntary_mandated", ""),
                        "city": recall.get("city", ""),
                        "state": recall.get("state", ""),
                    }
                    all_records.append(record)
                
                skip += len(results)
                progress.update(task, completed=skip)
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.output_dir / "drug_recalls.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} drug recalls[/green]")
        return df
    
    def _get_first(self, lst: list, default: str = "") -> str:
        """Get first element of list or default."""
        if isinstance(lst, list) and lst:
            return lst[0] if isinstance(lst[0], str) else str(lst[0])
        return default
    
    def run(self, labels: bool = True, events: bool = True, recalls: bool = True) -> dict:
        """
        Run the complete download pipeline.
        
        Args:
            labels: Download drug labels
            events: Download adverse events  
            recalls: Download recall data
            
        Returns:
            Dictionary with all downloaded DataFrames
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]OPENFDA API DOWNLOADER[/bold cyan]")
        console.print("="*60 + "\n")
        
        results = {}
        
        if labels:
            results["labels"] = self.download_drug_labels()
        
        if events:
            results["adverse_events"] = self.download_adverse_events()
        
        if recalls:
            results["recalls"] = self.download_drug_recalls()
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ openFDA Download Complete![/bold green]")
        for name, df in results.items():
            console.print(f"  {name}: {len(df)} records")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return results


if __name__ == "__main__":
    downloader = OpenFDADownloader()
    data = downloader.run()
