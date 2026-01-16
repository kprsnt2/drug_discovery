"""
ClinicalTrials.gov API Downloader

Downloads clinical trial data including:
- Trial phases and status
- Interventions (drugs being tested)
- Outcomes and termination reasons
- Sponsor information

API Documentation: https://clinicaltrials.gov/data-api/api
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
from config import CLINICALTRIALS_API

console = Console()


class ClinicalTrialsDownloader:
    """Downloads clinical trial data from ClinicalTrials.gov API v2."""
    
    def __init__(self):
        self.base_url = CLINICALTRIALS_API["base_url"]
        self.output_dir = CLINICALTRIALS_API["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_per_page = CLINICALTRIALS_API["max_results_per_page"]
    
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make a request to ClinicalTrials.gov API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return {}
    
    def download_drug_trials(
        self, 
        status: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Download drug clinical trials from ClinicalTrials.gov.
        
        Args:
            status: Filter by status (e.g., ["COMPLETED", "TERMINATED"])
            phases: Filter by phase (e.g., ["PHASE2", "PHASE3"])
            limit: Maximum number of trials to download
            
        Returns:
            DataFrame with clinical trial data
        """
        console.print("[bold blue]📥 Downloading Clinical Trials...[/bold blue]")
        
        all_records = []
        page_token = None
        total_downloaded = 0
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading trials...", total=limit)
            
            while total_downloaded < limit:
                # Build params for API v2
                params = {
                    "pageSize": min(self.max_per_page, limit - total_downloaded),
                    "format": "json",
                    "filter.overallStatus": ",".join(status) if status else None,
                    "filter.phase": ",".join(phases) if phases else None,
                }
                
                # Remove None values
                params = {k: v for k, v in params.items() if v is not None}
                
                if page_token:
                    params["pageToken"] = page_token
                
                data = self._make_request("/studies", params)
                
                if "studies" not in data:
                    break
                
                studies = data.get("studies", [])
                if not studies:
                    break
                
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    id_module = protocol.get("identificationModule", {})
                    status_module = protocol.get("statusModule", {})
                    design_module = protocol.get("designModule", {})
                    conditions_module = protocol.get("conditionsModule", {})
                    interventions_module = protocol.get("armsInterventionsModule", {})
                    outcomes_module = protocol.get("outcomesModule", {})
                    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
                    
                    # Get intervention names (drug names)
                    interventions = interventions_module.get("interventions", [])
                    drug_names = [
                        i.get("name", "") 
                        for i in interventions 
                        if i.get("type") in ["DRUG", "BIOLOGICAL"]
                    ]
                    
                    # Get outcomes
                    primary_outcomes = outcomes_module.get("primaryOutcomes", [])
                    outcome_measures = [o.get("measure", "") for o in primary_outcomes]
                    
                    record = {
                        "nct_id": id_module.get("nctId", ""),
                        "title": id_module.get("briefTitle", ""),
                        "official_title": id_module.get("officialTitle", ""),
                        "status": status_module.get("overallStatus", ""),
                        "why_stopped": status_module.get("whyStopped", ""),
                        "phase": ", ".join(design_module.get("phases", [])),
                        "study_type": design_module.get("studyType", ""),
                        "enrollment": design_module.get("enrollmentInfo", {}).get("count", ""),
                        "conditions": ", ".join(conditions_module.get("conditions", [])),
                        "drug_names": ", ".join(drug_names),
                        "intervention_types": ", ".join([i.get("type", "") for i in interventions]),
                        "primary_outcomes": ", ".join(outcome_measures),
                        "start_date": status_module.get("startDateStruct", {}).get("date", ""),
                        "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
                        "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
                        "sponsor_type": sponsor_module.get("leadSponsor", {}).get("class", ""),
                    }
                    all_records.append(record)
                
                total_downloaded += len(studies)
                progress.update(task, completed=total_downloaded)
                
                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                
                # Small delay to be respectful
                time.sleep(0.5)
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.output_dir / "clinical_trials.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} clinical trials[/green]")
        return df
    
    def download_failed_trials(self, limit: int = 10000) -> pd.DataFrame:
        """
        Download specifically terminated/withdrawn trials with reasons.
        
        Args:
            limit: Maximum number of trials
            
        Returns:
            DataFrame with failed trial data
        """
        console.print("[bold blue]📥 Downloading Failed/Terminated Trials...[/bold blue]")
        
        # Get terminated and withdrawn trials
        df = self.download_drug_trials(
            status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],
            limit=limit
        )
        
        # Filter to only those with why_stopped reasons
        failed_df = df[df["why_stopped"].notna() & (df["why_stopped"] != "")]
        
        failed_df.to_csv(self.output_dir / "failed_trials.csv", index=False)
        
        console.print(f"[green]✓ Found {len(failed_df)} trials with termination reasons[/green]")
        return failed_df
    
    def download_completed_trials(self, limit: int = 10000) -> pd.DataFrame:
        """
        Download completed Phase 3/4 trials (likely approved drugs).
        
        Args:
            limit: Maximum number of trials
            
        Returns:
            DataFrame with completed trial data
        """
        console.print("[bold blue]📥 Downloading Completed Phase 3/4 Trials...[/bold blue]")
        
        df = self.download_drug_trials(
            status=["COMPLETED"],
            phases=["PHASE3", "PHASE4"],
            limit=limit
        )
        
        df.to_csv(self.output_dir / "completed_trials.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} completed Phase 3/4 trials[/green]")
        return df
    
    def run(
        self, 
        all_trials: bool = True,
        failed_trials: bool = True,
        completed_trials: bool = True,
        limit: int = 50000
    ) -> dict:
        """
        Run the complete download pipeline.
        
        Args:
            all_trials: Download all drug trials
            failed_trials: Download terminated/withdrawn trials
            completed_trials: Download completed Phase 3/4 trials
            limit: Maximum records per category
            
        Returns:
            Dictionary with all downloaded DataFrames
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]CLINICALTRIALS.GOV DOWNLOADER[/bold cyan]")
        console.print("="*60 + "\n")
        
        results = {}
        
        if all_trials:
            results["all_trials"] = self.download_drug_trials(limit=limit)
        
        if failed_trials:
            results["failed_trials"] = self.download_failed_trials(limit=limit // 5)
        
        if completed_trials:
            results["completed_trials"] = self.download_completed_trials(limit=limit // 5)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ ClinicalTrials.gov Download Complete![/bold green]")
        for name, df in results.items():
            console.print(f"  {name}: {len(df)} records")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return results


if __name__ == "__main__":
    downloader = ClinicalTrialsDownloader()
    data = downloader.run(limit=10000)  # Start with smaller limit for testing
