"""
FDA Orange Book Downloader

Downloads the FDA Orange Book dataset containing all FDA-approved drugs
with patent and exclusivity information.

Source: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
"""

import os
import sys
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.append(str(Path(__file__).parent.parent))
from config import FDA_ORANGE_BOOK, RAW_DATA_DIR

console = Console()


class FDAOrangeBookDownloader:
    """Downloads and parses FDA Orange Book data."""
    
    def __init__(self):
        self.output_dir = FDA_ORANGE_BOOK["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_url = FDA_ORANGE_BOOK["download_url"]
    
    def download(self) -> Path:
        """
        Download the Orange Book ZIP file from FDA.
        
        Returns:
            Path to the downloaded ZIP file
        """
        console.print("[bold blue]📥 Downloading FDA Orange Book...[/bold blue]")
        
        zip_path = self.output_dir / "orange_book.zip"
        
        # Full browser headers to avoid being blocked by FDA
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files",
            "Connection": "keep-alive",
        }
        
        # Try multiple URLs (FDA sometimes changes these)
        urls_to_try = [
            FDA_ORANGE_BOOK.get("data_url", ""),
            self.download_url,
            "https://www.fda.gov/media/76860/download",  # Without attachment param
        ]
        
        for url in urls_to_try:
            if not url:
                continue
            
            try:
                console.print(f"  Trying: {url[:50]}...")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Downloading...", total=None)
                    
                    # Use session to maintain cookies
                    session = requests.Session()
                    session.headers.update(headers)
                    
                    response = session.get(url, stream=True, timeout=120, allow_redirects=True)
                    response.raise_for_status()
                    
                    # Check if we got a ZIP file (not HTML error page)
                    content_type = response.headers.get('Content-Type', '')
                    if 'html' in content_type.lower():
                        console.print(f"[yellow]  Got HTML instead of ZIP[/yellow]")
                        continue
                    
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    progress.update(task, completed=True)
                
                # Verify it's a valid ZIP
                if zip_path.stat().st_size < 1000:
                    console.print(f"[yellow]  File too small, likely error page[/yellow]")
                    continue
                
                console.print(f"[green]✓ Downloaded to {zip_path}[/green]")
                return zip_path
                
            except requests.RequestException as e:
                console.print(f"[yellow]  Failed: {str(e)[:50]}[/yellow]")
                continue
        
        console.print("[red]✗ All download URLs failed[/red]")
        raise Exception("Could not download FDA Orange Book from any URL")
    
    def extract(self, zip_path: Path) -> list:
        """
        Extract the Orange Book ZIP file.
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            List of extracted file paths
        """
        console.print("[bold blue]📦 Extracting files...[/bold blue]")
        
        extracted_files = []
        extract_dir = self.output_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = [extract_dir / name for name in zip_ref.namelist()]
        
        console.print(f"[green]✓ Extracted {len(extracted_files)} files[/green]")
        return extracted_files
    
    def parse_products(self, extracted_dir: Path) -> pd.DataFrame:
        """
        Parse the products.txt file from Orange Book.
        
        Contains: Drug name, active ingredient, dosage form, route, 
                  application number, product number, approval date, etc.
        """
        products_file = extracted_dir / "products.txt"
        
        if not products_file.exists():
            # Try to find it in subdirectories
            for f in extracted_dir.rglob("products.txt"):
                products_file = f
                break
        
        if not products_file.exists():
            console.print("[yellow]⚠ products.txt not found[/yellow]")
            return pd.DataFrame()
        
        console.print(f"[bold blue]📄 Parsing {products_file.name}...[/bold blue]")
        
        # Orange Book uses ~ as delimiter
        df = pd.read_csv(
            products_file, 
            sep='~', 
            encoding='latin-1',
            low_memory=False
        )
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        console.print(f"[green]✓ Parsed {len(df)} products[/green]")
        return df
    
    def parse_patents(self, extracted_dir: Path) -> pd.DataFrame:
        """Parse the patent.txt file from Orange Book."""
        patent_file = extracted_dir / "patent.txt"
        
        if not patent_file.exists():
            for f in extracted_dir.rglob("patent.txt"):
                patent_file = f
                break
        
        if not patent_file.exists():
            console.print("[yellow]⚠ patent.txt not found[/yellow]")
            return pd.DataFrame()
        
        console.print(f"[bold blue]📄 Parsing {patent_file.name}...[/bold blue]")
        
        df = pd.read_csv(
            patent_file,
            sep='~',
            encoding='latin-1',
            low_memory=False
        )
        
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        console.print(f"[green]✓ Parsed {len(df)} patent records[/green]")
        return df
    
    def parse_exclusivity(self, extracted_dir: Path) -> pd.DataFrame:
        """Parse the exclusivity.txt file from Orange Book."""
        excl_file = extracted_dir / "exclusivity.txt"
        
        if not excl_file.exists():
            for f in extracted_dir.rglob("exclusivity.txt"):
                excl_file = f
                break
        
        if not excl_file.exists():
            console.print("[yellow]⚠ exclusivity.txt not found[/yellow]")
            return pd.DataFrame()
        
        console.print(f"[bold blue]📄 Parsing {excl_file.name}...[/bold blue]")
        
        df = pd.read_csv(
            excl_file,
            sep='~',
            encoding='latin-1',
            low_memory=False
        )
        
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        console.print(f"[green]✓ Parsed {len(df)} exclusivity records[/green]")
        return df
    
    def run(self) -> dict:
        """
        Run the complete download and parse pipeline.
        
        Returns:
            Dictionary with DataFrames for products, patents, exclusivity
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]FDA ORANGE BOOK DOWNLOADER[/bold cyan]")
        console.print("="*60 + "\n")
        
        # Download
        zip_path = self.download()
        
        # Extract
        extracted_files = self.extract(zip_path)
        extracted_dir = self.output_dir / "extracted"
        
        # Parse all files
        products_df = self.parse_products(extracted_dir)
        patents_df = self.parse_patents(extracted_dir)
        exclusivity_df = self.parse_exclusivity(extracted_dir)
        
        # Save processed data
        if not products_df.empty:
            products_df.to_csv(self.output_dir / "products.csv", index=False)
        if not patents_df.empty:
            patents_df.to_csv(self.output_dir / "patents.csv", index=False)
        if not exclusivity_df.empty:
            exclusivity_df.to_csv(self.output_dir / "exclusivity.csv", index=False)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ FDA Orange Book Download Complete![/bold green]")
        console.print(f"  Products: {len(products_df)}")
        console.print(f"  Patents: {len(patents_df)}")
        console.print(f"  Exclusivity: {len(exclusivity_df)}")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return {
            "products": products_df,
            "patents": patents_df,
            "exclusivity": exclusivity_df,
        }
    
    def download_via_openfda(self, limit: int = 50000) -> pd.DataFrame:
        """
        Fallback: Download approved drug data via openFDA NDC API.
        
        Args:
            limit: Maximum number of records
            
        Returns:
            DataFrame with approved drug data
        """
        console.print("[bold blue]📥 Using openFDA NDC API as fallback...[/bold blue]")
        
        from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
        
        all_records = []
        skip = 0
        max_per_query = 1000
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading NDC data...", total=limit)
            
            while skip < limit:
                try:
                    batch_size = min(max_per_query, limit - skip)
                    url = f"https://api.fda.gov/drug/ndc.json?limit={batch_size}&skip={skip}"
                    
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code != 200:
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    
                    if not results:
                        break
                    
                    for item in results:
                        ingredients = item.get("active_ingredients", [])
                        ingredient_str = "; ".join([
                            f"{i.get('name', '')} ({i.get('strength', '')})" 
                            for i in ingredients
                        ])
                        
                        record = {
                            "product_ndc": item.get("product_ndc", ""),
                            "generic_name": item.get("generic_name", ""),
                            "brand_name": item.get("brand_name", ""),
                            "labeler_name": item.get("labeler_name", ""),
                            "active_ingredients": ingredient_str,
                            "dosage_form": item.get("dosage_form", ""),
                            "route": ", ".join(item.get("route", [])),
                            "product_type": item.get("product_type", ""),
                            "marketing_category": item.get("marketing_category", ""),
                            "application_number": item.get("application_number", ""),
                        }
                        all_records.append(record)
                    
                    skip += len(results)
                    progress.update(task, completed=skip)
                    
                except Exception as e:
                    console.print(f"[yellow]  Error: {e}[/yellow]")
                    break
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.output_dir / "products.csv", index=False)
        
        console.print(f"[green]✓ Downloaded {len(df)} products via openFDA[/green]")
        return df
    
    def run(self) -> dict:
        """
        Run the complete download and parse pipeline.
        Falls back to openFDA NDC API if Orange Book download fails.
        
        Returns:
            Dictionary with DataFrames for products, patents, exclusivity
        """
        console.print("\n" + "="*60)
        console.print("[bold cyan]FDA ORANGE BOOK DOWNLOADER[/bold cyan]")
        console.print("="*60 + "\n")
        
        try:
            # Try to download Orange Book ZIP
            zip_path = self.download()
            
            # Extract
            extracted_files = self.extract(zip_path)
            extracted_dir = self.output_dir / "extracted"
            
            # Parse all files
            products_df = self.parse_products(extracted_dir)
            patents_df = self.parse_patents(extracted_dir)
            exclusivity_df = self.parse_exclusivity(extracted_dir)
            
        except Exception as e:
            console.print(f"[yellow]⚠ Orange Book download failed: {e}[/yellow]")
            console.print("[yellow]  Falling back to openFDA NDC API...[/yellow]\n")
            
            # Fallback to openFDA
            products_df = self.download_via_openfda(limit=10000)
            patents_df = pd.DataFrame()
            exclusivity_df = pd.DataFrame()
        
        # Save processed data
        if not products_df.empty:
            products_df.to_csv(self.output_dir / "products.csv", index=False)
        if not patents_df.empty:
            patents_df.to_csv(self.output_dir / "patents.csv", index=False)
        if not exclusivity_df.empty:
            exclusivity_df.to_csv(self.output_dir / "exclusivity.csv", index=False)
        
        # Summary
        console.print("\n" + "="*60)
        console.print("[bold green]✓ FDA Drug Data Download Complete![/bold green]")
        console.print(f"  Products: {len(products_df)}")
        console.print(f"  Patents: {len(patents_df)}")
        console.print(f"  Exclusivity: {len(exclusivity_df)}")
        console.print(f"  Output: {self.output_dir}")
        console.print("="*60 + "\n")
        
        return {
            "products": products_df,
            "patents": patents_df,
            "exclusivity": exclusivity_df,
        }


if __name__ == "__main__":
    downloader = FDAOrangeBookDownloader()
    data = downloader.run()
