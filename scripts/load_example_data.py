"""
Sample data loader for the Automotive Specs RAG system.

This script loads example automotive data into the system, including:
1. YouTube videos about car reviews
2. PDF manuals for various car models
3. Manually entered specifications data

Usage:
    python scripts/load_example_data.py

Requirements:
    - API server must be running
    - Default API key must be set in environment variable or config
"""

import argparse
import json
import os
import sys
import httpx
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings


# Example YouTube videos about car reviews
YOUTUBE_EXAMPLES = [
    {
        "url": "https://www.youtube.com/watch?v=MrJwij65z7k",
        "metadata": {
            "manufacturer": "Toyota",
            "model": "Camry",
            "year": 2023,
            "category": "sedan",
            "engine_type": "hybrid"
        }
    },
    {
        "url": "https://www.youtube.com/watch?v=uRpgk2VJX3E",
        "metadata": {
            "manufacturer": "Honda",
            "model": "Civic",
            "year": 2022,
            "category": "sedan",
            "engine_type": "gasoline"
        }
    },
    {
        "url": "https://www.youtube.com/watch?v=ozXr_f2aJq4",
        "metadata": {
            "manufacturer": "Ford",
            "model": "F-150",
            "year": 2021,
            "category": "truck",
            "engine_type": "gasoline"
        }
    }
]

# Example Bilibili videos about cars
BILIBILI_EXAMPLES = [
    {
        "url": "https://www.bilibili.com/video/BV12V4y1M7qm",
        "metadata": {
            "manufacturer": "BYD",
            "model": "Han",
            "year": 2023,
            "category": "sedan",
            "engine_type": "electric"
        }
    },
    {
        "url": "https://www.bilibili.com/video/BV1m14y1V7wV",
        "metadata": {
            "manufacturer": "Nio",
            "model": "ET7",
            "year": 2022,
            "category": "sedan",
            "engine_type": "electric"
        }
    }
]

# Example Youku videos about cars
YOUKU_EXAMPLES = [
    {
        "url": "https://v.youku.com/v_show/id_XNTk1NDAxMzgwNA==.html",
        "metadata": {
            "manufacturer": "Geely",
            "model": "Geometry A",
            "year": 2023,
            "category": "sedan",
            "engine_type": "electric"
        }
    },
    {
        "url": "https://v.youku.com/v_show/id_XNTk1MDk5OTk4OA==.html",
        "metadata": {
            "manufacturer": "Xpeng",
            "model": "P7",
            "year": 2022,
            "category": "sedan",
            "engine_type": "electric"
        }
    }
]

# Example manual entries for car specifications
MANUAL_EXAMPLES = [
    {
        "content": """
        The 2023 Toyota Camry Hybrid combines a 2.5-liter four-cylinder engine with an electric motor for a total system output of 208 horsepower.
        The hybrid system is paired with an electronically controlled continuously variable transmission (eCVT).
        Fuel economy is estimated at 51 mpg city, 53 mpg highway, and 52 mpg combined for the LE model, while the SE and XLE models achieve 44/47/46 mpg.
        The Camry Hybrid's battery pack is located under the rear seat, which allows for a generous 15.1 cubic feet of trunk space.
        Standard safety features include Toyota Safety Sense 2.5+ with pre-collision warning, automatic emergency braking, adaptive cruise control, lane departure alert with steering assist, automatic high beams, and road sign assist.
        """,
        "metadata": {
            "source": "manual",
            "source_id": "manual-toyota-camry-hybrid-2023",
            "title": "2023 Toyota Camry Hybrid Specifications",
            "manufacturer": "Toyota",
            "model": "Camry",
            "year": 2023,
            "category": "sedan",
            "engine_type": "hybrid",
            "transmission": "cvt"
        }
    },
    {
        "content": """
        The 2022 Ford Mustang GT is powered by a 5.0-liter V8 engine producing 460 horsepower and 420 lb-ft of torque.
        Transmission options include a 6-speed manual or a 10-speed automatic.
        The Mustang GT can accelerate from 0-60 mph in approximately 4.2 seconds.
        Fuel economy with the automatic transmission is EPA-rated at 15 mpg city, 24 mpg highway, and 18 mpg combined.
        Standard features include a limited-slip rear differential, selectable drive modes, and an electronic line-lock for burnouts.
        Available performance packages include the GT Performance Package with Brembo brakes, heavy-duty front springs, larger radiator, and chassis tuning.
        """,
        "metadata": {
            "source": "manual",
            "source_id": "manual-ford-mustang-gt-2022",
            "title": "2022 Ford Mustang GT Specifications",
            "manufacturer": "Ford",
            "model": "Mustang",
            "year": 2022,
            "category": "sports",
            "engine_type": "gasoline",
            "transmission": "manual"
        }
    },
    {
        "content": """
        The 2022 Tesla Model 3 Long Range features dual electric motors with a combined output of approximately 346 horsepower.
        The Long Range model has an EPA-estimated range of 358 miles on a full charge.
        Acceleration from 0-60 mph takes approximately 4.2 seconds.
        The Model 3 supports Supercharging at up to 250 kW, allowing it to gain up to 175 miles of range in 15 minutes.
        Standard features include Autopilot with Traffic-Aware Cruise Control and Autosteer.
        The Model 3 has a 15-inch central touchscreen that controls most vehicle functions and a premium audio system with 14 speakers.
        """,
        "metadata": {
            "source": "manual",
            "source_id": "manual-tesla-model-3-2022",
            "title": "2022 Tesla Model 3 Long Range Specifications",
            "manufacturer": "Tesla",
            "model": "Model 3",
            "year": 2022,
            "category": "sedan",
            "engine_type": "electric",
            "transmission": "automatic"
        }
    }
]


def make_api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    api_key: str = settings.api_key,
    base_url: str = "http://localhost:8000",
) -> httpx.Response:
    """Make a request to the API."""
    headers = {"x-token": api_key}
    url = f"{base_url}{endpoint}"

    try:
        with httpx.Client() as client:
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                if files:
                    response = client.post(url, headers=headers, data=data, files=files)
                else:
                    response = client.post(url, headers=headers, json=data)
            else:
                print(f"Unsupported method: {method}")
                return None

            return response
    except Exception as e:
        print(f"API request error: {str(e)}")
        return None


def load_youtube_examples(api_key: str, base_url: str) -> None:
    """Load YouTube examples."""
    print("Loading YouTube examples...")

    for i, example in enumerate(YOUTUBE_EXAMPLES):
        print(f"Loading YouTube example {i+1}/{len(YOUTUBE_EXAMPLES)}: {example['url']}")

        response = make_api_request(
            endpoint="/ingest/youtube?force_whisper=true",
            method="POST",
            data=example,
            api_key=api_key,
            base_url=base_url,
        )

        if response and response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
        else:
            print(f"Error: {response.text if response else 'No response'}")


def load_bilibili_examples(api_key: str, base_url: str) -> None:
    """Load Bilibili examples."""
    print("Loading Bilibili examples...")

    for i, example in enumerate(BILIBILI_EXAMPLES):
        print(f"Loading Bilibili example {i+1}/{len(BILIBILI_EXAMPLES)}: {example['url']}")

        response = make_api_request(
            endpoint="/ingest/bilibili?force_whisper=true",
            method="POST",
            data=example,
            api_key=api_key,
            base_url=base_url,
        )

        if response and response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
        else:
            print(f"Error: {response.text if response else 'No response'}")


def load_youku_examples(api_key: str, base_url: str) -> None:
    """Load Youku examples."""
    print("Loading Youku examples...")

    for i, example in enumerate(YOUKU_EXAMPLES):
        print(f"Loading Youku example {i+1}/{len(YOUKU_EXAMPLES)}: {example['url']}")

        response = make_api_request(
            endpoint="/ingest/youku?force_whisper=true",
            method="POST",
            data=example,
            api_key=api_key,
            base_url=base_url,
        )

        if response and response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
        else:
            print(f"Error: {response.text if response else 'No response'}")


def load_pdf_examples(api_key: str, base_url: str, data_dir: str) -> None:
    """Load PDF examples from the data directory."""
    print("Loading PDF examples...")

    pdf_dir = os.path.join(data_dir, "pdfs")
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return

    # Find all PDFs in the directory
    pdfs = list(Path(pdf_dir).glob("*.pdf"))

    for i, pdf_path in enumerate(pdfs):
        print(f"Loading PDF example {i+1}/{len(pdfs)}: {pdf_path.name}")

        # Extract metadata from filename
        # Format: manufacturer_model_year.pdf
        filename = pdf_path.stem
        parts = filename.split("_")

        metadata = {}
        if len(parts) >= 1:
            metadata["manufacturer"] = parts[0].capitalize()
        if len(parts) >= 2:
            metadata["model"] = parts[1].capitalize()
        if len(parts) >= 3:
            try:
                metadata["year"] = int(parts[2])
            except ValueError:
                pass

        # Add title
        metadata["title"] = f"{metadata.get('year', '')} {metadata.get('manufacturer', '')} {metadata.get('model', '')} Manual".strip()

        # Upload the PDF
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": (pdf_path.name, pdf_file, "application/pdf")}
            data = {"metadata": json.dumps(metadata)}

            response = make_api_request(
                endpoint="/ingest/pdf",
                method="POST",
                data=data,
                files=files,
                api_key=api_key,
                base_url=base_url,
            )

            if response and response.status_code == 200:
                result = response.json()
                print(f"Success: {result['message']}")
            else:
                print(f"Error: {response.text if response else 'No response'}")


def load_manual_examples(api_key: str, base_url: str) -> None:
    """Load manual entry examples."""
    print("Loading manual entry examples...")

    for i, example in enumerate(MANUAL_EXAMPLES):
        print(f"Loading manual example {i+1}/{len(MANUAL_EXAMPLES)}: {example['metadata']['title']}")

        response = make_api_request(
            endpoint="/ingest/text",
            method="POST",
            data=example,
            api_key=api_key,
            base_url=base_url,
        )

        if response and response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
        else:
            print(f"Error: {response.text if response else 'No response'}")


def main():
    parser = argparse.ArgumentParser(description="Load sample data for the Automotive Specs RAG system")
    parser.add_argument("--api-key", default=settings.api_key, help="API key for authentication")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--data-dir", default="data", help="Directory containing data files")
    parser.add_argument("--skip-youtube", action="store_true", help="Skip loading YouTube examples")
    parser.add_argument("--skip-bilibili", action="store_true", help="Skip loading Bilibili examples")
    parser.add_argument("--skip-youku", action="store_true", help="Skip loading Youku examples")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip loading PDF examples")
    parser.add_argument("--skip-manual", action="store_true", help="Skip loading manual entry examples")

    args = parser.parse_args()

    # Check if API is available
    response = make_api_request(
        endpoint="/health",
        method="GET",
        api_key=args.api_key,
        base_url=args.base_url,
    )

    if not response or response.status_code != 200:
        print(f"API is not available at {args.base_url}")
        return

    print(f"API is available at {args.base_url}")

    # Load examples
    if not args.skip_youtube:
        load_youtube_examples(args.api_key, args.base_url)

    if not args.skip_bilibili:
        load_bilibili_examples(args.api_key, args.base_url)

    if not args.skip_youku:
        load_youku_examples(args.api_key, args.base_url)

    if not args.skip_pdf:
        load_pdf_examples(args.api_key, args.base_url, args.data_dir)

    if not args.skip_manual:
        load_manual_examples(args.api_key, args.base_url)

    print("Sample data loading complete!")


if __name__ == "__main__":
    main()