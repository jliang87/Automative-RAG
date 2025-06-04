"""
End-to-end test script for the Automotive Specs RAG system.

This script tests the entire pipeline from ingestion to querying with ColBERT reranking.

Usage:
    python scripts/test_end_to_end.py

Requirements:
    - API server must be running
    - Default API key must be set in environment variable or config
"""

import argparse
import json
import os
import sys
import time
import httpx
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings


SAMPLE_DATA = {
    "content": """
    The 2024 BMW 5 Series features a range of powertrain options:
    
    1. 530i: 2.0L turbocharged inline-4 engine producing 255 horsepower and 295 lb-ft of torque
    2. 540i: 3.0L turbocharged inline-6 engine with 48V mild hybrid producing 375 horsepower and 398 lb-ft of torque
    3. i5 eDrive40: Single electric motor producing 335 horsepower and 295 lb-ft of torque with an estimated range of 295 miles
    4. i5 M60 xDrive: Dual electric motors producing 590 horsepower and 586 lb-ft of torque with an estimated range of 256 miles
    
    The 530i and 540i come with an 8-speed automatic transmission and rear-wheel drive as standard, with all-wheel drive available as an option.
    
    The i5 electric models support DC fast charging at up to 205 kW, allowing a 10-80% charge in approximately 30 minutes.
    
    All models include advanced driver assistance systems as standard, including Frontal Collision Warning, Lane Departure Warning, and Parking Assistant.
    
    The 2024 BMW 5 Series has grown in dimensions compared to its predecessor:
    - Length: 199.2 inches (+3.4 inches)
    - Width: 74.8 inches (+1.3 inches)
    - Height: 59.6 inches (+1.5 inches)
    - Wheelbase: 118.2 inches (+2.3 inches)
    """,
    "metadata": {
        "source": "manual",
        "source_id": "test-bmw-5-series-2024",
        "title": "2024 BMW 5 Series Specifications",
        "manufacturer": "BMW",
        "model": "5 Series",
        "year": 2024,
        "category": "sedan",
        "engine_type": "gasoline",
        "transmission": "automatic"
    }
}

TEST_QUERIES = [
    {
        "query": "What is the horsepower of the 2024 BMW 540i?",
        "metadata_filter": {
            "manufacturer": "BMW",
            "model": "5 Series"
        },
        "expected_keywords": ["375", "horsepower", "540i", "turbocharged"]
    },
    {
        "query": "What is the electric range of the BMW i5 eDrive40?",
        "metadata_filter": {
            "manufacturer": "BMW",
            "year": 2024
        },
        "expected_keywords": ["295", "miles", "i5", "eDrive40", "range"]
    },
    {
        "query": "How much has the BMW 5 Series grown in length compared to the previous generation?",
        "metadata_filter": {},
        "expected_keywords": ["3.4", "inches", "199.2", "length"]
    },
    {
        "query": "What is the fast charging capability of the BMW i5?",
        "metadata_filter": {
            "engine_type": "electric"
        },
        "expected_keywords": ["205", "kW", "30 minutes", "10-80%"]
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


def ingest_test_data(api_key: str, base_url: str) -> Dict:
    """Ingest test data and return the response."""
    print("Ingesting test data...")
    
    response = make_api_request(
        endpoint="/ingest/text",
        method="POST",
        data=SAMPLE_DATA,
        api_key=api_key,
        base_url=base_url,
    )
    
    if response and response.status_code == 200:
        result = response.json()
        print(f"Success: {result['message']}")
        return result
    else:
        print(f"Error: {response.text if response else 'No response'}")
        return {}


def evaluate_query_response(query_info: Dict, response_data: Dict) -> Dict:
    """Evaluate the query response against expected keywords."""
    answer = response_data.get("answer", "").lower()
    expected_keywords = [kw.lower() for kw in query_info["expected_keywords"]]
    
    results = {
        "query": query_info["query"],
        "answer": answer,
        "expected_keywords": expected_keywords,
        "found_keywords": [],
        "missing_keywords": [],
        "execution_time": response_data.get("execution_time", 0),
    }
    
    # Check for expected keywords
    for keyword in expected_keywords:
        if keyword in answer:
            results["found_keywords"].append(keyword)
        else:
            results["missing_keywords"].append(keyword)
            
    # Calculate score
    if expected_keywords:
        results["score"] = len(results["found_keywords"]) / len(expected_keywords)
    else:
        results["score"] = 0.0
        
    return results


def run_test_queries(api_key: str, base_url: str) -> List[Dict]:
    """Run test queries and evaluate responses."""
    print("Running test queries...")
    
    results = []
    
    for i, query_info in enumerate(TEST_QUERIES):
        print(f"Query {i+1}/{len(TEST_QUERIES)}: {query_info['query']}")
        
        response = make_api_request(
            endpoint="/query/",
            method="POST",
            data={
                "query": query_info["query"],
                "metadata_filter": query_info["metadata_filter"],
                "top_k": 5,
            },
            api_key=api_key,
            base_url=base_url,
        )
        
        if response and response.status_code == 200:
            response_data = response.json()
            evaluation = evaluate_query_response(query_info, response_data)
            results.append(evaluation)
            
            # Print evaluation
            print(f"  Score: {evaluation['score']:.2f}")
            print(f"  Found keywords: {', '.join(evaluation['found_keywords'])}")
            if evaluation['missing_keywords']:
                print(f"  Missing keywords: {', '.join(evaluation['missing_keywords'])}")
        else:
            print(f"  Error: {response.text if response else 'No response'}")
        
    return results


def cleanup_test_data(document_ids: List[str], api_key: str, base_url: str) -> None:
    """Clean up test data."""
    print("Cleaning up test data...")
    
    for doc_id in document_ids:
        response = make_api_request(
            endpoint=f"/ingest/documents/{doc_id}",
            method="DELETE",
            api_key=api_key,
            base_url=base_url,
        )
        
        if response and response.status_code == 200:
            print(f"Deleted document: {doc_id}")
        else:
            print(f"Error deleting document: {doc_id}")


# Add this test function to validate the Unicode handling:
def test_unicode_decoding():
    """Test function to validate Unicode decoding works correctly"""
    from src.utils.unicode_handler import decode_unicode_escapes

    # Test cases with common Unicode escape patterns
    test_cases = [
        # Chinese characters with Unicode escapes
        ("\\u6b3e", "款"),
        ("\\u8f66\\u5728\\u7ebf", "车在线"),
        ("25\\u6b3e\\u8fdc\\u9014\\u88c5\\u9970", "25款远途装饰"),

        # Already properly encoded text
        ("正常中文", "正常中文"),
        ("Normal English", "Normal English"),

        # Mixed content
        ("2023\\u5e74\\u5b9d\\u9a6cX5", "2023年宝马X5"),

        # Double-escaped
        ("\\\\u6b3e", "款"),
    ]

    print("Testing Unicode decoding...")
    for input_text, expected in test_cases:
        result = decode_unicode_escapes(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")

    print("Unicode decoding test completed.")



def main():
    parser = argparse.ArgumentParser(description="Run end-to-end tests for the Automotive Specs RAG system")
    parser.add_argument("--api-key", default=settings.api_key, help="API key for authentication")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of test data")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    
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
    
    # Ingest test data
    ingest_result = ingest_test_data(args.api_key, args.base_url)
    document_ids = ingest_result.get("document_ids", [])
    
    if not document_ids:
        print("Failed to ingest test data. Exiting tests.")
        return
        
    # Wait for indexing to complete
    print("Waiting for indexing to complete...")
    time.sleep(2)
    
    # Run test queries
    test_results = run_test_queries(args.api_key, args.base_url)
    
    # Calculate overall score
    overall_score = sum(result["score"] for result in test_results) / len(test_results) if test_results else 0
    print(f"\nOverall score: {overall_score:.2f}")
    
    # Save results to file if requested
    if args.output:
        output_data = {
            "timestamp": time.time(),
            "overall_score": overall_score,
            "query_results": test_results,
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output}")

    test_unicode_decoding()
    
    # Clean up test data
    if not args.no_cleanup:
        cleanup_test_data(document_ids, args.api_key, args.base_url)
    

if __name__ == "__main__":
    main()
