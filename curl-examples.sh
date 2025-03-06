# Authentication
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=password"

# Get System Status
curl -X GET "http://localhost:8000/ingest/status" \
  -H "x-token: default-api-key"

# Ingest YouTube Video
curl -X POST "http://localhost:8000/ingest/youtube" \
  -H "Content-Type: application/json" \
  -H "x-token: default-api-key" \
  -d '{
    "url": "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID",
    "metadata": {
      "manufacturer": "Toyota",
      "model": "Camry",
      "year": 2023
    }
  }'

# Ingest PDF
curl -X POST "http://localhost:8000/ingest/pdf" \
  -H "x-token: default-api-key" \
  -F "file=@/path/to/your/document.pdf" \
  -F "metadata={\"title\":\"2023 Toyota Camry Manual\",\"manufacturer\":\"Toyota\",\"model\":\"Camry\",\"year\":2023}"

# Ingest Manual Text
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -H "x-token: default-api-key" \
  -d '{
    "content": "The 2023 Toyota Camry comes with a 2.5L 4-cylinder engine that produces 203 horsepower. It has an EPA estimated fuel economy of 28 mpg city, 39 mpg highway, and 32 mpg combined.",
    "metadata": {
      "source": "manual",
      "source_id": "manual-entry",
      "title": "Toyota Camry Specifications",
      "manufacturer": "Toyota",
      "model": "Camry",
      "year": 2023,
      "category": "sedan",
      "engine_type": "gasoline",
      "transmission": "automatic"
    }
  }'

# Query API
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -H "x-token: default-api-key" \
  -d '{
    "query": "What is the horsepower of the 2023 Toyota Camry?",
    "metadata_filter": {
      "manufacturer": "Toyota",
      "model": "Camry",
      "year": 2023
    },
    "top_k": 5
  }'

# Get Available Manufacturers
curl -X GET "http://localhost:8000/query/manufacturers" \
  -H "x-token: default-api-key"

# Get Available Models for a Specific Manufacturer
curl -X GET "http://localhost:8000/query/models?manufacturer=Toyota" \
  -H "x-token: default-api-key"

# Get Available Categories
curl -X GET "http://localhost:8000/query/categories" \
  -H "x-token: default-api-key"

# Delete a Document
curl -X DELETE "http://localhost:8000/ingest/documents/document-id-here" \
  -H "x-token: default-api-key"

# Reset Vector Store (Dangerous Operation)
curl -X POST "http://localhost:8000/ingest/reset" \
  -H "x-token: default-api-key"
