# Retrieval Augmented Generation (RAG) Medical Symptom Checker

This directory contains a Retrieval Augmented Generation (RAG) system that helps users find information about medical conditions by searching through a comprehensive disease database. The system uses Google's Gemini AI model for both embeddings and text generation.

## Overview

The RAG system provides two search modes:
1. **Disease Name Search**: Direct lookup by disease name
2. **Semantic Symptom Search**: Natural language search using embeddings to find diseases based on symptom descriptions

## How It Works

### RAG Pipeline

1. **Data Preparation**:
   - Downloads a comprehensive disease dataset from Kaggle
   - Cleans and formats the data into a structured format
   - Aggregates symptoms, causes, treatments, diagnoses, and complications for each disease

2. **Embedding Creation**:
   - Creates text documents for each disease by combining all relevant information
   - Uses Google's `text-embedding-004` model to generate vector embeddings for each disease document
   - Stores embeddings in a matrix for efficient similarity search

3. **Retrieval**:
   - **Mode 1 (Disease Name)**: Exact match lookup in the disease database
   - **Mode 2 (Symptom Search)**: 
     - Embeds the user's symptom query
     - Uses cosine similarity to find the top N most relevant diseases
     - Retrieves the full context for those diseases

4. **Generation**:
   - Augments the user's query with retrieved disease context
   - Uses Gemini 2.5 Flash model to generate comprehensive answers
   - Provides information about symptoms, causes, treatments, complications, prognosis, and more

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Kaggle account (for dataset download)
- Internet connection (for initial dataset download and API calls)

## Installation

### 1. Install Dependencies

Navigate to the `Back_End` directory and install the required packages:

```bash
cd Back_End
pip install -r requirements.txt
```

Required packages include:
- `google-generativeai` - For Gemini AI and embeddings
- `pandas` - For data manipulation
- `numpy` - For vector operations
- `kagglehub` - For dataset download
- `openpyxl` - For Excel file reading
- `python-dotenv` - For environment variable management

### 2. Set Up Kaggle API (Optional)

If you haven't already, you may need to set up Kaggle API credentials for dataset download. The `kagglehub` library typically handles this automatically, but you can configure it if needed.

### 3. Configure Environment Variables

Create a `.env` file in the `Back_End` directory (or ensure one exists) with your Gemini API key:

```env
GEMINI_API_KEY=your-api-key-here
```

**How to get a Gemini API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key and add it to your `.env` file

## Running the RAG System

### Quick Start

1. Navigate to the AI directory:
   ```bash
   cd Back_End/AI
   ```

2. Run the setup script:
   ```bash
   python setup.py
   ```

### What Happens When You Run

1. **Initial Setup** (first run only):
   - Downloads the disease dataset from Kaggle (~30 seconds)
   - Cleans and processes the data
   - Creates embeddings for all diseases (~1-2 minutes depending on dataset size)
   - Prints "READY" when complete

2. **Interactive Chat Loop**:
   - You'll see a menu with two options:
     - Option 1: Search by Disease Name
     - Option 2: Search by Symptoms (semantic search)
   - Type `quit` to exit

### Usage Examples

#### Example 1: Search by Disease Name
```
Enter your choice (1 or 2): 1
Enter the disease name: malaria
```
The system will retrieve information about malaria and generate a comprehensive answer.

#### Example 2: Search by Symptoms
```
Enter your choice (1 or 2): 2
Describe how you feel (full sentences are ok): I have a high fever, headache, and feel very tired
```
The system will:
- Embed your query
- Find the top 3 most similar diseases
- Generate a differential diagnosis with information about each disease

## File Structure

```
AI/
├── setup.py              # Main setup and initialization script
├── processing.py         # Core RAG functions (embedding, retrieval, generation)
├── chatloop.py          # Interactive chat interface
├── keyword_search.py    # Legacy keyword-based search (not used)
└── README.md            # This file
```

## Key Functions

### `processing.py`
- `clean_data()` - Cleans and formats the dataset
- `format_data()` - Melts wide-format data into long format
- `prep_RAG()` - Prepares data for RAG by aggregating information
- `setup_embedding()` - Creates embeddings for all disease documents
- `embed_query()` - Embeds user queries
- `find_nearest_neighbors()` - Finds most similar diseases using cosine similarity
- `retrieve_context()` - Retrieves disease information by name
- `generate_answer()` - Generates answers for single disease queries
- `generate_differential_answer()` - Generates differential diagnosis answers

### `setup.py`
- Downloads and processes the disease dataset
- Sets up the embedding model
- Initializes the chat loop

### `chatloop.py`
- Provides the interactive user interface
- Handles user input and routing
- Manages the conversation flow

## Troubleshooting

### Issue: "GEMINI_API_KEY environment variable is not set"
**Solution**: Make sure you have a `.env` file in the `Back_End` directory with your API key.

### Issue: Dataset download fails
**Solution**: 
- Check your internet connection
- Ensure you have sufficient disk space
- Verify Kaggle dataset is accessible

### Issue: Embedding creation fails
**Solution**:
- Verify your API key is valid and has access to the embedding model
- Check your internet connection
- Ensure you have sufficient API quota

### Issue: Import errors
**Solution**:
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure you're running from the correct directory
- Check that Python version is 3.8 or higher

## Performance Notes

- **First Run**: Takes 2-3 minutes to download dataset and create embeddings
- **Subsequent Runs**: Much faster as dataset is cached locally
- **Query Time**: 
  - Disease name search: < 1 second
  - Symptom search: 2-5 seconds (includes embedding + similarity search + generation)

## Limitations

- This is a demonstration system and should not be used for actual medical diagnosis
- Always consult with healthcare professionals for medical advice
- The system is limited to diseases in the downloaded dataset
- API usage is subject to Google's rate limits and pricing

## Future Enhancements

- Caching embeddings to disk for faster startup
- Batch processing for multiple queries
- Integration with web interface
- Additional search modes (by region, severity, etc.)
- Confidence scores for search results

## License

Please refer to the main project README for license information.

