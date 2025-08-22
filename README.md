# Compendia Backend

Compendia is an intelligent data story generation system that automatically crawls web articles, extracts data facts, organizes them into meaningful clusters, and generates compelling data narratives with visualization recommendations.

## 🚀 Features

- **Intelligent Article Crawling**: Automated web scraping from specified domains
- **Fact Extraction**: AI-powered extraction of data facts from articles using GPT-4
- **Data Organization**: Clustering and grouping of related facts
- **Narrative Generation**: Automatic creation of data stories with contextual narratives
- **Visualization Recommendations**: Smart suggestions for chart types and data visualizations
- **Multi-stage Processing Pipeline**: Comprehensive workflow from raw articles to polished data stories

## 🏗️ Architecture

The system follows a multi-stage pipeline architecture:

1. **Article Crawler**: Searches and retrieves relevant articles from specified websites
2. **Fact Extraction**: Extracts structured data facts from article content
3. **Fact Organization**: Groups and clusters related facts by topic and similarity
4. **Presentation**: Generates narratives and visualization recommendations

### Key Components

- **FastAPI Server**: RESTful API for story generation
- **GPT Integration**: Uses OpenAI GPT-4 for intelligent text processing
- **Data Models**: Comprehensive Pydantic models for structured data handling
- **Prompt Engineering**: Specialized prompts for different processing stages

## 📋 Prerequisites

- Python 3.12.6 or higher
- OpenAI API key
- Internet connection for web crawling

## 🛠️ Installation

### 1. Environment Setup

Create and activate a new conda environment:

```bash
conda create --no-default-packages -n compendia python=3.12.6
conda activate compendia
```

### 2. Install Dependencies

Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Set up your [OpenAI API](https://openai.com/api/) key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Set up your [Serper](https://serper.dev/) API key as an environment variable:

```bash
export SERPER_API_KEY="your-serper-api-key-here"
```
Set up your [SearchApi](https://www.searchapi.io/) key as an environment variable:

```bash
export SEARCH_API_KEY="your-search-api-key-here"
```

## 🚀 Running the Application

### Start the FastAPI Server

```bash
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

The server will start at `http://localhost:8000`


## 📁 Project Structure

```
compendia-backend/
├── common/                 # Shared utilities and configuration
│   ├── config.py          # System configuration
│   ├── gpt_helper.py      # OpenAI API integration
│   └── utils/             # Utility functions
├── crawler/               # Web crawling components
│   ├── searchAPI.py       # Search API interface
│   ├── serper.py          # Serper API integration
│   └── serper_crawler.py  # Main crawler implementation
├── models/                # Data models
│   └── models.py          # Pydantic models for all data structures
├── prompts/               # AI prompts for different stages
│   ├── 1_generate_search_queries.txt
│   ├── 2_extract_and_filter_para.txt
│   └── ... (15+ specialized prompts)
├── stages/                # Processing pipeline stages
│   ├── ArticleCrawler/    # Article collection stage
│   ├── FactExtraction/    # Data fact extraction stage
│   ├── FactOrganization/  # Fact clustering and organization
│   ├── Presentation/      # Narrative and visualization generation
│   └── story_generator.py # Main orchestration logic
├── results/               # Generated story outputs
├── main.py               # FastAPI application entry point
└── requirements.txt      # Python dependencies
```

## 🔄 Processing Pipeline

1. **Search Query Generation**: Creates targeted search queries from user input
2. **Article Crawling**: Retrieves relevant articles from specified websites
3. **Paragraph Extraction**: Filters and extracts relevant paragraphs
4. **Fact Extraction**: Identifies and extracts structured data facts
5. **Data Validation**: Validates and refines extracted data
6. **Fact Clustering**: Groups similar facts by topic and content
7. **Fact Merging**: Combines related facts into coherent units
8. **Narrative Generation**: Creates compelling stories from fact clusters
9. **Visualization Recommendation**: Suggests appropriate chart types
10. **Story Assembly**: Combines all elements into final data story

This project is part of the Compendia research initiative.

---

**Happy Data Storytelling! ✨**
