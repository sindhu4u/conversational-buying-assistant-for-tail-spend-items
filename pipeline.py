#!/usr/bin/env python3
# enhanced_agentic_pipeline_with_preferences_and_rag_compliance.py
import json
import re
import pandas as pd
import torch
import os
from serpapi.google_search import GoogleSearch
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import PyPDF2  # For PDF text extraction
from datetime import datetime
import ast

# === Firebase Integration ===
import firebase_admin
from firebase_admin import credentials, storage, firestore
from typing import Optional
import hashlib

# Set random seed for reproducibility
torch.random.manual_seed(0)

# === PDF Text Extraction ===
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""
    return text
# === Firebase Manager ===
class FirebaseManager:
    """
    Manages Firebase Cloud Storage and Firestore operations for CSV storage and query logging.
    """
    def __init__(self, credentials_path: str, storage_bucket: str):
        """Initialize Firebase with credentials."""
        try:
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                "storageBucket": storage_bucket
            })

            self.bucket = storage.bucket()
            self.db = firestore.client(database_id="csv-logs")
            print("✅ Firebase initialized successfully")
        except Exception as e:
            print(f"❌ Firebase initialization error: {e}")
            raise e

    def upload_csv_to_storage(self, local_file_path: str, product_name: str, file_type: str = "scraped") -> Optional[str]:
        """
        Upload CSV file to Firebase Storage in organized folder structure.

        Args:
            local_file_path: Path to local CSV file
            product_name: Name of the product (used for folder organization)
            file_type: Type of file - "scraped" or "reasoned"

        Returns:
            Firebase storage path if successful, None otherwise
        """
        try:
            if not os.path.exists(local_file_path):
                print(f"❌ File not found: {local_file_path}")
                return None

            # Create organized path: Data/product_name/filename
            file_name = os.path.basename(local_file_path)
            firebase_path = f"Data/{product_name}/{file_name}"

            # Upload to Firebase Storage
            blob = self.bucket.blob(firebase_path)
            blob.upload_from_filename(local_file_path)

            print(f"✅ Uploaded {file_type} CSV to Firebase: {firebase_path}")
            return firebase_path

        except Exception as e:
            print(f"❌ Error uploading CSV to Firebase: {e}")
            return None

    def generate_query_hash(self, query: str) -> str:
        """Generate a hash for the query to use as document ID."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def log_query_to_firestore(self, query: str, product_name: str, csv_file_name: str) -> bool:
        """
        Log query details to Firestore Logs collection.

        Args:
            query: User query string
            product_name: Name of the product
            csv_file_name: Name of the reasoned CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract date from CSV filename (format: query_result_*_YYYYMMDD_HHMMSS.csv)
            date_match = re.search(r'(\d{8})_(\d{6})', csv_file_name)
            if date_match:
                date_str = date_match.group(1)
                time_str = date_match.group(2)
                query_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            else:
                query_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create log entry
            log_entry = {
                "query": query,
                "product_name": product_name,
                "date": query_date,
                "csv_file_name": csv_file_name,
                "firebase_path": f"Data/{product_name}/{csv_file_name}",
                "timestamp": firestore.SERVER_TIMESTAMP
            }

            # Use query hash as document ID for easy retrieval
            query_hash = self.generate_query_hash(query)

            # Store in Logs collection
            self.db.collection("Logs").document(query_hash).set(log_entry)

            print(f"✅ Logged query to Firestore: {query_hash}")
            return True

        except Exception as e:
            print(f"❌ Error logging to Firestore: {e}")
            return False

    def check_query_exists(self, query: str) -> Optional[dict]:
        """
        Check if query exists in logs and retrieve CSV info.

        Args:
            query: User query string

        Returns:
            Log entry dictionary if found, None otherwise
        """
        try:
            query_hash = self.generate_query_hash(query)
            doc_ref = self.db.collection("Logs").document(query_hash)
            doc = doc_ref.get()

            if doc.exists:
                log_data = doc.to_dict()
                print(f"✅ Found cached query: {query_hash}")
                return log_data
            else:
                print(f"ℹ️ Query not found in cache: {query_hash}")
                return None

        except Exception as e:
            print(f"❌ Error checking query cache: {e}")
            return None

    def download_csv_from_storage(self, firebase_path: str, local_destination: str) -> bool:
        """
        Download CSV file from Firebase Storage.

        Args:
            firebase_path: Path in Firebase Storage
            local_destination: Local file path to save

        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(firebase_path)
            blob.download_to_filename(local_destination)
            print(f"✅ Downloaded CSV from Firebase: {firebase_path}")
            return True

        except Exception as e:
            print(f"❌ Error downloading CSV from Firebase: {e}")
            return False
class AgenticPlanner:
    """
    Enhanced Groq-based planner using LLaMA 3.3 70B that handles agent selection,
    query conversion, and autonomous follow-up detection.
    """
    def __init__(self, groq_api_key: str):
        """Initialize with Groq API key."""
        self.client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"

        # Enhanced prompt template with autonomous query type detection
        self.prompt_template ="""You are an overseer LLM who acts as an agentic planner with query conversion capabilities and justification routing.

SOURCES: myntra, flipkart, amazon, snapdeal, ajio, nykaa, paytm mall, shopclues, tata cliq, reliancedigital, croma, bigbasket, grofers, pepperfry, urban ladder, lenskart, pharmeasy, 1mg, ebay, alibaba, aliexpress, walmart, target, bestbuy

PRICE FORMATS:
Thousands: 1000, 1,000, 1K, 1k, one thousand
Ten Thousands: 10000, 10,000, 10K, 10k, ten thousand
Lakhs: 100000, 1,00,000, 1L, 1l, 1 lakh, one lakh
Crores: 10000000, 1,00,00,000, 1Cr, 1cr, 1 crore, one crore
Under/Below: under 50k, below 1 lakh, less than 10000
Above/Over: above 25k, over 1 lakh, more than 50000
Range: 10k-50k, 10000 to 50000, between 25k and 75k
Exact: exactly 50000, around 1 lakh, approximately 75k
Currency: ₹50000, Rs. 50000, INR 50000, $500, USD 500, 50000 rupees, 500 dollars, 50k bucks, 1 lakh ka, paisa 50000

JUSTIFICATION INDICATORS:
why, justify, explain, reason, because, how, what makes, tell me why, explain why, justify this, why this, what's the reason, how is this better, what's special, why should I choose, why recommend, rationale, basis, criteria, logic, proof, evidence, support, defend, validate, convince me, make case for, argument for, pros and cons, benefits, advantages, disadvantages, comparison, analysis, breakdown, detailed explanation, elaborate, clarify, demonstrate, show me why, prove it, back up, substantiate, reasoning behind, thought process, decision basis, selection criteria, evaluation, assessment, review analysis, what factors, considerations, metrics, parameters, evaluation criteria, judgment basis, recommendation logic, preference reasoning, choice justification, decision rationale, selection reasoning, why better, why worse, why preferred, why chosen, why selected, why recommended, analysis of results, result explanation, outcome reasoning, finding justification, conclusion basis, verdict reasoning, assessment rationale, evaluation logic, judgment explanation, choice analysis, option comparison, alternative analysis, preference explanation, selection analysis, recommendation analysis, decision analysis, choice reasoning, option justification, alternative reasoning, preference rationale, selection rationale, recommendation rationale, decision rationale, choice rationale, option rationale, alternative rationale

FOLLOW-UP INDICATORS:

Basic Referential Patterns:
get me, show me, filter, only, just, narrow down, these, those, them, it, above results, previous ones, what is the most affordable, what is the cheapest, get me the most affordable, get me the cheapest, which is cheapest, which is most affordable, show cheapest, show most affordable

Referential with Comparison/Selection:
out of these, from these, among these, within these results, from the above, from those shown, which of these, out of them, among them, between these, from this list, out of the previous ones, among the results, from what you showed

Ambiguous Action Verbs (Context-Dependent):
filter, sort, pick, select, choose, recommend, suggest, highlight, narrow, refine, find, get, show, display, list, give me

Implicit Reference Patterns (No Product Noun):
get me those under [price], show me ones with [criteria], filter by [attribute], sort by [attribute], just the [adjective] ones, only [adjective] options, pick [superlative], select [comparison]

Price/Criteria Filtering Without Product Context:
under [amount], above [amount], below [amount], over [amount], less than [amount], more than [amount], between [range], within [range], around [amount]

Quality/Characteristic Filters:
most reliable, best quality, most trusted, highest rated, most popular, well-reviewed, top-performing, most durable, most recommended, safest, most effective, premium options, budget-friendly ones, value for money, cost-effective, economical ones

Selection/Action Patterns:
pick the best, select top, choose from, recommend from these, suggest from above, find the cheapest, get the most expensive, show the highest, display the lowest

Context-Dependent Single Words/Short Phrases:
reliable ones, trusted ones, good ones, better ones, cheaper ones, expensive ones, quality ones, branded ones, recommended ones, suitable ones, appropriate ones, filter products, sort items, best deals, top picks

AMBIGUOUS QUERY CONVERSIONS:
most affordable → Which products have the lowest price?
cheapest → Which products have the lowest price?
budget-friendly → Which products have price less than inferred amount?
economical → Which products have the lowest price?
pocket-friendly → Which products have price less than inferred amount?
value for money → Which products have the best price to rating ratio?
best → Which products have the highest rating?
top-rated → Which products have the highest rating?
highly rated → Which products have rating more than 4 stars?
good quality → Which products have rating more than 4 stars?
excellent → Which products have rating more than 4.5 stars?
premium → Which products have the highest price?
better options → Which products have higher rating?
cheaper alternatives → Which products have lower price?
more expensive → Which products have higher price?
upgrade options → Which products have higher price and rating?
downgrade options → Which products have lower price?
popular → Which products have more than 100 reviews?
well-reviewed → Which products have more than 50 reviews?
trusted → Which products have more than 200 reviews and rating more than 4 stars?
tried and tested → Which products have more than 100 reviews?
most reliable → Which products have more than 200 reviews and rating more than 4 stars?
reliable ones → Which products have more than 200 reviews and rating more than 4 stars?
Apple products → Which products contain Apple in the title?
Samsung phones → Which products contain Samsung in the title?
Nike shoes → Which products contain Nike in the title?
gaming laptops → Which products contain gaming laptop in the title?
wireless earbuds → Which products contain wireless earbuds in the title?
from myntra → Which products are from the source myntra?
available on flipkart → Which products are from the source flipkart?
amazon exclusive → Which products are from the source amazon?
reliable sellers → Which products are from trusted sources?

THE BELOW AMBIGUOUS QUERIES ARE OFTEN FOLLOW UP QUERIES YOU SHOULD LEARN IT AND UNDERSTAND TO DIFFERENTIATE WHETHER IT IS A FOLLOW UP OR NOT BASED ON CONTEXT TOO
most affordable → Which products have the lowest price?
cheapest → Which products have the lowest price?
budget-friendly → Which products have price less than inferred amount?
economical → Which products have the lowest price?
pocket-friendly → Which products have price less than inferred amount?
value for money → Which products have the best price to rating ratio?
best → Which products have the highest rating?
top-rated → Which products have the highest rating?
highly rated → Which products have rating more than 4 stars?
good quality → Which products have rating more than 4 stars?
excellent → Which products have rating more than 4.5 stars?
premium → Which products have the highest price?
better options → Which products have higher rating?
cheaper alternatives → Which products have lower price?
more expensive → Which products have higher price?
upgrade options → Which products have higher price and rating?
downgrade options → Which products have lower price?
popular → Which products have more than 100 reviews?
well-reviewed → Which products have more than 50 reviews?
trusted → Which products have more than 200 reviews and rating more than 4 stars?
tried and tested → Which products have more than 100 reviews?
most reliable → Which products have more than 200 reviews and rating more than 4 stars?

FOLLOW-UP QUERY DETECTION RULES:
1. If query contains referential pronouns (these, those, them, it) + any filtering/sorting criteria → FOLLOW-UP
2. If query starts with action verbs (filter, sort, pick, select, get, show) without specific product nouns → FOLLOW-UP
3. If query contains price/criteria constraints without product context and previous results exist → FOLLOW-UP
4. If query uses comparative/superlative adjectives without product specification → FOLLOW-UP
5. If query is ambiguous but contains selection/filtering intent and session has previous results → FOLLOW-UP

EXPANDED FOLLOW-UP EXAMPLES:
• "get me those under 40000" → FOLLOW-UP
• "filter products" → FOLLOW-UP
• "sort by price" → FOLLOW-UP
• "show me the reliable ones" → FOLLOW-UP
• "just the discounted items" → FOLLOW-UP
• "pick the most popular" → FOLLOW-UP
• "get cheaper alternatives" → FOLLOW-UP
• "filter by brand" → FOLLOW-UP
• "under 25000" → FOLLOW-UP (if previous results exist)
• "above 4 stars rating" → FOLLOW-UP (if previous results exist)
• "from amazon only" → FOLLOW-UP
• "with free delivery" → FOLLOW-UP
• "most affordable options" → FOLLOW-UP (if previous results exist)
• "better quality ones" → FOLLOW-UP
• "select top 5" → FOLLOW-UP
• "narrow down to apple products" → FOLLOW-UP
• "highlight premium models" → FOLLOW-UP
• "show discounted" → FOLLOW-UP
• "get latest models" → FOLLOW-UP (if previous results exist)
• "filter electronics" → FOLLOW-UP (if previous results contain electronics)

Given:
• User query: "{query}"
• User preferences: {preferences}
• {hit_info}

Your responsibilities:
1. Identify if query is asking for justification/explanation
2. Convert user queries to TAPAS-optimized "Which" or "What" questions
3. Determine agent execution plan

JUSTIFICATION QUERY DETECTION:
If the query contains justification indicators (why, justify, explain, reason, etc.), route to justification agent.

TAPAS Query Conversion Guidelines:
- TAPAS works best with "Which" or "What" questions that can be answered by looking at table data
- Convert price filters: "under 50000" → "Which products have price less than 50000?"
- Convert rating filters: "rating more than 4" → "Which products have rating more than 4 stars?"
- Convert source filters: "from myntra" → "Which products are from the source myntra?"
- Convert comparative queries: "cheapest" → "Which products have the lowest price?"
- Convert brand queries: "MacBook" → "Which products contain MacBook in the title?"
- Convert review queries: "more than 100 reviews" → "Which products have more than 100 reviews?"
- Convert reliability queries: "most reliable" → "Which products have more than 200 reviews and rating more than 4 stars?"

Examples:
• "laptops under 50000" → "Which products have price less than 50000?"
• "phones with rating more than 4 stars" → "Which products have rating more than 4 stars?"
• "from myntra" → "Which products are from the source myntra?"
• "cheapest options" → "Which products have the lowest price?"
• "best rated products" → "Which products have the highest rating?"
• "MacBook Pro models" → "Which products contain MacBook Pro in the title?"
• "products with more than 100 reviews" → "Which products have more than 100 reviews?"
• "expensive items" → "Which products have the highest price?"
• "under 30000 and good rating" → "Which products have price less than 30000 and rating more than 4 stars?"
• "Out of these what are the most reliable ones" → "Which products have more than 200 reviews and rating more than 4 stars?"
• "From these show me the cheapest" → "Which products have the lowest price?"
• "Among them which are highly rated" → "Which products have rating more than 4 stars?"
• "get me those under 40000" → "Which products have price less than 40000?"
• "filter products" → "Which products match the current criteria?"

There are three agents:
1. The scraper needs product keywords (nouns like: watches, macbook, laptop, chairs, phones, etc.)
2. The reasoner filters scraped data using the converted TAPAS query
3. The justifier explains results based on user preferences

Query Analysis:
- Justification queries ask for explanations: "why this product", "justify your choice", "explain the recommendation"
- Follow-up queries filter previous results: "get me cheaper options", "filter from myntra", "show products under 30000", "out of these what are the most reliable ones", "get me those under 40000", "filter products"
- New queries require both scraping and reasoning: "find me laptops under 50000", "get me tissot watches"

Return only the following JSON, with no extra text:

For JUSTIFICATION queries:
{{
  "query_type": "justification",
  "steps": [
    {{
      "agent": "justifier",
      "args": {{
        "query": "{query}",
        "preferences": {preferences}
      }}
    }}
  ]
}}

For NEW queries (not follow-up):
{{
  "query_type": "new",
  "tapas_query": "converted TAPAS question using Which/What format",
  "steps": [
    {{
      "agent": "scraper",
      "args": {{
        "keywords": ["product_name"]
      }}
    }},
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "converted TAPAS question"
      }}
    }}
  ]
}}

For FOLLOW-UP queries:
{{
  "query_type": "follow_up",
  "tapas_query": "converted TAPAS question using Which/What format",
  "steps": [
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "converted TAPAS question"
      }}
    }}
  ]
}}

Example 1 - Justification query: "why did you recommend this laptop"
{{
  "query_type": "justification",
  "steps": [
    {{
      "agent": "justifier",
      "args": {{
        "query": "why did you recommend this laptop",
        "preferences": {preferences}
      }}
    }}
  ]
}}

Example 2 - New query: "get me good MacBook under 1,00,000 and rating more than 4.5"
{{
  "query_type": "new",
  "tapas_query": "Which products contain MacBook in the title and have price less than 100000 and rating more than 4.5 stars?",
  "steps": [
    {{
      "agent": "scraper",
      "args": {{
        "keywords": ["MacBook"]
      }}
    }},
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "Which products contain MacBook in the title and have price less than 100000 and rating more than 4.5 stars?"
      }}
    }}
  ]
}}

Example 3 - Follow-up query: "get me these under 80,000"
{{
  "query_type": "follow_up",
  "tapas_query": "Which products have price less than 80000?",
  "steps": [
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "Which products have price less than 80000?"
      }}
    }}
  ]
}}

Example 4 - Follow-up query: "Out of these what are the most reliable ones"
{{
  "query_type": "follow_up",
  "tapas_query": "Which products have more than 200 reviews and rating more than 4 stars?",
  "steps": [
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "Which products have more than 200 reviews and rating more than 4 stars?"
      }}
    }}
  ]
}}

Example 5 - Follow-up query: "get me those under 40000"
{{
  "query_type": "follow_up",
  "tapas_query": "Which products have price less than 40000?",
  "steps": [
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "Which products have price less than 40000?"
      }}
    }}
  ]
}}

Example 6 - Follow-up query: "filter products"
{{
  "query_type": "follow_up",
  "tapas_query": "Which products match the current criteria?",
  "steps": [
    {{
      "agent": "reasoner",
      "args": {{
        "tapas_query": "Which products match the current criteria?"
      }}
    }}
  ]
}}
"""

    def run(self, query: str, preferences: list, previous_context: dict = None) -> dict:
        """Generate planning response with autonomous query type detection."""
        try:
            # Determine context information based on previous session state
            hit_info = ""
            if previous_context and previous_context.get("has_previous_results"):
                hit_info = f"Previous results available from: {previous_context.get('previous_product', 'unknown product')} query"
            else:
                hit_info = "No previous query context"

            prompt_text = self.prompt_template.format(
                query=query,
                preferences=preferences,
                hit_info=hit_info
            )

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert agentic planner with autonomous query analysis capabilities. You must return only valid JSON with no extra text or explanations. Analyze each query independently and determine its type based on content and context."
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                model=self.model_name,
                temperature=0.1,  # Slightly higher for better reasoning
                max_tokens=512,
                top_p=1,
                stream=False,
                stop=None
            )

            response_text = chat_completion.choices[0].message.content.strip()
            print(f"🤖 Groq LLaMA 3.3 Orchestrator response: {response_text}")

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in planner output")

            json_str = json_match.group(0)
            parsed_result = json.loads(json_str)

            # Log the autonomous decision
            query_type = parsed_result.get('query_type', 'unknown')
            print(f"🎯 Orchestrator autonomous decision: {query_type.upper()} query")

            # Print the converted TAPAS query if available
            if "tapas_query" in parsed_result and parsed_result["tapas_query"]:
                print(f"📝 LLM converted to TAPAS query: {parsed_result['tapas_query']}")

            return parsed_result

        except Exception as e:
            print(f"❌ Groq planner error: {e}")
            raise e
# === Scraper ===
class Scraper:
    def __init__(self, firebase_manager: Optional[FirebaseManager] = None):
        self.api_key = "e33bdd3a23dbd41d9bf753c8da8b0ec2d77c8c5ac5b694dc7292786a024f40f6"
        self.last_keywords = []  # Track last used keywords
        self.firebase_manager = firebase_manager

    def run(self, keywords: list) -> str:
        params = {
            "engine": "google_shopping",
            "q": " ".join(keywords),
            "gl": "in",
            "hl": "en",
            "api_key": self.api_key
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            shopping_results = results.get("shopping_results", [])
        except Exception as e:
            print(f"❌ Scraper error: {e}")
            return ""

        data = []
        for item in shopping_results:
            offers = item.get("offers", [{}])
            first_offer = offers[0] if offers else {}

            row = {
                "title": item.get("title"),
                "price": item.get("price"),
                "extracted_price": item.get("extracted_price"),
                "source": item.get("source") or first_offer.get("source"),
                "product_link": item.get("product_link"),
                "rating": item.get("rating") or first_offer.get("rating", "N/A"),
                "reviews": item.get("reviews") or first_offer.get("reviews", 0),
                "image": item.get("thumbnail", "")
            }

            if row["title"] and row["extracted_price"]:
                data.append(row)

        if not data:
            print("❌ No valid data scraped")
            return ""

        # Save keywords for later use
        self.last_keywords = keywords

        # Save data to CSV with product name-based file name
        file_prefix = "_".join(keywords).lower().replace(" ", "_")
        csv_file = f"{file_prefix}_data.csv"

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        print(f"✅ Scraped {len(data)} items, saved to {csv_file}")

        # Upload scraped CSV to Firebase
        if self.firebase_manager:
            product_name = "_".join(keywords)
            self.firebase_manager.upload_csv_to_storage(csv_file, product_name, "scraped")

        return csv_file

    def get_last_keywords(self) -> list:
        """Get the last used keywords."""
        return self.last_keywords
# === Groq-based Pandas Code Generator (Reasoner Replacement) ===
class GroqPandasCodeGenerator:
    def __init__(self, csv_path, api_key, firebase_manager: Optional[FirebaseManager] = None):
        self.csv_path = csv_path
        self.firebase_manager = firebase_manager

        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.client = Groq(api_key=api_key)

        print(f"📊 Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Sample data preview:")
        print(self.df.head(2).to_string())

    def clean_code(self, raw_code):
        """Clean and extract executable code from LLM response"""
        # Remove markdown code blocks
        code = re.sub(r'``````', '', raw_code)

        # Remove leading/trailing whitespace
        code = code.strip()

        # Split by lines and clean each line
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                lines.append(line)

        # Join back into executable code
        cleaned_code = '\n'.join(lines)

        print(f"🧹 Cleaned code:\n{cleaned_code}")
        return cleaned_code

    def generate_pandas_code(self, query):
        """Generate pandas code from natural language query"""

        # FULL SYSTEM PROMPT FROM ORIGINAL CODE
        system_prompt = """You are an expert pandas data analyst. Generate executable pandas code to answer queries about product data.

Dataset Information:
- DataFrame variable: df
- Shape: {self.df.shape}
- Columns: {list(self.df.columns)}
- Data Types: {dict(self.df.dtypes)}

Sample Data:
{self.df.head(3).to_string()}

IMPORTANT RULES:

1. Use ONLY pandas operations with df variable
2. Store final result in variable 'result'
3. Return ONLY executable pandas code, no markdown, no explanations
4. Handle missing values with .fillna() or .dropna() when needed
5. Use .str.contains() for text searches (case-insensitive with case=False)
6. For extracted_price/numeric comparisons, ensure numeric conversion if needed
7. ALWAYS return ALL columns in result - do NOT limit to specific columns
8. MAINLY if the query is like Which product has ergonomic chairs in the title or which product has tissot watches etc, in the code do not put tissot watches or ergonomic chairs directly just use tissot or ergonomic, watches and chairs etc are nouns do not use it in the code

TRAINING EXAMPLES - QUERY → PANDAS CODE:

### BASIC FILTERING QUERIES:
Query: "Which products have price less than 50000?"
Code: result = df[df['extracted_price'] < 50000]

Query: "Which products have rating more than 4 stars?"
Code: result = df[df['rating'] > 4.0]

Query: "Which products are from the source myntra?"
Code: result = df[df['source'].str.contains('myntra', case=False, na=False)]

Query: "Which products contain MacBook Pro in the title?"
Code: result = df[df['title'].str.contains('MacBook Pro', case=False, na=False)]

Query: "Which products have more than 100 reviews?"
Code: result = df[df['reviews'] > 100]

### SUPERLATIVE QUERIES:
Query: "Which products are cheap?" / "Which products have the lowest price?"
Code: result = df.nsmallest(10, 'extracted_price')

Query: "Which products have the highest rating?"
Code: result = df.nlargest(10, 'rating')

Query: "Which products have the highest price?" / "Which products are expensive?"
Code: result = df.nlargest(10, 'extracted_price')

### COMPOUND CONDITIONS:
Query: "Which products have price less than 30000 and rating more than 4 stars?"
Code: result = df[(df['extracted_price'] < 30000) & (df['rating'] > 4.0)]

Query: "Which products have more than 200 reviews and rating more than 4 stars?"
Code: result = df[(df['reviews'] > 200) & (df['rating'] > 4.0)]

Query: "Which products have price less than 40000?"
Code: result = df[df['extracted_price'] < 40000]

### RANGE QUERIES:
Query: "Which products have price between 20000 and 80000?"
Code: result = df[(df['extracted_price'] >= 20000) & (df['extracted_price'] <= 80000)]

Query: "Which products have rating between 3.5 and 4.5?"
Code: result = df[(df['rating'] >= 3.5) & (df['rating'] <= 4.5)]

Query: "Which products have price under 25000 and are highly rated?"
Code: result = df[(df['extracted_price'] < 25000) & (df['rating'] > 4.0)]

### COMPLEX BUSINESS QUERIES:
Query: "Which products are expensive but have low ratings?"
Code: result = df[(df['extracted_price'] > df['extracted_price'].quantile(0.8)) & (df['rating'] < 3.5)]

Query: "Which products are cheap but highly rated?"
Code: result = df[(df['extracted_price'] < df['extracted_price'].quantile(0.3)) & (df['rating'] > 4.0)]

Query: "Which products have the best value for money?"
Code: result = df[(df['extracted_price'] < df['extracted_price'].median()) & (df['rating'] > 4.0)].nlargest(10, 'rating')

Query: "Which products are from amazon and cost more than 50000?"
Code: result = df[(df['source'].str.contains('amazon', case=False, na=False)) & (df['extracted_price'] > 50000)]

Query: "Which products have more than 500 reviews and rating above 4.2?"
Code: result = df[(df['reviews'] > 500) & (df['rating'] > 4.2)]

### TEXT SEARCH WITH CONDITIONS:
Query: "Which laptops have rating more than 4 stars?"
Code: result = df[(df['title'].str.contains('laptop', case=False, na=False)) & (df['rating'] > 4.0)]

Query: "Which phones are under 50000 and from flipkart?"
Code: result = df[(df['title'].str.contains('phone', case=False, na=False)) & (df['extracted_price'] < 50000) & (df['source'].str.contains('flipkart', case=False, na=False))]

Query: "Which watches have more than 50 reviews?"
Code: result = df[(df['title'].str.contains('watch', case=False, na=False)) & (df['reviews'] > 50)]

### COUNTING QUERIES:
Query: "How many products cost more than 100000?"
Code: result = len(df[df['extracted_price'] > 100000])

Query: "How many products have rating more than 4 stars?"
Code: result = len(df[df['rating'] > 4.0])

Query: "How many products are from myntra?"
Code: result = len(df[df['source'].str.contains('myntra', case=False, na=False)])

### STATISTICAL QUERIES:
Query: "What is the average price of all products?"
Code: result = df['extracted_price'].mean()

Query: "What is the highest rated product under 30000?"
Code: result = df[df['extracted_price'] < 30000].nlargest(1, 'rating')

Query: "What is the most expensive product with good rating?"
Code: result = df[df['rating'] > 4.0].nlargest(1, 'extracted_price')

### FILTERING WITH MULTIPLE SOURCES:
Query: "Which products are from flipkart or amazon?"
Code: result = df[(df['source'].str.contains('flipkart', case=False, na=False)) | (df['source'].str.contains('amazon', case=False, na=False))]

Query: "Which products are NOT from myntra?"
Code: result = df[~df['source'].str.contains('myntra', case=False, na=False)]

KEYWORD MAPPINGS:
- "cheap/cheapest" = nsmallest() or extracted_price < quantile(0.3)
- "expensive/highest price" = nlargest() or extracted_price > quantile(0.8)
- "highly rated/best rating" = rating > 4.0 or nlargest('rating')
- "reliable" = reviews > 200 & rating > 4.0
- "good rating" = rating > 4.0
- "popular" = reviews > median reviews
- "under X" = extracted_price < X
- "above/over X" = extracted_price > X
- "between X and Y" = (extracted_price >= X) & (extracted_price <= Y)
- "value for money" = low extracted_price + high rating
"""

        user_prompt = f"Generate pandas code for: {query}"

        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=300
            )

            raw_code = completion.choices[0].message.content
            cleaned_code = self.clean_code(raw_code)
            return cleaned_code

        except Exception as e:
            return f"# Error generating code: {e}"

    def validate_code_safety(self, code):
        """Validate code for safety"""
        dangerous_keywords = ['import', 'exec', 'eval', 'open', 'file', 'input', 'subprocess', 'system', 'os.', 'sys.', '__']

        code_lower = code.lower()
        for keyword in dangerous_keywords:
            if keyword in code_lower:
                return False, f"Dangerous keyword detected: {keyword}"

        return True, "Code is safe"

    def execute_pandas_code(self, code):
        """Execute pandas code and return result"""
        try:
            # Create execution context
            exec_context = {
                'df': self.df,
                'pd': pd,
                'result': None
            }

            # Execute code
            exec(code, exec_context)

            result = exec_context['result']

            if result is None:
                return "❌ Code did not produce a result. Make sure to assign output to 'result' variable."

            return result

        except Exception as e:
            return f"❌ Execution error: {str(e)}"

    def save_result_to_csv(self, result, query, filename=None, product_name=None):
        """Save result to CSV file with timestamp and query-based naming, and upload to Firebase."""
        if filename is None:
            # Generate filename based on timestamp and query
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r'[^\w\s-]', '', query)[:30]  # Remove special chars, limit length
            safe_query = re.sub(r'[-\s]+', '_', safe_query)  # Replace spaces/hyphens with underscore
            filename = f"query_result_{safe_query}_{timestamp}.csv"

        try:
            if isinstance(result, pd.DataFrame):
                result.to_csv(filename, index=False)
                print(f"💾 Result saved to: {filename}")
                print(f"📊 Saved {len(result)} rows to CSV")

                # Upload reasoned CSV to Firebase
                if self.firebase_manager and product_name:
                    self.firebase_manager.upload_csv_to_storage(filename, product_name, "reasoned")

                return filename
            else:
                # For non-DataFrame results, create a simple CSV
                simple_df = pd.DataFrame({'result': [result], 'query': [query]})
                simple_df.to_csv(filename, index=False)
                print(f"💾 Result saved to: {filename}")

                # Upload to Firebase
                if self.firebase_manager and product_name:
                    self.firebase_manager.upload_csv_to_storage(filename, product_name, "reasoned")

                return filename
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")
            return None

    def query(self, natural_query, save_to_csv=False, product_name=None):
        """Process a natural language query"""
        print(f"\n{'='*80}")
        print(f"🔍 Query: {natural_query}")
        print(f"{'='*80}")

        # Generate code
        code = self.generate_pandas_code(natural_query)

        if code.startswith("# Error"):
            print(code)
            return None

        # Safety check
        is_safe, safety_msg = self.validate_code_safety(code)
        if not is_safe:
            return f"⚠️ Safety check failed: {safety_msg}"

        # Execute code
        result = self.execute_pandas_code(code)

        # Display result
        print(f"\n📊 Result:")
        if isinstance(result, pd.DataFrame):
            print(f"Found {len(result)} rows:")
            if len(result) > 0:
                display_df = result.head(10) if len(result) > 10 else result
                print(display_df.to_string(index=False))
                if len(result) > 10:
                    print(f"... and {len(result) - 10} more rows")
            else:
                print("No matching records found")
        elif isinstance(result, (int, float)):
            print(f"Answer: {result}")
        elif isinstance(result, str) and result.startswith("❌"):
            print(result)
        else:
            print(f"Result: {result}")

        # Save to CSV if requested
        if save_to_csv and not (isinstance(result, str) and result.startswith("❌")):
            saved_file = self.save_result_to_csv(result, natural_query, product_name=product_name)
            return result, saved_file

        return result
# === New Reasoner using GroqPandasCodeGenerator ===
class Reasoner:
    """
    Enhanced reasoner that uses GroqPandasCodeGenerator with CSV chaining support.
    """
    def __init__(self, groq_api_key: str, firebase_manager: Optional[FirebaseManager] = None):
        """Initialize the Groq-based reasoner."""
        print("🤖 Initializing Groq-based Pandas Reasoner...")
        self.groq_api_key = groq_api_key
        self.firebase_manager = firebase_manager
        self.pandas_generator = None
        self.previous_results = None
        self.previous_results_file = None
        self.current_product_name = None
        self.last_generated_csv = None  # NEW: Track the last generated CSV file
        print("✅ Groq-based Reasoner initialized successfully")

    def process(self, pandas_query: str, csv_file: str, use_previous_results: bool = False, product_name: str = None) -> list:
        try:
            print(f"🔍 Processing query: {pandas_query}")
            print(f"📂 Using data file: {csv_file}")

            # Store product name
            if product_name:
                self.current_product_name = product_name

            # Initialize pandas generator for this file
            if self.pandas_generator is None or self.pandas_generator.csv_path != csv_file:
                self.pandas_generator = GroqPandasCodeGenerator(csv_file, self.groq_api_key, self.firebase_manager)

            # Process the query with save to CSV enabled
            result = self.pandas_generator.query(pandas_query, save_to_csv=True, product_name=self.current_product_name)

            # Handle tuple result (result, filename)
            if isinstance(result, tuple):
                actual_result, saved_file = result
                self.last_generated_csv = saved_file  # NEW: Store the filtered CSV filename
                print(f"💾 Filtered CSV saved: {saved_file}")

                if isinstance(actual_result, pd.DataFrame):
                    results_list = actual_result.to_dict('records')
                elif isinstance(actual_result, (int, float)):
                    results_list = [{"result": actual_result, "type": "scalar"}]
                else:
                    results_list = []
            elif isinstance(result, pd.DataFrame):
                results_list = result.to_dict('records')
                # NEW: Even if not tuple, save the result and track it
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_query = re.sub(r'[^\w\s-]', '', pandas_query)[:30]
                safe_query = re.sub(r'[-\s]+', '_', safe_query)
                saved_file = f"query_result_{safe_query}_{timestamp}.csv"
                result.to_csv(saved_file, index=False)
                self.last_generated_csv = saved_file
                print(f"💾 Filtered CSV saved: {saved_file}")
            elif isinstance(result, (int, float)):
                results_list = [{"result": result, "type": "scalar"}]
            else:
                results_list = []

            # Store results for potential chaining
            self.previous_results = results_list

            if results_list:
                print(f"✅ Found {len(results_list)} matching products")
                # Show first 5 results
                for i, product in enumerate(results_list[:5], 1):
                    if "result" in product and "type" in product:
                        print(f"{i}. Result: {product['result']}")
                    else:
                        title = product.get('title', 'N/A')
                        price = product.get('price', product.get('extracted_price', 'N/A'))
                        rating = product.get('rating', 'N/A')
                        source = product.get('source', 'N/A')
                        print(f"{i}. {title} | Price: {price} | Rating: {rating} | Source: {source}")

                if len(results_list) > 5:
                    print(f"... and {len(results_list) - 5} more results")
            else:
                print("❌ No matches found")

            return results_list

        except Exception as e:
            print(f"❌ Reasoner processing error: {e}")
            return []

    def get_last_csv_file(self) -> Optional[str]:
        """NEW: Get the last generated CSV file path."""
        return self.last_generated_csv


# === RAG-based Compliance Checker ===
class ComplianceChecker:
    def __init__(self, groq_api_key: str, pdf_path: str = None):
        """Initialize compliance checker with optional PDF path."""
        print("🔒 Initializing Compliance Checker...")
        self.client = Groq(api_key=groq_api_key)
        self.model_name = "qwen/qwen3-32b"

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load and process PDF
        if pdf_path and os.path.exists(pdf_path):
            print(f"📄 Loading compliance policy from: {pdf_path}")
            self.policy_text = extract_text_from_pdf(pdf_path)
            if self.policy_text:
                self.policy_chunks = self._chunk_text(self.policy_text)
                self.chunk_embeddings = self.embedding_model.encode(self.policy_chunks, convert_to_tensor=True)
                print(f"✅ Policy loaded: {len(self.policy_chunks)} chunks indexed")
            else:
                print("⚠️ Could not extract text from PDF. Using default policy.")
                self.policy_text = self._get_default_policy()
                self.policy_chunks = self._chunk_text(self.policy_text)
                self.chunk_embeddings = self.embedding_model.encode(self.policy_chunks, convert_to_tensor=True)
        else:
            print("📋 No PDF provided. Using default compliance policy.")
            self.policy_text = self._get_default_policy()
            self.policy_chunks = self._chunk_text(self.policy_text)
            self.chunk_embeddings = self.embedding_model.encode(self.policy_chunks, convert_to_tensor=True)

        print("✅ Compliance Checker initialized")

    def _get_default_policy(self) -> str:
        """Return a default compliance policy if no PDF is provided."""
        return """ """

    def _chunk_text(self, text: str, chunk_size: int = 500) -> list:
        """Split text into chunks for embedding."""
        sentences = text.split('\n')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve most relevant policy chunks for a query."""
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.policy_chunks)))

            relevant_chunks = []
            for idx in top_results[1]:
                relevant_chunks.append(self.policy_chunks[idx])

            return "\n\n---\n\n".join(relevant_chunks)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return self.policy_text[:1000]  # Return first 1000 chars as fallback

    def check_compliance(self, user_role: str, product_name: str, product_price: float, vendor: str = "Unknown") -> str:
        try:
            # Retrieve relevant policy context
            query = f"purchase authorization limits for {user_role} buying {product_name} at ₹{product_price} from {vendor}"
            context = self.retrieve(query, top_k=5)

            # FULL COMPLIANCE CHECK PROMPT FROM ORIGINAL CODE
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert compliance officer analyzing procurement requests against company policy.

You must provide ONLY a structured compliance analysis.

**DO NOT include any thinking process, reasoning steps, or explanations of your analysis process.**

Provide a direct, professional compliance analysis with:

1. **Compliance Status:** Compliant/Non-Compliant/Needs Approval
2. **Allowance Analysis:** Check against role-based limits
3. **Approval Requirements:** What approvals are needed
4. **Vendor Compliance:** Check if vendor is approved
5. **Next Steps:** Specific actions required

Be specific about policy sections and provide clear recommendations.

**Do not show your thinking process.**"""
                },
                {
                    "role": "user",
                    "content": f"""**POLICY CONTEXT:**
{context}

**PURCHASE REQUEST:**
- Employee Role: {user_role}
- Item: {product_name}
- Price: ₹{product_price} INR
- Vendor: {vendor}

Provide a direct compliance analysis following the structure above.

**Do not include thinking process or reasoning steps.**"""
                }
            ]

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.1,  # Very low temperature for consistent, direct responses
                max_tokens=1500,
                top_p=1,
                stream=False,
                stop=None
            )

            raw_response = chat_completion.choices[0].message.content.strip()

            # Clean the response to remove any thinking process
            cleaned_response = self._clean_response(raw_response)

            return cleaned_response

        except Exception as e:
            # Provide basic compliance check as fallback
            try:
                return self._basic_compliance_check(user_role, product_name, product_price, vendor)
            except Exception as fallback_error:
                return f"I apologize, but I encountered an error while checking compliance: {str(e)}"

    def _clean_response(self, response: str) -> str:
        """Remove thinking process from response."""
        # Remove patterns like <think>...</think> or similar
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'\*\*think\*\*.*?\*\*/think\*\*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove any remaining thinking patterns
        cleaned = re.sub(r'\.\.\.think\.\.\..*?\.\.\.\/think\.\.\.', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned.strip()

    def _basic_compliance_check(self, user_role: str, product_name: str, product_price: float, vendor: str) -> str:
        """Provide basic compliance check as fallback."""
        limits = {
            "junior staff": 25000,
            "senior staff": 50000,
            "manager": 100000,
            "director": 500000
        }

        user_limit = limits.get(user_role.lower(), 25000)

        compliance_status = "Compliant" if product_price <= user_limit else "Needs Approval"

        return f"""
**COMPLIANCE ANALYSIS**

**Compliance Status:** {compliance_status}

**Allowance Analysis:**
Your role ({user_role}) has a purchase limit of ₹{user_limit}.
Requested amount: ₹{product_price}

**Approval Requirements:**
{"No additional approval required." if product_price <= user_limit else f"Requires supervisor approval as price exceeds your limit by ₹{product_price - user_limit}."}

**Vendor Compliance:**
Vendor: {vendor}
{"Approved vendor" if vendor.lower() in ["amazon", "flipkart", "myntra", "croma", "reliance digital"] else "Vendor status needs verification"}

**Next Steps:**
{f"Proceed with purchase." if product_price <= user_limit else f"Submit approval request to your supervisor."}
"""

    def get_policy_summary(self) -> str:
        """Get a summary of the loaded policy."""
        summary_query = "policy summary allowances roles approval workflow"
        try:
            relevant_chunks = self.retrieve(summary_query, top_k=3)
            return relevant_chunks[:1000] + "..." if len(relevant_chunks) > 1000 else relevant_chunks
        except Exception as e:
            return f"Policy summary unavailable due to error: {e}"

# === Justification Agent ===
class Justifier:
    """
    Justification agent that uses Qwen model to explain recommendations
    based on user preferences.
    """
    def __init__(self, groq_api_key: str):
        """Initialize with Groq API key."""
        self.client = Groq(api_key=groq_api_key)
        self.model_name = "qwen/qwen3-32b"

    def run(self, query: str, preferences: list, csv_file: str = None) -> str:
        """Generate justification based on user preferences and last results."""
        try:
            # Load the last saved CSV file if available
            data_context = ""
            if csv_file and os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Convert dataframe to readable format
                        data_context = f"\n\n**SEARCH RESULTS ({len(df)} products):**\n"
                        for idx, row in df.head(10).iterrows():  # Show first 10 products
                            data_context += f"{idx+1}. {row.get('title', 'N/A')} - Price: {row.get('price', 'N/A')}, Rating: {row.get('rating', 'N/A')}, Reviews: {row.get('reviews', 'N/A')}, Source: {row.get('source', 'N/A')}\n"
                        if len(df) > 10:
                            data_context += f"... and {len(df) - 10} more products\n"
                except Exception as e:
                    print(f"❌ Error loading CSV for justification: {e}")
                    data_context = "\n\n**CONTEXT:** Recent search results available."
            else:
                data_context = "\n\n**CONTEXT:** Recent search results available."

            # Create preference-based justification prompt
            preference_focus = ""
            if "rating_conscious" in preferences:
                preference_focus += "- You prioritize products with high ratings and good reviews\n"
            if "price_conscious" in preferences:
                preference_focus += "- You prioritize products with competitive pricing and value for money\n"
            if "review_conscious" in preferences:
                preference_focus += "- You prioritize products with substantial user reviews and feedback\n"

            # FULL JUSTIFICATION PROMPT FROM ORIGINAL CODE
            justification_prompt = f"""You are an expert product recommendation justifier.

Your task is to explain and justify product recommendations based on user preferences.

**Do not generate your thinking process, just generate the justifications.**

**USER PREFERENCES:**
{preference_focus}

**USER QUERY:**
{query}

**PRODUCT DATA CONTEXT:**
{data_context}

**Your task is to:**
1. Analyze the user's justification query
2. Based on their preferences ({', '.join(preferences)}), explain why certain products were recommended
3. Provide clear, logical reasoning that aligns with their stated preferences
4. Use specific data from the product results to support your justification
5. Be concise but thorough in your explanation

**Guidelines:**
- If user is RATING CONSCIOUS: Focus on ratings, quality indicators, star ratings, user satisfaction
- If user is PRICE CONSCIOUS: Focus on pricing, value for money, cost-effectiveness, budget optimization
- If user is REVIEW CONSCIOUS: Focus on review count, user feedback, community validation, popularity metrics
- Use specific numbers and data points from the product results
- Explain the trade-offs and reasoning behind the recommendations
- Be honest about limitations or compromises made

Provide a clear, helpful justification that addresses the user's query and aligns with their preferences.

**Do not generate your thinking process, just generate the justifications.**
"""

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert product recommendation justifier who provides clear, logical explanations based on user preferences and data."
                    },
                    {
                        "role": "user",
                        "content": justification_prompt
                    }
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None
            )

            justification = chat_completion.choices[0].message.content.strip()
            return justification

        except Exception as e:
            print(f"❌ Justification error: {e}")
            return f"I apologize, but I encountered an error while generating the justification: {str(e)}"
class Pipeline:
    def __init__(self, groq_api_key: str, compliance_pdf_path: str = None, firebase_credentials_path: str = None, firebase_storage_bucket: str = None):
        self.planner = AgenticPlanner(groq_api_key)

        # Initialize Firebase Manager if credentials provided
        self.firebase_manager = None
        if firebase_credentials_path and firebase_storage_bucket:
            try:
                self.firebase_manager = FirebaseManager(firebase_credentials_path, firebase_storage_bucket)
            except Exception as e:
                print(f"⚠️ Firebase initialization failed: {e}")
                print("Continuing without Firebase integration...")

        self.scraper = Scraper(self.firebase_manager)
        self.reasoner = Reasoner(groq_api_key, self.firebase_manager)
        self.justifier = Justifier(groq_api_key)
        self.compliance_checker = ComplianceChecker(groq_api_key, compliance_pdf_path)

        self.previous_query = None
        self.previous_csv_file = ""
        self.original_csv_file = ""
        self.last_results = []
        self.user_preferences = []
        self.current_product_name = None
        self.csv_chain = []

    def set_preferences(self, preferences: list):
        """Set user preferences for the pipeline."""
        self.user_preferences = preferences
        print(f"🎯 User preferences set: {', '.join(preferences)}")

    def get_context_info(self) -> dict:
        """Get current context information for the orchestrator."""
        return {
            "has_previous_results": bool(self.previous_csv_file and os.path.exists(self.previous_csv_file)),
            "previous_product": self.current_product_name,
            "csv_chain_length": len(self.csv_chain)
        }

    def run(self, query: str):
        """Main pipeline execution with autonomous orchestrator decision making."""
        try:
            print(f"\n🚀 Processing query: '{query}'")

            # Get context information for the orchestrator
            context_info = self.get_context_info()
            print(f"📋 Context: {context_info}")

            # Let the orchestrator autonomously decide query type and generate plan
            plan = self.planner.run(query, self.user_preferences, context_info)

            # Store original query in plan for logging
            plan["original_query"] = query

            print(f"📋 Orchestrator Plan: {plan}")

            # Execute plan based on autonomous query type detection
            query_type = plan.get("query_type", "new")

            if query_type == "justification":
                return self._handle_justification(plan)
            elif query_type == "follow_up":
                return self._handle_follow_up(plan)
            else:  # new or any other type
                return self._handle_new_query(plan)

        except Exception as e:
            print(f"❌ Pipeline execution error: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"

    def _handle_justification(self, plan: dict):
        """Handle justification queries."""
        try:
            step = plan["steps"][0]
            result = self.justifier.run(
                step["args"]["query"],
                step["args"].get("preferences", self.user_preferences),
                self.previous_csv_file
            )
            return result
        except Exception as e:
            print(f"❌ Justification handling error: {e}")
            return "I apologize, but I couldn't generate a proper justification."

    def _handle_follow_up(self, plan: dict):
        """Handle autonomous follow-up queries that filter previous results."""
        try:
            if not self.previous_csv_file or not os.path.exists(self.previous_csv_file):
                # If no previous results but orchestrator detected follow-up,
                # treat as new query with a warning
                print("⚠️ Orchestrator detected follow-up but no previous results found. Treating as new query.")
                return self._handle_new_query(plan)

            print(f"🔗 Orchestrator using CSV from previous query: {self.previous_csv_file}")

            step = plan["steps"][0]
            results = self.reasoner.process(
                step["args"]["tapas_query"],
                self.previous_csv_file,
                use_previous_results=True,
                product_name=self.current_product_name
            )

            if not results:
                return "❌ No products match your filter criteria."

            # Get the newly generated filtered CSV file
            new_filtered_csv = self.reasoner.get_last_csv_file()

            if new_filtered_csv and os.path.exists(new_filtered_csv):
                # Update the previous_csv_file to the NEW filtered CSV for the next follow-up
                self.previous_csv_file = new_filtered_csv
                self.csv_chain.append(new_filtered_csv)
                print(f"✅ Updated active CSV to: {new_filtered_csv}")
                print(f"📊 CSV Chain: {' -> '.join(self.csv_chain)}")

                # Upload the filtered CSV to Firebase
                if self.firebase_manager and self.current_product_name:
                    self.firebase_manager.upload_csv_to_storage(
                        new_filtered_csv,
                        self.current_product_name,
                        "filtered"
                    )

            # Update last results
            self.last_results = results
            return results

        except Exception as e:
            print(f"❌ Follow-up handling error: {e}")
            return f"I apologize, but I couldn't filter the previous results: {str(e)}"

    def _handle_new_query(self, plan: dict):
        """Handle new queries that require scraping and reasoning."""
        try:
            # Check if query exists in Firebase logs
            if self.firebase_manager:
                original_query = plan.get("original_query", "")
                cached_log = self.firebase_manager.check_query_exists(original_query)

                if cached_log:
                    print("🔄 Found cached query result in Firebase")
                    csv_file_name = cached_log.get("csv_file_name")
                    firebase_path = cached_log.get("firebase_path")
                    product_name = cached_log.get("product_name")

                    # Download CSV from Firebase
                    local_file = f"cached_{csv_file_name}"
                    if self.firebase_manager.download_csv_from_storage(firebase_path, local_file):
                        # Load and return cached results
                        df = pd.read_csv(local_file)
                        results = df.to_dict('records')
                        self.last_results = results
                        self.previous_csv_file = local_file
                        self.original_csv_file = local_file
                        self.current_product_name = product_name
                        self.csv_chain = [local_file]
                        return results

            csv_file = ""
            results = []
            reasoned_csv_file = ""

            # Execute each step in the plan
            for step in plan["steps"]:
                agent = step["agent"]
                args = step["args"]

                if agent == "scraper":
                    keywords = args["keywords"]
                    self.current_product_name = "_".join(keywords)
                    csv_file = self.scraper.run(keywords)
                    if not csv_file:
                        return "❌ Failed to scrape product data. Please try again."

                    # Set BOTH original and current CSV to the scraped file
                    self.original_csv_file = csv_file
                    self.previous_csv_file = csv_file
                    self.csv_chain = [csv_file]
                    print(f"📊 CSV Chain initialized: {csv_file}")

                elif agent == "reasoner":
                    tapas_query = args["tapas_query"]
                    results = self.reasoner.process(
                        tapas_query,
                        csv_file,
                        use_previous_results=False,
                        product_name=self.current_product_name
                    )

                    # Get the reasoned CSV filename
                    reasoned_csv_file = self.reasoner.get_last_csv_file()

                    # Update the previous_csv_file to the reasoned CSV
                    if reasoned_csv_file and os.path.exists(reasoned_csv_file):
                        self.previous_csv_file = reasoned_csv_file
                        self.csv_chain.append(reasoned_csv_file)
                        print(f"✅ Active CSV updated to: {reasoned_csv_file}")
                        print(f"📊 CSV Chain: {' -> '.join(self.csv_chain)}")

            if not results:
                return "❌ No products found matching your criteria."

            # Log to Firebase if manager available
            if self.firebase_manager and reasoned_csv_file and self.current_product_name:
                original_query = plan.get("original_query", "")
                self.firebase_manager.log_query_to_firestore(
                    original_query,
                    self.current_product_name,
                    reasoned_csv_file
                )

            # Update last results
            self.last_results = results
            return results

        except Exception as e:
            print(f"❌ New query handling error: {e}")
            return f"I apologize, but I couldn't process your new query: {str(e)}"

    def _format_results(self, results: list, justify: bool = False) -> str:
        """Format results for display."""
        if not results:
            return "No products found."

        # Check if results contain scalar values
        if len(results) == 1 and "result" in results[0] and "type" in results[0]:
            return f"📊 Result: {results[0]['result']}"

        output = f"✅ Found {len(results)} products:\n\n"

        for i, product in enumerate(results[:10], 1):  # Show top 10
            title = product.get('title', 'N/A')
            price = product.get('price', product.get('extracted_price', 'N/A'))
            rating = product.get('rating', 'N/A')
            reviews = product.get('reviews', 'N/A')
            source = product.get('source', 'N/A')

            output += f"{i}. {title}\n"
            output += f"   💰 Price: {price} | ⭐ Rating: {rating} | 📝 Reviews: {reviews} | 🏪 Source: {source}\n\n"

        if len(results) > 10:
            output += f"... and {len(results) - 10} more products\n"

        return output

    def clear_session(self):
        """Clear the current session including CSV chain."""
        self.previous_query = None
        self.previous_csv_file = ""
        self.original_csv_file = ""
        self.last_results = []
        self.current_product_name = None
        self.csv_chain = []
        print("🧹 Session cleared. Ready for a new query.")

    # Rest of the methods remain the same...
    def select_and_check_compliance(self):
        """Allow user to select a product and check compliance."""
        if not self.last_results:
            print("❌ No products to select from.")
            return

        try:
            product_index = int(input(f"\nEnter product number (1-{len(self.last_results)}): ")) - 1

            if product_index < 0 or product_index >= len(self.last_results):
                print("❌ Invalid product number.")
                return

            selected_product = self.last_results[product_index]

            print("\n📋 Selected Product:")
            print(f"Title: {selected_product.get('title', 'N/A')}")
            print(f"Price: {selected_product.get('price', selected_product.get('extracted_price', 'N/A'))}")

            # Get user role
            print("\n👤 User Role Options:")
            print("1. Junior Staff")
            print("2. Senior Staff")
            print("3. Manager")
            print("4. Director")
            role_choice = input("Select your role (1-4): ").strip()

            role_map = {
                "1": "Junior Staff",
                "2": "Senior Staff",
                "3": "Manager",
                "4": "Director"
            }

            user_role = role_map.get(role_choice, "Junior Staff")

            # Extract price
            price_str = str(selected_product.get('extracted_price', selected_product.get('price', '0')))
            price = float(re.sub(r'[^\d.]', '', price_str)) if price_str != 'N/A' else 0

            # Check compliance
            print("\n🔍 Checking compliance...")
            compliance_result = self.compliance_checker.check_compliance(
                user_role=user_role,
                product_name=selected_product.get('title', 'Unknown Product'),
                product_price=price,
                vendor=selected_product.get('source', 'Unknown')
            )

            print("\n" + "="*80)
            print("COMPLIANCE CHECK RESULT")
            print("="*80)
            print(compliance_result)
            print("="*80)

        except ValueError:
            print("❌ Please enter a valid number.")
        except Exception as e:
            print(f"❌ Error during compliance check: {e}")

'''

def main():
    """Main function to run the enhanced agentic pipeline."""
    print("🛍️ Enhanced Agentic E-commerce Pipeline with Firebase Integration")
    print("="*80)

    # Get Groq API key
    groq_api_key = input("Please enter your Groq API key: ").strip()
    if not groq_api_key:
        print("❌ Groq API key is required!")
        return

    # Get Firebase credentials
    print("\n🔥 Firebase Configuration (Optional)")
    firebase_creds = input("Enter path to Firebase credentials JSON file (press Enter to skip): ").strip()
    firebase_bucket = None

    if firebase_creds and os.path.exists(firebase_creds):
        firebase_bucket = input("Enter Firebase Storage Bucket (e.g., your-app.firebasestorage.app): ").strip()
        if not firebase_bucket:
            print("⚠️ Storage bucket required for Firebase. Skipping Firebase integration.")
            firebase_creds = None
    elif firebase_creds:
        print("⚠️ Firebase credentials file not found. Skipping Firebase integration.")
        firebase_creds = None

    print("\n📄 PDF Compliance Policy (Optional)")
    compliance_pdf_path = input("Enter path to compliance policy PDF (press Enter to use default policy): ").strip()
    if compliance_pdf_path and not os.path.exists(compliance_pdf_path):
        print("⚠️ PDF file not found. Using default policy instead.")
        compliance_pdf_path = None

    # Initialize pipeline
    try:
        pipeline = Pipeline(groq_api_key, compliance_pdf_path, firebase_creds, firebase_bucket)
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return

    # Set user preferences
    print("\n🎯 Setting User Preferences")
    print("Available preferences: rating_conscious, price_conscious, review_conscious")
    preferences_input = input("Enter your preferences (comma-separated): ").strip()

    if preferences_input:
        preferences = [p.strip() for p in preferences_input.split(',')]
        pipeline.set_preferences(preferences)
    else:
        pipeline.set_preferences(['rating_conscious'])

    print("\n💬 You can now start asking questions!")
    print("Examples:")
    print("- 'find me MacBook under 100000'")
    print("- 'get me phones with rating more than 4'")
    print("- 'show me cheaper options'")
    print("- Type 'clear' to clear session")
    print("- Type 'quit' to exit")

    # Main interaction loop
    while True:
        try:
            user_query = input("\n🎯 Your query: ").strip()

            if not user_query:
                continue

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Enhanced Agentic Pipeline!")
                break

            if user_query.lower() == 'clear':
                pipeline.clear_session()
                continue

            # Process the query
            response = pipeline.run(user_query)
            print(f"\n🤖 Response:\n{response}")

            if pipeline.last_results:
                check_compliance = input("\nDo you want to select a product and check compliance? (yes/no): ").strip().lower()
                if check_compliance == 'yes':
                    pipeline.select_and_check_compliance()

        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using the Enhanced Agentic Pipeline!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()

'''
