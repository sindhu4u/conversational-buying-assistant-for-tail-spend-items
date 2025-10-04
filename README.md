# Conversational Buying Assistant

This project implements a **Slack-based Conversational Buying Assistant** that helps users submit procurement requests, performs automated research, compliance checks, and assists in purchase order (PO) creation.

The system leverages multiple agents:

1. **User Query** – A user types a product/procurement request into Slack.
2. **Overseer LLM** – Oversees the query, clarifies if needed, and issues tasks.
3. **Scraper Agent** – Scrapes data from various online sources and saves raw results (CSV).
4. **Reasoner Agent** – Ingests the CSV, filters, ranks, and selects the most relevant items.
5. **Display Results** – Shows selected items with short justifications (price, vendor, key specs, compliance notes).
6. **User Selection** – User picks items they want to buy.
7. **Compliance Check** – Verifies company rules like approved vendors, budget limits, and policy constraints.
8. **PO Creation** – Proceeds with purchase order generation once approved by the manager.

---

## Project Structure

```
.
├── main.py                  # Entry point for Slack-based execution
├── pipeline.py              # Pipeline for independent execution
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys, tokens, etc.)
├── conversational-buying-assistant-firebase-adminsdk.json # Firebase credentials
└── README.md
```

---

## Installation

1. **Clone the repository:**

```bash
git clone <repository_url>
cd <repository_folder>
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. **Install required packages:**

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory and define the following **placeholder variables**:

```env
SLACK_TOKEN=your-slack-bot-token
SIGNING_SECRET=your-slack-signing-secret
APP_TOKEN=your-slack-app-token
GROQ_API_KEY=your-groq-api-key
SERPAPI_KEY=your-serpapi-key
FIREBASE_CREDENTIALS_PATH=path-to-your-firebase-credentials.json
FIREBASE_BUCKET=your-firebase-storage-bucket
```

> **Instructions:**
>
> * **Slack Tokens**: Create a Slack App in your workspace and obtain the bot token, signing secret, and app token.
> * **Groq API Key**: Sign up at [Groq](https://groq.ai/) and generate an API key.
> * **SerpAPI Key**: Sign up at [SerpAPI](https://serpapi.com/) and get an API key.
> * **Firebase Credentials**: Create a Firebase project, generate a service account key JSON, and place it in your project directory. Update `FIREBASE_BUCKET` with your storage bucket name.

---

## Running the Project

### 1. Slack-based Execution

Run the Slack bot:

```bash
python main.py
```

* The bot will listen to Slack messages and interact with users according to the workflow described above.

### 2. Independent Execution

* For testing or running outside Slack (e.g., in Google Colab or a local Python shell), you can directly run the pipeline:

```bash
python pipeline.py
```

---

## Features

* Multi-agent conversational workflow
* Automated web scraping and data filtering
* Compliance and policy checks
* Slack integration for seamless user experience
* Firebase storage for persistence

---

## Notes

* Ensure all API keys and tokens are correctly set in `.env`.
* Python 3.10+ recommended for full compatibility.
* This bot is designed for enterprise environments; make sure proper security practices are followed when handling credentials.

---

## License

[MIT License](LICENSE)
