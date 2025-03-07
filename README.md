# 🧠 Deep Research AI Agent

## 📌 Overview
This project implements a **Deep Research AI Agentic System** using **Tavily for web crawling**, with **LangChain & LangGraph** for multi-agent processing. The system consists of:

1. **Research Agent** – Collects relevant articles.  
2. **Fact-Checking Agent** – Analyzes credibility and verifies claims.  
3. **Answer Drafting Agent** – Structures a response using LLMs.  

The project is built using **Python, Streamlit, LangChain, LangGraph, and Tavily**.  

---

## 📌 Features
✅ Uses **Tavily API** for online research  
✅ Leverages **LangGraph & LangChain** for multi-agent workflow  
✅ Extracts keywords using **spaCy**  
✅ Fact-checks claims using **Llama3**  
✅ Generates structured fact-checking reports  

---

## 📌 Installation & Setup
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
streamlit run app.py
## 📌 Architechture
🔹 Multi-Agent System

| 🏗 **Agent**              | 🔍 **Function**                                   |
|---------------------------|---------------------------------------------------|
| **Research Agent**        | Collects relevant articles from Tavily & Wikipedia|
| **Fact-Checking Agent**   | Assigns reliability scores & verifies claims      |

🔹 Workflow
1️⃣ User enters a claim →
2️⃣ Research Agent collects data →
3️⃣ Fact-Checking Agent verifies the claim →
4️⃣ Draft Agent summarizes findings →
✅ Result is displayed in Streamlit with sources & reliability scores.

📌 How to Use
Enter a claim in the Streamlit web app
Press "Enter"
View Fact-Check Results with reliability scores 
Download full report (Markdown)

## 📌 Repository Structure  

```bash
├── app.py              # Main Streamlit app
├── config.yaml         # Search & API configurations
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
├── .gitignore          # Excludes sensitive files
└── YourNotebook.ipynb  # Google Colab notebook (if included)

✅ 1️⃣ Step-by-Step Demo Flow

## **📌 Step 1: Open Google Colab & Launch Ollama**  
Since this project requires **Ollama (Llama 3)** for fact-checking, it must be installed and launched before running Streamlit.  

### **▶ 1️⃣ Open the Google Colab Notebook**  
[Click here to open the Colab notebook](YOUR_NOTEBOOK_LINK_HERE)  

### **▶ 2️⃣ Install Ollama & Launch It**  
Run these commands inside the Colab notebook:  
```bash
!pip install colab-xterm
%load_ext colabxterm
%xterm

Once the terminal opens, run these commands inside the xterm terminal:
 # curl https://ollama.ai/install.sh | sh
 # ollama serve & ollama pull llama3
 # ollama pull llama3
🔹 This installs Ollama and pulls the Llama 3 model for fact-checking.

## **📌 Step 2: Install Requirements & Run All Cells**  

Before starting the Streamlit app, follow these steps:  

### **▶ 1️⃣ Install Required Dependencies**  
Run this in **Google Colab**:  
```bash
!pip install langchain-ollama tavily-python langgraph langchain langchain-community
!pip install wikipedia-api transformers streamlit pyyaml pyngrok ngrok

### **▶ 2️⃣ Run All Notebook Cells**  

🔹 **Run all the cells** once packages are installed in Google Colab.  
🔹 This will **set up all agents, load models, and prepare the system**.  
🔹 **Enter your own Tavily API key** when prompted.


📌 Step 3: Start the Streamlit App in Google Colab
After running all cells, start Streamlit:
!streamlit run /content/app.py &>/dev/null&

Then, expose the app using ngrok and don't forget to add the ngrok authorization token:
from pyngrok import ngrok

ngrok.set_auth_token("your_ngrok_auth_token")

public_url = ngrok.connect(addr=8501, proto="http", auth=None)

print("Streamlit app is running at:", public_url)

Finally! Click on the ngrok link generated in output section and open it in your browser.

## 📌 Demo Screenshots  

### 🏠 Streamlit Interface  
![Streamlit Interface]([streamlit_interface.png](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/streamlit_interface.png))  

### 🔍 Fact-Check Result  
![Fact-Check Result](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/fact_check_result.png)  

### ✍ Draft Answer  
![Draft Answer]([draft_answer.png](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/draft_answer.png))  

### ⚠ Bias Detection & Source Reliability Scores Table
![Bias Detection](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/bias_detection.png)  

### 📊 Source Reliability Chart  
![Source Reliability]([source_reliability.png](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/source_reliability.png)  

### 📥 Downloading the Report  
![Download Report Button]([download_report_button.png](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/download_report_button.png))  

### 📄 Viewing the Downloaded Report  
![Downloaded Report]([downloaded_report.png](https://github.com/ssonali6/Fact-Checking-AI-Assistant/blob/main/downloaded_report.png))  






