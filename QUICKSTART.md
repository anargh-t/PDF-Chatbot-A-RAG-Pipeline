# Quick Start Guide

Get started with the PDF Chatbot Streamlit Web UI in 3 simple steps!

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection (for first-time model downloads)

## Installation

1. **Clone the repository** (or download the files)
   ```bash
   git clone https://github.com/anargh-t/PDF-Chatbot-A-RAG-Pipeline.git
   cd PDF-Chatbot-A-RAG-Pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

3. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

The app will automatically open in your browser at `http://localhost:8501`

## First Use

1. **Upload a PDF**: Click "Browse files" in the sidebar
2. **Process it**: Click the "🔄 Process PDF" button
3. **Ask questions**: Type your question and click "🚀 Ask"

That's it! 🎉

## Example Questions to Try

- "What are the main topics in this document?"
- "Can you summarize the key findings?"
- "What does section X say about Y?"
- "List the important points from page Z"

## Tips

- **First run takes longer**: Models need to download (~2GB)
- **Adjust retrieval**: Use the slider to control how many sources to use
- **Clear history**: Use the button in the sidebar to start fresh
- **Multiple PDFs**: Upload and process different PDFs without restarting

## Troubleshooting

**Models downloading slowly?**
- Be patient on first run - models cache for future use

**Out of memory?**
- Try processing smaller PDFs
- Close other applications
- Reduce the number of sources in settings

**App not starting?**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (needs 3.8+)

## Need Help?

- Check [DEMO.md](DEMO.md) for detailed usage guide
- Read [README.md](README.md) for full documentation
- Open an issue on GitHub for bugs or questions

Enjoy chatting with your PDFs! 📚✨
