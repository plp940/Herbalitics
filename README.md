**Herbalitics**
==========================

**Project Title:** Herbalitics
 A cutting-edge platform for ayurvedic knowledge discovery

**📖 Description**
---------------

The Intelligent Learning Hub is a revolutionary AI-powered platform designed to revolutionize the way we learn and discover new knowledge. By leveraging the power of natural language processing, machine learning, and cloud computing, this platform provides a unique and personalized learning experience for users. With its intuitive interface and advanced algorithms, the Intelligent Learning Hub enables users to explore a vast repository of educational content, discover new topics and interests, and connect with like-minded individuals from around the world.

The platform is built using a combination of cutting-edge technologies, including Streamlit for its web-based interface, Sentence Transformers for natural language processing, and FAISS for efficient similarity search. By integrating these technologies, the Intelligent Learning Hub provides a seamless and engaging learning experience that is both fun and informative.

**project live link ** https://herbalitics.streamlit.app/

**🧰 Tech Stack Table**
----------------------

| Technology | Description |
| --- | --- |
|Python	| Primary programming language |
| Streamlit | Web-based interface for building and deploying machine learning models |
| Sentence Transformers | Natural language processing library for sentence embeddings |
| FAISS | Efficient similarity search library for large-scale datasets |
| Numpy | Library for efficient numerical computation |
| Pandas | Library for efficient data manipulation and analysis |
| Scikit-learn | Machine learning library for classification, regression, and clustering |
| Docker | Containerization platform for deploying and managing applications |
| LLM |	Generates fluent responses from retrieved context |
| dotenv |Securely manages API keys from .env file |
|Git LFS	| Manages large files like embeddings and index|

-----------------------
**📁 Project Structure**
```
Project Structure
Herbalitics/
│
├─ Home.py                     # Main Streamlit app (remedy finder with dosha integration)
├─ pages/
│   └─ Learning_Hub.py         # Fun Facts, Glossary, and Challenge modes
├─ .gitignore
├─ .gitattributes              # Git LFS file tracking
├─ requirements.txt            # Project dependencies
├─ chunks.json                 # Text chunks used for semantic retrieval
├─ chunk_sources.json          # Metadata for text chunks
├─ embeddings.npy              # Precomputed embeddings for text chunks
├─ faiss_index.index           # FAISS index for efficient search
└─ plant_images/               # Local images of Ayurvedic plants
```

The project consists of two main files: `Home.py` and `Learning_Hub.py`, which contain the main logic for the platform. The `streamlit_app.py` file is used to build and deploy the web-based interface using Streamlit. The `requirements.txt` file specifies the dependencies required to run the platform.

**⚙️ How to Run**
----------------

Getting Started

Clone the Repository
'''
git clone https://github.com/plp940/Herbalitics.git
cd Herbalitics
'''

Set up Virtual Environment
'''
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows Powershell
'''

Install Dependencies
'''
pip install -r requirements.txt
'''

Configure Environment Variables
'''
Copy .env.example to .env and set your API keys:
'''
llm model api key=your_key
HTTP_REFERER=your_application_url
'''

Run the Application
'''
streamlit run Home.py
'''

How to Use

Remedy Finder (Home.py):

Enter a symptom, plant, or Ayurvedic query.

Take the dosha quiz in the sidebar.

View personalized remedies with plant images and sources.

Learning Hub (via sidebar navigation):

Fun Facts: Refresh to learn a new Ayurvedic tidbit.

Glossary: Look up terms like “Vata”, “Panchakarma”.

Challenge Quiz: Test your Ayurveda knowledge with a mini-quiz.


### Deploy

1. Deploy the platform to streamlit by pushing all files to github
2.push the heavy files through github lfs



Notes

Embedding Setup: If you add new data, regenerate embeddings and update embeddings.npy and faiss_index.index.

Git LFS is used for chunky data files — do git lfs install if contributing locally.

Model Caching: The semantic model will be downloaded and cached automatically upon first run.

Contribution

Contributions are welcome! If you'd like to help enhance the Learning Hub, improve plant image recognition, or expand the knowledge base, feel free to open an issue or submit a pull request.
