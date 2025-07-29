# 🎥 Hybrid Movie Recommendation System 🍿

Welcome to the **Hybrid Movie Recommendation System**! 🚀 This project is a sleek **Streamlit web application** designed to suggest movies 🎬 tailored to your tastes using a powerful hybrid approach that combines **content-based** and **collaborative filtering** techniques. Say goodbye to endless scrolling and hello to personalized movie nights! 🌟

---

## 🌟 Overview

In today's fast-paced world, recommendation systems are like your personal movie concierge! 🧑‍💼 They save time by curating content that matches your preferences, sparing you the hassle of sifting through countless options. This project uses **Artificial Intelligence** 🤖 to analyze user profiles, browsing history, and preferences of similar users to deliver spot-on movie recommendations.

The hybrid model blends the best of both worlds, overcoming the limitations of individual approaches to provide diverse and accurate suggestions. 🎯

---

## 🎯 Types of Recommendation Systems

| **Type**                | **Description**                                                                 | **Examples**         | **Challenges**                                                                 |
|-------------------------|---------------------------------------------------------------------------------|----------------------|-------------------------------------------------------------------------------|
| **Content-Based** 📋     | Recommends items based on their attributes and user preferences. Creates vectors for features like genres or actors. | YouTube, Spotify 🎵 | Over-specialization: may miss diverse recommendations outside known categories. |
| **Collaborative Filtering** 🤝 | Uses user-item interactions (e.g., ratings) to find similar users and recommend their liked items. | Amazon, Netflix 📺 | Computationally expensive; favors popular items, may overlook new content.      |
| **Hybrid-Based** 🔄     | Combines content-based and collaborative filtering for better accuracy and diversity. Uses embeddings like word2vec. | Modern platforms   | More complex but mitigates issues of both approaches.                         |

---

## 🎬 About This Project

This **Streamlit web app** 🌐 recommends movies based on your interests using a hybrid recommendation system. It leverages **cosine similarity** to measure how similar movies are, ensuring you get recommendations that hit the mark! 🎯

### 🎥 Demo
👉 [Click here to try the app live!](#) *(Update with actual demo link)*

### 📊 Dataset
The dataset powering this project can be found [here](#) *(Update with actual dataset link)*.

### 🛠️ Model Details
- **Core Concept**: Cosine Similarity 📐
  - Measures similarity between movie feature vectors.
  - **Range**: [0, 1]
    - **0**: Movies are completely different. 😕
    - **1**: Movies are identical. 🎉
  - Vectors are created from movie features (e.g., genres, tags) using embeddings.
  - Learn more: [Cosine Similarity Glossary](https://www.learndatasci.com/glossary/cosine-similarity/)

- **Model File**: `model.pkl`, generated via the `Movie Recommender System Data Analysis.ipynb` notebook.

---

## 🚀 How to Run

Get this project up and running in just a few steps! 🛠️

### 📋 Prerequisites
- Python 3.7.10 🐍
- Conda (for environment management) 🧪
- Git 🌳

### 🛠️ Steps to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mohamed-Teba/Hybrid_Movie_Recommendation_System.git
   cd Hybrid_Movie_Recommendation_System
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create -n movie python=3.7.10 -y
   conda activate movie
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the Model**:
   - Open and run the Jupyter notebook:
     ```bash
     jupyter notebook "Movie Recommender System Data Analysis.ipynb"
     ```
   - Follow the notebook to create the `model.pkl` file.

5. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

6. **Enjoy the App**:
   - Open your browser and go to `http://localhost:8501` to explore movie recommendations! 🎉

---

## 📂 Project Structure

| **File/Folder**                              | **Description**                                                                 |
|----------------------------------------------|--------------------------------------------------------------------------------|
| `app.py`                                     | Main Streamlit app file to run the web interface.                              |
| `Movie Recommender System Data Analysis.ipynb` | Jupyter notebook for data analysis and model creation.                        |
| `model.pkl`                                  | Pre-trained model for movie recommendations.                                   |
| `requirements.txt`                           | List of required Python packages.                                             |
| `README.md`                                  | You're reading it! Project documentation with setup instructions.              |

---

## 🌈 Future Improvements

- 🗳️ Add real-time user feedback to refine recommendations.
- 📈 Incorporate more features like user demographics or detailed movie metadata.
- ⚡ Optimize for scalability to handle larger datasets.

---

## 🙌 Acknowledgments

This project builds on the foundation laid by [entbappy's Movie Recommender System](https://github.com/entbappy/Movie-Recommender-System-Using-Machine-Learning), with enhancements for a hybrid approach. Big thanks! 🙏

---

## 📜 Footer
© 2025 GitHub, Inc. All rights reserved.