# LinkedIn Job AI/ML Tool: Project README

## Status: Incomplete

---

### Table of Contents

- [Overview](#overview)
- [Design Decisions](#design-decisions)
- [Tools Used](#tools-used)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

---

## Overview

The LinkedIn Job AI/ML Tool aims to provide an optimized, data-driven approach to job searching on LinkedIn. Utilizing Machine Learning algorithms, Natural Language Processing, and Web Scraping, the tool intends to analyze job descriptions, skills required, and more, to match the user with the most fitting opportunities.

### Features (Work in Progress)

- Scrape job postings for keywords related to skills, industries, and more.
- NLP-based sentiment analysis to gauge company culture from reviews and postings.
- Match score algorithm to rate how well a user's profile aligns with the job description.

## Design Decisions

### 1. Data Source - Web Scraping vs LinkedIn API

**Decision**: Web Scraping 

While LinkedIn API offers a more stable data collection method, it has limitations on the kind of data you can pull and is subject to rate limits. Web scraping provides a broader scope of data access, although it might present legal and ethical challenges.

### 2. Machine Learning Model - Classification vs Clustering

**Decision**: Classification

The focus of this tool is to match job seekers with postings rather than clustering job postings into categories. Thus, a classification approach is more suitable.

### 3. Back-end Language - Python vs Java

**Decision**: Python

Python offers a wide range of libraries for machine learning and data manipulation, making it easier to implement the algorithms required for this project.

### 4. Frontend Framework - React.js vs Angular

**Decision**: React.js

Given the need for a modern, aesthetic, and symmetric UI as per the industry standards, React.js offers more flexibility and a large community of developers which makes it a clear choice for this project.

## Tools Used

- **Web Scraping**: BeautifulSoup, Selenium
- **Machine Learning**: scikit-learn, TensorFlow
- **Natural Language Processing**: NLTK, spaCy
- **Backend**: Python with Flask
- **Frontend**: React.js
- **Database**: MongoDB
- **Version Control**: Git
- **Containerization**: Docker

## Contributing

This project is currently in development and open for contributions. Please refer to the `CONTRIBUTING.md` file for guidelines.

## Contact Information

For more details, please contact the repository owner or raise an issue.

---
