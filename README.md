# Loan Qualification AI

A Python AI system that predicts loan eligibility and default risk.

## Quick Start

1. Install: `pip install pandas scikit-learn numpy joblib`
2. Add datasets to `archive_folder/`
3. Run: `python models/train_model.py`
4. Predict: `python app/main.py`

#🧱 Architecture


The system primarily follows: Data-Centric Microservices Architecture with Modular Monolith characteristics
This architecture is particularly well-suited for ML systems because it:

1. Handles complex data processing workflows

2. Supports multiple data sources and formats

3. Enables independent development of ML components

4. Facilitates testing and maintenance

5. Allows evolutionary architecture toward microservices
   
<img width="1024" height="1024" alt="Gemini_Generated_Image_gppeo8gppeo8gppe" src="https://github.com/user-attachments/assets/d9d3c2d5-37ab-4f1a-9fc9-e65ea32d1126" />

Architecture Standards 
| **Area**             | **Standards & Protocols**                                  | **Tools & Technologies**                    |
|----------------------|-----------------------------------------------------------|---------------------------------------------|
| **Data Ingestion**   | REST API, Batch File Processing, Change Data Capture (CDC) | Python Pandas, FastAPI, Apache Airflow     |
| **Data Storage**     | Schema-on-Read, Columnar Formats                           | CSV, Parquet, PostgreSQL                   |
| **Machine Learning Pipeline** | CRISP-DM, MLOps Best Practices                         | Scikit-learn, MLflow, Joblib               |
| **API Design**       | RESTful Principles, OpenAPI Specification                  | FastAPI, Swagger, JWT Authentication        |
| **Monitoring**       | Logging, Metrics, Alerting                                 | Python Logging, Prometheus, Grafana        |
| **Deployment**       | Containerization, Continuous Integration/Continuous Deployment (CI/CD) | Docker, GitHub Actions, AWS                |


Architecture Classification:
| Aspect                   |       System's Approach                       | Architecture Pattern               |
|--------------------------|----------------------------------------------|------------------------------------|
| Data Flow                | Sequential pipeline processing                | Pipeline Architecture               |
| Component Organization    | Loosely coupled modules                      | Microservices-inspired              |
| Deployment               | Single codebase with multiple entry points   | Modular Monolith                    |
| Data Handling            | Centralized data processing                   | Data-Centric Architecture           |
| ML Lifecycle             | Structured training followed by serving workflow | MLOps Pipeline Architecture         |

Architecture Evolution Path
Current State: Modular Pipeline Architecture

![unnamed](https://github.com/user-attachments/assets/c0ef42ed-8c73-4de1-b816-649f1b63a94d)

Future State: Event-Driven Microservices

<img width="1024" height="1024" alt="Gemini_Generated_Image_7uboo77uboo77ubo" src="https://github.com/user-attachments/assets/b8eff1ea-92d9-41dd-9edb-779b67ab8ae9" />


## Executive Summary
This enterprise-grade system leverages ensemble machine learning to transform raw financial data into actionable loan qualification insights. The platform processes multiple data sources, trains predictive models, and provides real-time risk assessments through an intuitive interface.

## Features
- Loan default prediction
- Risk assessment
- Interactive interface
- Model persistence
