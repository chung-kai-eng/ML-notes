## Different tools for MLOps
![image](https://github.com/chung-kai-eng/ML-notes/assets/54303314/ce7929c4-1b40-4a2f-8677-7a2321149739)

### Why need MLOps?
- Environment: Container (execution environment)
- Data version: the date range of data (e.g. June~March)
- Experiment Tracking
- Model version
- Which model inference on the test data

![image](https://github.com/chung-kai-eng/ML-notes/assets/54303314/1541c085-9695-43a4-b7db-947c5bc36ba3)


### MLflow Tracking [\(Link\)](https://mlflow.org/docs/latest/tracking.html)
- Storage
  - Backend store: record each result
  - Artifact store: artifact, code, file, any object information
- If cooperating with team members,
    1. Use a server machine, run the code on the server. Both backend and artifact store are stored in directory (file system)
    2. Use scenario 4, backend store is a database (e.g. PostgreSQL), and artifact store is object storage server (e.g. S3)



## Experience for system build
- Version Control below 3 parts
  - Data
  - Model
  - Inference data: which version of model inference on the dataset

### Data
- Maybe `.csv` file can be stored about 3 months or remain the file for a period of time. After the period, remove the file.
- If you want to reproduce the result, use `data_meta_data.json` to reload the specific data.
- data folder
  ```bash
  data/
  |--- UUID_1/
  |   |--- data_1.csv (.pickle/.parquet)
  |   |--- data_meta_data.json
  |--- UUID_2/
  |   |--- data_1.csv
  |   |--- data_meta_data.json
  ```
- data_meta_data.json
  ```yaml
  CREATE_TIME: datetime 
  PRODUCT: str
  CONDITION_1: str
  CONDITION_2: str
  ```


### Model (train)
- model folder
  ```bash
  model/
  |--- UUID_1/
  |   |--- XGB/
  |   |   |--- target_1.pkl
  |   |   |--- target_2.pkl
  |   |--- plot/
  |   |--- feature_importance/
  |   |   |--- target_1.csv
  |   |   |--- target_2.csv
  |   |--- model_meta_data.json
  |--- UUID_2/
  |   |--- data_1.csv
  |   |--- meta_data.json
  ```
- model_meta_data
  ```yaml
  DATA_UUID: str 
  condition_1: str
  
  # mlflow setting
  VERSION: str
  MODEL_STAGE: str (e.g. Staging, Production)
  MODEL_TAGS: list[str]
    - tag1: str
    - tag2: str
  ```


### Inference
- inference folder
  ```bash
  predict/
  |--- UUID_1/
  |   |--- XGB/
  |   |   |--- report.csv
  |   |   |--- post_processs_result.csv
  |   |   |--- inference_meta_data.json
  ```
- inference_meta_data.json
  - let you know which data is the testing set, and use which model to inference for the result.  
  ```yaml
  DATA_UUID: str
  MODEL_UUID: str
  ```

### MLflow in 
- Connect different parts using UUID for version control. Use **`MLflow tag`** to set up needed conditions for searching.
- `client.search_experiments(filter_string)`: for experiment searching
  - name: experiment_name
  - tags.<tag_key>
- `client.search_model_versions(filter_string)`: for registered model searching
  - name: model_name
  - tags.<tag_key>
- view more about [client API](https://mlflow.org/docs/latest/python_api/mlflow.client.html)
