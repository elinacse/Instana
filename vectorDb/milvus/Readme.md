## Running Milvus DB Locally using Docker Container

Pre-requisites and the steps to run the python application rag-groq.py are described below to refer Vector DB Milvus running locally in Docker container.

### Pre-requisites

1. Recommended Python version is 3.11

2. Run OTel data collector as steps described [here](https://www.ibm.com/docs/en/instana-observability/285?topic=technologies-monitoring-llms#installing-otel-data-collector-for-llm-odcl)

3. Run Instana agent locally

### Steps to run python application

1. Create virtual environment and activate it

```bash
$ python3 -m venv .milvus
$ source .milvus/bin/activate
```

2. Install required packages using **requirements.txt** file

```bash
$ pip install -r requirements.txt
```

3. Set environment variable to export the OTel data

```bash
$ export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 
```
4. Run Milvus docker container

If you want to download docker-compose file here is the command. But we arleady downloaded this file, so you can skip it.

```bash
$ wget https://github.com/milvus-io/milvus/releases/download/v2.3.3/milvus-standalone-docker-compose.yml -O docker-compose.yml 
```
Running Docker compose file

```bash
$ docker-compose up -d     
```

Check docker containers are running
```bash
$ docker ps  
```
If you face any error related to the folder permissions, follow below steps.
Create folder with sudo permissions.

```bash
$ sudo mkdir -p /Users/madhutadiparthi/work/gitrepos/ai-observability/milvus/volumes/milvus    
```
Provide the permissions to the folder

```bash
$ sudo chown -R $(whoami) /Users/madhutadiparthi/work/gitrepos/ai-observability/milvus/volumes/milvus     
```

5. Set GROQ_API_KEY

6. Run the application

```bash
$ python3 rag-groq.py      
```