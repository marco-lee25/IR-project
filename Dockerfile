# Use Elasticsearch official image
FROM docker.elastic.co/elasticsearch/elasticsearch:8.17.3

# Set Elasticsearch as a single-node cluster
ENV discovery.type=single-node
USER root

# Install Python
RUN apt-get update
RUN apt-get install -y python3 python3-pip curl

# Install Python packages
RUN pip3 install elasticsearch requests

# Copy index files
COPY ./database/data/arxiv_index_mapping.json /usr/share/elasticsearch/arxiv_index_mapping.json
COPY ./database/data/arxiv_index_settings.json /usr/share/elasticsearch/arxiv_index_settings.json
COPY ./database/import_index.py /usr/share/elasticsearch/import_index.py

# Ensure the script is executable
RUN chmod +x /usr/share/elasticsearch/import_index.py

USER elasticsearch

# Start Elasticsearch, wait for it, then import index
# CMD ["/bin/bash", "-c", "/usr/local/bin/docker-entrypoint.sh & echo 'Waiting for Elasticsearch to start...' && until curl -s http://localhost:9200 >/dev/null; do sleep 5; done"]
