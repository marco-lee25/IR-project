# Use Elasticsearch official image
FROM docker.elastic.co/elasticsearch/elasticsearch:7.10.2

# Set Elasticsearch as a single-node cluster
ENV discovery.type=single-node

# Install Python
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-* && \
    dnf install -y python3 python3-pip curl && \
    dnf clean all

# Install Python packages
RUN pip3 install elasticsearch requests

# Copy index files
COPY ./database/data/arxiv_index_mapping.json /usr/share/elasticsearch/arxiv_index_mapping.json
COPY ./database/data/arxiv_index_settings.json /usr/share/elasticsearch/arxiv_index_settings.json
COPY ./database/data/arxiv_index_data.json /usr/share/elasticsearch/arxiv_index_data.json
COPY ./database/import_index.py /usr/share/elasticsearch/import_index.py

# Ensure the script is executable
RUN chmod +x /usr/share/elasticsearch/import_index.py

# Start Elasticsearch, wait for it, then import index
CMD ["/bin/bash", "-c", "/usr/local/bin/docker-entrypoint.sh & echo 'Waiting for Elasticsearch to start...' && until curl -s http://localhost:9200 >/dev/null; do sleep 5; done && echo 'Elasticsearch started. Importing index...' && python3 /usr/share/elasticsearch/import_index.py && wait"]