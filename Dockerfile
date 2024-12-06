# Use a pre-built PySpark Docker image
FROM jupyter/all-spark-notebook

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app