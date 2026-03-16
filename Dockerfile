FROM python:3.10

# Set up the working directory
WORKDIR /code

# Copy your root requirements file and install the heavy AI libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your project into the server
COPY . /code

# Hugging Face requires APIs to broadcast on port 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
