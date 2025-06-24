FROM python:3.10


COPY . /

# Create and change to the app directory.
WORKDIR /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

RUN chmod 444 main.py
RUN chmod 444 requirements.txt

ENV PORT 8081

# Run the web service on container startup.
# CMD ["/1.sh"]
# CMD [ "python", "main.py" ]
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 main:app