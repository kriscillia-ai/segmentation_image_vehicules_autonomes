FROM python:3.11-bookworm

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=True
# Copy local code to the container image.
ENV APP_HOME=/back-end
WORKDIR $APP_HOME
COPY . ./

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8080}/ || exit 1

CMD streamlit run --server.port $PORT streamlit_app.py