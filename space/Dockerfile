# Space Docker para a aplicação Streamlit. O SDK "streamlit" dos Spaces foi
# descontinuado em 2025-04-30; apps Streamlit agora rodam pelo SDK Docker.
FROM python:3.12-slim

# Os Spaces executam o contêiner com o UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
