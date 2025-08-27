# ----------------------------
# 1. Base Image
# ----------------------------
FROM python:3.9-slim

# ----------------------------
# 2. Set Working Directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 3. Install Dependencies
# ----------------------------
# Copy requirements first (better layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# 4. Copy Project Files
# ----------------------------
COPY . .

# ----------------------------
# 5. Expose Port (Flask default is 5000, we used 8080 in app.py)
# ----------------------------
EXPOSE 8080

# ----------------------------
# 6. Run the App
# ----------------------------
CMD ["python", "app.py"]
