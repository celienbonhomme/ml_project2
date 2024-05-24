# Wind speed regression

Follow these instructions to run the Dash application

1. Build the Docker image
```bash
docker build -t img-cb-lv .
```

2. Execute the application
```bash
docker run -h localhost -p 9000:9000 -d --name container-cb-lv img-cb-lv
```