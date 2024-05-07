# Wind speed regression

Follow these instructions to run the Dash application

1. Build the Docker image
```bash
docker build -t img-CB-LV
```

2. Execute the application
```bash
docker run -h localhost -p 9000:9000 -d --name container-CB-LV img-CB-LV
```