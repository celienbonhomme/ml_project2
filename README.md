# Wind speed regression

Follow these instructions to run the Dash application

1. Import the Docker image
```bash
docker image import archive_name.tar image_name
```

2. Execute the application
```bash
docker run -h localhost -p 9000:9000 -d --name container_name image_name
```