### General notes
- Use Anaconda Python to create an environment and test your code
- Use Anaconda Power Shell Prompt to create docker container


### Uninstall all Python packages in Windows

```
pip freeze | % {pip uninstall -y $_}
```

### Build the container
- Make sure docker desktop is running

```
 docker build -t housing_fall25 .
```

### Run the container

```
docker run -d --name housing_fall25_deployed -p 8002:80 housing_fall25
````

### Stop and Remove Container to Rebuild

```
docker stop housing_fall25_deployed
docker rm housing_fall25_deployed
docker image rm housing_fall25
````


To push a Docker image to **Docker Hub**, follow these steps:

---

### **1. Log in to Docker Hub**
Before pushing an image, you need to log in to your Docker Hub account:
```bash
docker login
```
- Enter your **Docker Hub username** and **password** when prompted.

---

### **2. Tag the Image**
Docker Hub requires images to be tagged with your Docker Hub username and repository name. Use the following command to tag your image:

```bash
docker tag housing_fall25:latest mkzia/housing_fall25:v1
```

- Replace `<image_name>` with the name of your local image.
- Replace `<tag>` with the version tag (e.g., `latest` or `v1.0`).
- Replace `<dockerhub_username>` with your Docker Hub username.
- Replace `<repository_name>` with the name of the repository on Docker Hub.

**Example:**
If your local image is named `housing_fall25:latest` and your Docker Hub username is `john_doe`, tag it like this:
```bash
docker tag housing_fall25:latest john_doe/housing_fall25:v1
```

---

### **3. Push the Image**
Once the image is tagged, push it to Docker Hub using the following command:
```bash
docker push <dockerhub_username>/<repository_name>:<tag>
```

**Example:**
```bash
docker push john_doe/myapp:latest
```

---

### **4. Verify the Image on Docker Hub**
After the push is complete, you can verify the image by logging into your Docker Hub account and checking the repository.

---

### **Full Example Workflow**
1. Build the Docker image:
   ```bash
   docker build -t myapp:latest .
   ```
2. Tag the image:
   ```bash
   docker tag myapp:latest john_doe/myapp:latest
   ```
3. Log in to Docker Hub:
   ```bash
   docker login
   ```
4. Push the image:
   ```bash
   docker push john_doe/myapp:latest
   ```


### Digital Ocean Commands

```
ssh root@159.65.246.99

```

https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository


```
docker pull mkzia/housing_fall25:v1
```