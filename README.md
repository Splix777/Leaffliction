Project in the works, check back soon for updates.

To run Tensorflow on docker with GPU support, use the following command:

```bash
docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python
```

To run Tensorflow on docker with CPU support, use the following command:

```bash
docker run -it tensorflow/tensorflow bash
```
