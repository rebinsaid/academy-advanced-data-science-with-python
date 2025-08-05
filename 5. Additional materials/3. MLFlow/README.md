# Hackathon

We have access to a remote tracking server deployed on a azure kubernetes cluster and we can use it to log our experiments and display a leaderboard.

*This assumes that you have the azure cli installed and configured, else please contact @nitin for instructions*

To start the service, run the following command:

```bash
az aks start --name adswpmlflow --resource-group xd-academy-mlflow-server 
```

To retrieve the IP address of the service, run the following command:

```bash
kubectl get service mlflow-service --watch
```

Please remember to stop the service after the course.

## In class

```python
import os
import mlflow


# Fill in your name below. This will make sure that 
# whatever you log to mlflow will be associated to you
your_name = "your_name_here"

# mlflow config
os.environ["LOGNAME"] = your_name

# we set the tracking server uri to the remote server
mlflow.set_tracking_uri(uri="<remote_tracking_uri>")
```

## Setup

Refer to the [MLOPS workshop](https://github.com/godatadriven/mlops-workshop/tree/main/infra) for details regarding the infrastructure setup.