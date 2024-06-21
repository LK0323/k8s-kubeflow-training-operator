# PaddlePaddle 训练 (PaddleJob)

### paddlepaddle-simple-cpu.yaml

```yaml
apiVersion: "kubeflow.org/v1"
kind: PaddleJob
metadata:
  name: paddle-simple-cpu
  namespace: kubeflow
spec:
  paddleReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: paddle
              image: registry.baidubce.com/paddlepaddle/paddle:2.4.0rc0-cpu
              command:
                - python
              args:
                - "-m"
                - paddle.distributed.launch
                - "run_check"
              ports:
                - containerPort: 37777
                  name: master
              imagePullPolicy: Always
```

### 执行命令：

```sh
kubectl create -f https://raw.githubusercontent.com/kubeflow/training-operator/master/examples/paddlepaddle/simple-cpu.yaml
```

### 镜像名称：

```yaml
image: registry.baidubce.com/paddlepaddle/paddle:2.4.0rc0-cpu
```

### **执行命令**:

```sh
python -m paddle.distributed.launch run_check
```

通过查找网址https://github.com/kubeflow/training-operator/tree/master/examples/paddlepaddle发现官方并未提供执行的python文件，且也未提供镜像文档或 Dockerfile，无法了解数据集如何挂载和分布式训练模块 `run_check`的具体位置。查看镜像内部的代码结构和配置文件，也未找到：

```sh
[root@master ~]# docker run -it --rm registry.baidubce.com/paddlepaddle/paddle:2.4.0rc0-cpu /bin/bash
λ eed04160a108 /home ls -l
total 0
drwxr-xr-x. 6 root root 52 Aug 17  2022 cmake-3.16.0-Linux-x86_64/
λ eed04160a108 /home ^C
λ eed04160a108 /home cd cmake-3.16.0-Linux-x86_64
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64 ls -l
total 0
drwxr-xr-x. 2 root root 76 Nov 26  2019 bin/
drwxr-xr-x. 3 root root 19 Nov 26  2019 doc/
drwxr-xr-x. 4 root root 30 Nov 26  2019 man/
drwxr-xr-x. 7 root root 84 Nov 26  2019 share/
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64 cd bin
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/bin ls -l
total 71M
-rwxr-xr-x. 1 root root 11M Nov 26  2019 ccmake
-rwxr-xr-x. 1 root root 11M Nov 26  2019 cmake
-rwxr-xr-x. 1 root root 26M Nov 26  2019 cmake-gui
-rwxr-xr-x. 1 root root 12M Nov 26  2019 cpack
-rwxr-xr-x. 1 root root 13M Nov 26  2019 ctest
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/bin ^C
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/bin cd ..
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64 cd man
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man ls -l
total 4.0K
drwxr-xr-x. 2 root root   86 Nov 26  2019 man1/
drwxr-xr-x. 2 root root 4.0K Nov 26  2019 man7/
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man cd man1
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man/man1 ls -l
total 136K
-rw-r--r--. 1 root root  12K Nov 26  2019 ccmake.1
-rw-r--r--. 1 root root  34K Nov 26  2019 cmake.1
-rw-r--r--. 1 root root 7.0K Nov 26  2019 cmake-gui.1
-rw-r--r--. 1 root root 9.9K Nov 26  2019 cpack.1
-rw-r--r--. 1 root root  65K Nov 26  2019 ctest.1
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man/man1 cd ..
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man cd man7
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man/man7 ls -l
total 2.1M
-rw-r--r--. 1 root root  40K Nov 26  2019 cmake-buildsystem.7
-rw-r--r--. 1 root root 382K Nov 26  2019 cmake-commands.7
-rw-r--r--. 1 root root  15K Nov 26  2019 cmake-compile-features.7
-rw-r--r--. 1 root root  17K Nov 26  2019 cmake-developer.7
-rw-r--r--. 1 root root  18K Nov 26  2019 cmake-env-variables.7
-rw-r--r--. 1 root root  42K Nov 26  2019 cmake-file-api.7
-rw-r--r--. 1 root root  28K Nov 26  2019 cmake-generator-expressions.7
-rw-r--r--. 1 root root  27K Nov 26  2019 cmake-generators.7
-rw-r--r--. 1 root root  21K Nov 26  2019 cmake-language.7
-rw-r--r--. 1 root root 638K Nov 26  2019 cmake-modules.7
-rw-r--r--. 1 root root  29K Nov 26  2019 cmake-packages.7
-rw-r--r--. 1 root root 141K Nov 26  2019 cmake-policies.7
-rw-r--r--. 1 root root 281K Nov 26  2019 cmake-properties.7
-rw-r--r--. 1 root root  11K Nov 26  2019 cmake-qt.7
-rw-r--r--. 1 root root  25K Nov 26  2019 cmake-server.7
-rw-r--r--. 1 root root  26K Nov 26  2019 cmake-toolchains.7
-rw-r--r--. 1 root root 217K Nov 26  2019 cmake-variables.7
-rw-r--r--. 1 root root 115K Nov 26  2019 cpack-generators.7
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man/man7 cd ..
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/man cd ..
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64 cd doc
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/doc ls -l
total 0
drwxr-xr-x. 11 root root 181 Nov 26  2019 cmake/
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/doc cd ..
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64 cd share
λ eed04160a108 /home/cmake-3.16.0-Linux-x86_64/share ls -l
total 0
drwxr-xr-x. 2 root root  22 Nov 26  2019 aclocal/
drwxr-xr-x. 2 root root  31 Nov 26  2019 applications/
drwxr-xr-x. 9 root root 115 Nov 26  2019 cmake-3.16/
drwxr-xr-x. 3 root root  21 Nov 26  2019 icons/
drwxr-xr-x. 3 root root  22 Nov 26  2019 mime/

```

该PaddleJob包含两个 Worker 副本，每个副本都运行一个名为 `paddle` 的容器。容器使用 `registry.baidubce.com/paddlepaddle/paddle:2.4.0rc0-cpu` 镜像，并运行 `python -m paddle.distributed.launch run_checky` 命令。

```sh
[root@master ~]# kubectl get pod -n kubeflow
NAME                                 READY   STATUS      RESTARTS       AGE
paddle-simple-cpu-worker-0           0/1     Completed   0              81m
paddle-simple-cpu-worker-1           0/1     Completed   0              81m

```

### 导出 PaddleJob的 YAML 配置文件：

```sh
 kubectl get -o yaml paddlejobs paddle-simple-cpu -n kubeflow
```

### paddle.yaml:

```yaml
apiVersion: kubeflow.org/v1
kind: PaddleJob
metadata:
  creationTimestamp: "2024-06-21T00:38:36Z"
  generation: 1
  name: paddle-simple-cpu
  namespace: kubeflow
  resourceVersion: "33328"
  uid: bdb13a0a-3a14-47c8-aafd-f68a25536e3c
spec:
  paddleReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - args:
            - -m
            - paddle.distributed.launch
            - run_check
            command:
            - python
            image: registry.baidubce.com/paddlepaddle/paddle:2.4.0rc0-cpu
            imagePullPolicy: Always
            name: paddle
            ports:
            - containerPort: 37777
              name: master
              protocol: TCP
status:
  completionTime: "2024-06-21T00:47:54Z"
  conditions:
  - lastTransitionTime: "2024-06-21T00:38:36Z"
    lastUpdateTime: "2024-06-21T00:38:36Z"
    message: PaddleJob paddle-simple-cpu is created.
    reason: PaddleJobCreated
    status: "True"
    type: Created
  - lastTransitionTime: "2024-06-21T00:47:19Z"
    lastUpdateTime: "2024-06-21T00:47:19Z"
    message: PaddleJob kubeflow/paddle-simple-cpu is running.
    reason: PaddleJobRunning
    status: "False"
    type: Running
  - lastTransitionTime: "2024-06-21T00:47:54Z"
    lastUpdateTime: "2024-06-21T00:47:54Z"
    message: PaddleJob kubeflow/paddle-simple-cpu successfully completed.
    reason: PaddleJobSucceeded
    status: "True"
    type: Succeeded
  replicaStatuses:
    Worker:
      selector: training.kubeflow.org/job-name=paddle-simple-cpu,training.kubeflow.org/operator-name=paddlejob-controller,training.kubeflow.org/replica-type=worker
      succeeded: 2
  startTime: "2024-06-21T00:38:36Z"

```

