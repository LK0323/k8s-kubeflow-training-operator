# PyTorch 训练 （PyTorchJob）

### pytorch-simple.yaml

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: pytorch-simple
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
              imagePullPolicy: Always
              command:
                - "python3"
                - "/opt/pytorch-mnist/mnist.py"
                - "--epochs=1"
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
              imagePullPolicy: Always
              command:
                - "python3"
                - "/opt/pytorch-mnist/mnist.py"
                - "--epochs=1"
```

### 执行命令：

```sh
kubectl create -f https://raw.githubusercontent.com/kubeflow/training-operator/master/examples/pytorch/simple.yaml
```

### 镜像名称：

```yaml
image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
```

### 执行命令为 Python 脚本：python3 /opt/pytorch-mnist/mnist.py --epochs=1

```python
from __future__ import print_function

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DistributedSampler
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Attach tensors to the device.
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Attach tensors to the device.
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar("accuracy", float(correct) / len(test_loader.dataset), epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Attach model to the device.
    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    # Get FashionMNIST train and test dataset.
    train_ds = datasets.FashionMNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_ds = datasets.FashionMNIST(
        "../data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    # Add train and test loaders.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_ds),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        sampler=DistributedSampler(test_ds),
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        test(model, device, test_loader, writer, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
```

### 数据集：

MNIST 数据集，是一个手写数字图像数据集，由三部分组成：

- **训练集 (training set)**：包含 55,000 张手写数字图像及其对应的标签。
- **验证集 (validation set)**：包含 5,000 张手写数字图像及其对应的标签。
- **测试集 (test set)**：包含 10,000 张手写数字图像及其对应的标签。

使用 PyTorch 的 `DistributedSampler` 模块下载MNIST 数据集：

```python
from torch.utils.data import DistributedSampler
```

### 训练代码位置：

该 PyTorchJob包含一个 Master 容器和一个 Worker 容器。每个容器都使用`docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727` 镜像，并且在失败时都有相同的重启策略。容器的命令是执行 Python3 解释器并运行 `/opt/pytorch-mnist/mnist.py` 脚本。

```sh
[root@master ~]# PODNAME=$(kubectl get pods -l training.kubeflow.org/job-name=pytorch-simple,training.kubeflow.org/replica-type=master,training.kubeflow.org/replica-index=0 -o name -n kubeflow)
[root@master ~]# kubectl logs -f ${PODNAME} -n kubeflow
Using distributed PyTorch with gloo backend
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Processing...
Done!
2024-06-21T04:55:09Z INFO     Train Epoch: 1 [0/60000 (0%)]     loss=2.2975
2024-06-21T04:55:13Z INFO     Train Epoch: 1 [640/60000 (1%)]   loss=2.2965
2024-06-21T04:55:16Z INFO     Train Epoch: 1 [1280/60000 (2%)]  loss=2.2948
2024-06-21T04:55:20Z INFO     Train Epoch: 1 [1920/60000 (3%)]  loss=2.2833
2024-06-21T04:55:23Z INFO     Train Epoch: 1 [2560/60000 (4%)]  loss=2.2622
2024-06-21T04:55:26Z INFO     Train Epoch: 1 [3200/60000 (5%)]  loss=2.2193
2024-06-21T04:55:30Z INFO     Train Epoch: 1 [3840/60000 (6%)]  loss=2.2353
2024-06-21T04:55:34Z INFO     Train Epoch: 1 [4480/60000 (7%)]  loss=2.2295
2024-06-21T04:55:37Z INFO     Train Epoch: 1 [5120/60000 (9%)]  loss=2.1790
2024-06-21T04:55:40Z INFO     Train Epoch: 1 [5760/60000 (10%)] loss=2.1150
2024-06-21T04:55:43Z INFO     Train Epoch: 1 [6400/60000 (11%)] loss=2.0294
2024-06-21T04:55:47Z INFO     Train Epoch: 1 [7040/60000 (12%)] loss=1.9156
2024-06-21T04:55:51Z INFO     Train Epoch: 1 [7680/60000 (13%)] loss=1.7949
2024-06-21T04:55:55Z INFO     Train Epoch: 1 [8320/60000 (14%)] loss=1.5567
2024-06-21T04:55:58Z INFO     Train Epoch: 1 [8960/60000 (15%)] loss=1.3715
2024-06-21T04:56:01Z INFO     Train Epoch: 1 [9600/60000 (16%)] loss=1.3386
2024-06-21T04:56:04Z INFO     Train Epoch: 1 [10240/60000 (17%)]        loss=1.1649
2024-06-21T04:56:07Z INFO     Train Epoch: 1 [10880/60000 (18%)]        loss=1.0924
2024-06-21T04:56:10Z INFO     Train Epoch: 1 [11520/60000 (19%)]        loss=1.0665
2024-06-21T04:56:14Z INFO     Train Epoch: 1 [12160/60000 (20%)]        loss=1.0488
2024-06-21T04:56:17Z INFO     Train Epoch: 1 [12800/60000 (21%)]        loss=1.3654
2024-06-21T04:56:21Z INFO     Train Epoch: 1 [13440/60000 (22%)]        loss=1.0043
2024-06-21T04:56:24Z INFO     Train Epoch: 1 [14080/60000 (23%)]        loss=0.9411
2024-06-21T04:56:27Z INFO     Train Epoch: 1 [14720/60000 (25%)]        loss=0.8942
2024-06-21T04:56:31Z INFO     Train Epoch: 1 [15360/60000 (26%)]        loss=0.9586
2024-06-21T04:56:34Z INFO     Train Epoch: 1 [16000/60000 (27%)]        loss=1.1150
2024-06-21T04:56:38Z INFO     Train Epoch: 1 [16640/60000 (28%)]        loss=1.0944
2024-06-21T04:56:42Z INFO     Train Epoch: 1 [17280/60000 (29%)]        loss=0.8610
2024-06-21T04:56:46Z INFO     Train Epoch: 1 [17920/60000 (30%)]        loss=0.9365
2024-06-21T04:56:49Z INFO     Train Epoch: 1 [18560/60000 (31%)]        loss=0.7595
2024-06-21T04:56:52Z INFO     Train Epoch: 1 [19200/60000 (32%)]        loss=0.8755
2024-06-21T04:56:56Z INFO     Train Epoch: 1 [19840/60000 (33%)]        loss=1.1830
2024-06-21T04:56:59Z INFO     Train Epoch: 1 [20480/60000 (34%)]        loss=0.7637
2024-06-21T04:57:02Z INFO     Train Epoch: 1 [21120/60000 (35%)]        loss=0.8971
2024-06-21T04:57:07Z INFO     Train Epoch: 1 [21760/60000 (36%)]        loss=0.7019
2024-06-21T04:57:10Z INFO     Train Epoch: 1 [22400/60000 (37%)]        loss=0.7468
2024-06-21T04:57:14Z INFO     Train Epoch: 1 [23040/60000 (38%)]        loss=0.8303
2024-06-21T04:57:16Z INFO     Train Epoch: 1 [23680/60000 (39%)]        loss=0.8403
2024-06-21T04:57:20Z INFO     Train Epoch: 1 [24320/60000 (41%)]        loss=0.8833
2024-06-21T04:57:23Z INFO     Train Epoch: 1 [24960/60000 (42%)]        loss=0.8821
2024-06-21T04:57:25Z INFO     Train Epoch: 1 [25600/60000 (43%)]        loss=0.6553
2024-06-21T04:57:28Z INFO     Train Epoch: 1 [26240/60000 (44%)]        loss=0.8553
2024-06-21T04:57:32Z INFO     Train Epoch: 1 [26880/60000 (45%)]        loss=0.8560
2024-06-21T04:57:35Z INFO     Train Epoch: 1 [27520/60000 (46%)]        loss=0.9439
2024-06-21T04:57:38Z INFO     Train Epoch: 1 [28160/60000 (47%)]        loss=0.7415
2024-06-21T04:57:42Z INFO     Train Epoch: 1 [28800/60000 (48%)]        loss=0.8245
2024-06-21T04:57:46Z INFO     Train Epoch: 1 [29440/60000 (49%)]        loss=0.8443
2024-06-21T04:57:49Z INFO     Train Epoch: 1 [30080/60000 (50%)]        loss=0.6781
2024-06-21T04:57:53Z INFO     Train Epoch: 1 [30720/60000 (51%)]        loss=0.9853
2024-06-21T04:57:55Z INFO     Train Epoch: 1 [31360/60000 (52%)]        loss=0.8705
2024-06-21T04:57:59Z INFO     Train Epoch: 1 [32000/60000 (53%)]        loss=0.6735
2024-06-21T04:58:02Z INFO     Train Epoch: 1 [32640/60000 (54%)]        loss=0.7951
2024-06-21T04:58:05Z INFO     Train Epoch: 1 [33280/60000 (55%)]        loss=0.8220
2024-06-21T04:58:09Z INFO     Train Epoch: 1 [33920/60000 (57%)]        loss=0.8706
2024-06-21T04:58:12Z INFO     Train Epoch: 1 [34560/60000 (58%)]        loss=0.9538
2024-06-21T04:58:15Z INFO     Train Epoch: 1 [35200/60000 (59%)]        loss=0.6991
2024-06-21T04:58:18Z INFO     Train Epoch: 1 [35840/60000 (60%)]        loss=0.7417
2024-06-21T04:58:21Z INFO     Train Epoch: 1 [36480/60000 (61%)]        loss=0.8806
2024-06-21T04:58:25Z INFO     Train Epoch: 1 [37120/60000 (62%)]        loss=0.5654
2024-06-21T04:58:29Z INFO     Train Epoch: 1 [37760/60000 (63%)]        loss=0.8553
2024-06-21T04:58:32Z INFO     Train Epoch: 1 [38400/60000 (64%)]        loss=0.6486
2024-06-21T04:58:36Z INFO     Train Epoch: 1 [39040/60000 (65%)]        loss=0.5933
2024-06-21T04:58:40Z INFO     Train Epoch: 1 [39680/60000 (66%)]        loss=0.5394
2024-06-21T04:58:43Z INFO     Train Epoch: 1 [40320/60000 (67%)]        loss=0.7578
2024-06-21T04:58:46Z INFO     Train Epoch: 1 [40960/60000 (68%)]        loss=0.5938
2024-06-21T04:58:50Z INFO     Train Epoch: 1 [41600/60000 (69%)]        loss=0.7355
2024-06-21T04:58:53Z INFO     Train Epoch: 1 [42240/60000 (70%)]        loss=0.7312
2024-06-21T04:58:57Z INFO     Train Epoch: 1 [42880/60000 (71%)]        loss=0.7593
2024-06-21T04:59:00Z INFO     Train Epoch: 1 [43520/60000 (72%)]        loss=0.7412
2024-06-21T04:59:03Z INFO     Train Epoch: 1 [44160/60000 (74%)]        loss=0.5995
2024-06-21T04:59:07Z INFO     Train Epoch: 1 [44800/60000 (75%)]        loss=0.6418
2024-06-21T04:59:10Z INFO     Train Epoch: 1 [45440/60000 (76%)]        loss=0.8501
2024-06-21T04:59:12Z INFO     Train Epoch: 1 [46080/60000 (77%)]        loss=0.8012
2024-06-21T04:59:15Z INFO     Train Epoch: 1 [46720/60000 (78%)]        loss=0.9049
2024-06-21T04:59:18Z INFO     Train Epoch: 1 [47360/60000 (79%)]        loss=0.5929
2024-06-21T04:59:22Z INFO     Train Epoch: 1 [48000/60000 (80%)]        loss=0.5918
2024-06-21T04:59:25Z INFO     Train Epoch: 1 [48640/60000 (81%)]        loss=0.6389
2024-06-21T04:59:29Z INFO     Train Epoch: 1 [49280/60000 (82%)]        loss=0.5233
2024-06-21T04:59:33Z INFO     Train Epoch: 1 [49920/60000 (83%)]        loss=0.9672
2024-06-21T04:59:37Z INFO     Train Epoch: 1 [50560/60000 (84%)]        loss=0.7550
2024-06-21T04:59:39Z INFO     Train Epoch: 1 [51200/60000 (85%)]        loss=0.6280
2024-06-21T04:59:43Z INFO     Train Epoch: 1 [51840/60000 (86%)]        loss=0.5377
2024-06-21T04:59:46Z INFO     Train Epoch: 1 [52480/60000 (87%)]        loss=0.6016
2024-06-21T04:59:50Z INFO     Train Epoch: 1 [53120/60000 (88%)]        loss=0.4454
2024-06-21T04:59:53Z INFO     Train Epoch: 1 [53760/60000 (90%)]        loss=0.7935
2024-06-21T04:59:57Z INFO     Train Epoch: 1 [54400/60000 (91%)]        loss=0.5740
2024-06-21T05:00:00Z INFO     Train Epoch: 1 [55040/60000 (92%)]        loss=0.6581
2024-06-21T05:00:04Z INFO     Train Epoch: 1 [55680/60000 (93%)]        loss=0.5466
2024-06-21T05:00:06Z INFO     Train Epoch: 1 [56320/60000 (94%)]        loss=0.5859
2024-06-21T05:00:10Z INFO     Train Epoch: 1 [56960/60000 (95%)]        loss=0.5472
2024-06-21T05:00:13Z INFO     Train Epoch: 1 [57600/60000 (96%)]        loss=0.7145
2024-06-21T05:00:17Z INFO     Train Epoch: 1 [58240/60000 (97%)]        loss=0.7311
2024-06-21T05:00:20Z INFO     Train Epoch: 1 [58880/60000 (98%)]        loss=0.8890
2024-06-21T05:00:24Z INFO     Train Epoch: 1 [59520/60000 (99%)]        loss=0.5368
2024-06-21T05:00:35Z INFO     {metricName: accuracy, metricValue: 0.7313};{metricName: loss, metricValue: 0.6649}

[root@master ~]# kubectl get pytorchjobs -n kubeflow
NAME             STATE       AGE
pytorch-simple   Succeeded   35m
[root@master ~]# kubectl get pods -l training.kubeflow.org/job-name=pytorch-simple -n kubeflow
NAME                      READY   STATUS      RESTARTS   AGE
pytorch-simple-master-0   0/1     Completed   0          25m
pytorch-simple-worker-0   0/1     Completed   0          25m

```

### 导出 PyTorchJob 的 YAML 配置文件：

```sh
 kubectl get -o yaml pytorchjobs pytorch-simple -n kubeflow
```

### pytorch.yaml:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  creationTimestamp: "2024-06-21T04:41:27Z"
  generation: 1
  name: pytorch-simple
  namespace: kubeflow
  resourceVersion: "53223"
  uid: 4cfb59e8-e06d-44bf-9936-3a483f4e1729
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - command:
            - python3
            - /opt/pytorch-mnist/mnist.py
            - --epochs=1
            image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
            imagePullPolicy: Always
            name: pytorch
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - command:
            - python3
            - /opt/pytorch-mnist/mnist.py
            - --epochs=1
            image: docker.io/kubeflowkatib/pytorch-mnist:v1beta1-45c5727
            imagePullPolicy: Always
            name: pytorch
status:
  completionTime: "2024-06-21T05:00:37Z"
  conditions:
  - lastTransitionTime: "2024-06-21T04:41:27Z"
    lastUpdateTime: "2024-06-21T04:41:27Z"
    message: PyTorchJob pytorch-simple is created.
    reason: PyTorchJobCreated
    status: "True"
    type: Created
  - lastTransitionTime: "2024-06-21T04:53:39Z"
    lastUpdateTime: "2024-06-21T04:53:39Z"
    message: PyTorchJob pytorch-simple is running.
    reason: PyTorchJobRunning
    status: "False"
    type: Running
  - lastTransitionTime: "2024-06-21T05:00:37Z"
    lastUpdateTime: "2024-06-21T05:00:37Z"
    message: PyTorchJob pytorch-simple is successfully completed.
    reason: PyTorchJobSucceeded
    status: "True"
    type: Succeeded
  replicaStatuses:
    Master:
      selector: training.kubeflow.org/job-name=pytorch-simple,training.kubeflow.org/operator-name=pytorchjob-controller,training.kubeflow.org/replica-type=master
      succeeded: 1
    Worker:
      selector: training.kubeflow.org/job-name=pytorch-simple,training.kubeflow.org/operator-name=pytorchjob-controller,training.kubeflow.org/replica-type=worker
      succeeded: 1
  startTime: "2024-06-21T04:41:27Z"

```

