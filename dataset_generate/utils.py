import numpy as np
import os
import ipdb

candidate = [
    ["conv2d", 1, (14, 14), 512, 256, (3, 3), (2, 2), (1, 1, 1, 1)],
    ["conv2d", 1, (28, 28), 256, 128, (3, 3), (2, 2), (1, 1, 1, 1)],
    ["conv2d", 1, (56, 56), 128, 64, (3, 3), (2, 2), (1, 1, 1, 1)],
    ["conv2d", 1, (224, 224), 64, 3, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["winograd", 1, (56, 56), 64, 64, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["winograd", 1, (28, 28), 128, 128, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["winograd", 1, (14, 14), 256, 256, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["winograd", 1, (7, 7), 512, 512, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["conv2d", 1, (224, 224), 32, 3, (3, 3), (2, 2), (1, 1, 1, 1)],
    ["conv2d", 1, (112, 112), 32, 32, (1, 1), (1, 1), (0, 0, 0, 0)],
    ["depthwise", 1, (112, 112), 32, 32, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["depthwise", 1, (112, 112), 96, 96, (3, 3), (2, 2), (1, 1, 1, 1)],
    ["depthwise", 1, (56, 56), 144, 144, (3, 3), (1, 1), (1, 1, 1, 1)],
    ["conv2d", 1, (112, 112), 96, 16, (1, 1), (1, 1), (0, 0, 0, 0)],
    ["conv2d", 1, (224, 224), 64, 3, (11, 11), (4, 4), (2, 2, 2, 2)],
    ["winograd", 1, (27, 27), 192, 64, (5, 5), (1, 1), (2, 2, 2, 2)],
]

candidate2 = []
for i in range(len(candidate)):
    temp = []
    for j in range(len(candidate[0])):
        temp.append(str(candidate[i][j]))
    candidate2.append(temp)


def prior_random_sample(cand_index):
    import ast

    # ipdb.set_trace()
    cand = np.array([candidate2[cand_index]])
    length = cand.shape[1]
    result = [0 for _ in range(length)]
    index = [i for i in range(length)]
    np.random.shuffle(index)
    for i in index:
        value = np.random.choice(cand[:, i])
        if i != 0:
            result[i] = ast.literal_eval(value)
        else:
            result[i] = value
        cand = cand[cand[:, i] == value]
    return result


def config(kind, N, size, CO, CI, kernels, strides, padding, layout):
    import tvm
    from tvm import relay
    from tvm import autotvm

    dtype = "float32"
    if layout == "NCHW":
        data_layout = "NCHW"
        kernel_layout = "OIHW"
        if kind == "conv2d" or kind == "winograd":
            data = relay.var("data", shape=(N, CI, *size))
            kernel = relay.var("kernel", shape=(CO, CI, *kernels))
            kernel_shape = (CO, CI, *kernels)
        elif kind == "depthwise":
            data = relay.var("data", shape=(N, CI, *size))
            kernel = relay.var("kernel", shape=(CO, 1, *kernels))
            kernel_shape = (CO, 1, *kernels)
    elif layout == "NHWC":
        data_layout = "NHWC"

        if kind == "conv2d" or kind == "winograd":
            kernel_layout = "HWIO"
            data = relay.var("data", shape=(N, *size, CI))
            kernel = relay.var("kernel", shape=(*kernels, CI, CO))
            kernel_shape = (*kernels, CI, CO)
        elif kind == "depthwise":
            kernel_layout = "HWOI"
            data = relay.var("data", shape=(N, *size, CI))
            kernel = relay.var("kernel", shape=(*kernels, CO, 1))
            kernel_shape = (*kernels, CO, 1)

    if kind == "conv2d" or kind == "winograd":
        dilation = (1, 1)
        out = relay.nn.conv2d(
            data,
            kernel,
            strides=strides,
            padding=padding,
            dilation=dilation,
            channels=CO,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            kernel_size=(*kernels,),
            out_dtype=dtype,
        )
        op = "conv2d"
    elif kind == "depthwise":
        assert CO == CI  # depth-wise
        dilation = (1, 1)
        out = relay.nn.conv2d(
            data,
            kernel,
            groups=CO,
            strides=strides,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            padding=padding,
            dilation=dilation,
            channels=CO,
            kernel_size=(*kernels,),
            out_dtype=dtype,
        )
        op = "conv2d"
    ctx = tvm.gpu()
    mod = tvm.IRModule.from_expr(out)
    kernel_weights = tvm.nd.array(np.ones(kernel_shape, dtype=dtype), ctx)
    dict_params = {"kernel": kernel_weights}
    task = autotvm.task.extract_from_program(
        mod["main"], target="cuda", params=dict_params, ops=(relay.op.get(f"nn.{op}"),)
    )
    if layout == "NCHW":
        if kind == "depthwise":
            return task[0]
        elif kind == "conv2d":
            return task[0]
        elif kind == "winograd":
            return task[1]
    elif layout == "NHWC":
        return task
    return task


def get_random_data(path, iteration=10, sample_random=False, batch=1, layout="NCHW"):
    results = []
    os.makedirs(path, exist_ok=True)
    log = []
    for i in range(iteration):
        try:
            if sample_random:
                kind, N, size, CO, CI, kernels, strides, padding = candidate[
                    np.random.choice(range(len(candidate)))
                ]
            else:
                kind, N, size, CO, CI, kernels, strides, padding = prior_random_sample(i)
            if layout == "NCHW":
                task = config(kind, batch, size, CO, CI, kernels, strides, padding, layout)
                results.append(task)
                log.append([kind, batch, size, CO, CI, kernels, strides, padding, layout])
            elif layout == "NHWC":
                tasks = config(kind, batch, size, CO, CI, kernels, strides, padding, layout)
                for task in tasks:
                    results.append(task)
                    log.append([task.name, batch, size, CO, CI, kernels, strides, padding, layout])
        except Exception as e:
            print(e)
    import pandas as pd

    df = pd.DataFrame(log)
    df.to_csv(f"{path}/random_data_log.csv")
    print(results)
    # breakpoint()
    return results
